import numpy as np
import psutil
import random
import logging
from pathlib import Path
from typing import Union, Tuple, List, Optional
from pqdm.threads import pqdm
import os,io
# Setup Library Logger
logger = logging.getLogger("img_to_tensor")

_MEMORY_THRESHOLD = 0.7 
_BACKEND_CONFIG = {
    "PIL": {
        "nearest": "NEAREST", "bilinear": "BILINEAR", "bicubic": "BICUBIC",
        "area": "BOX", "lanczos": "LANCZOS"
    },
    "OPENCV": {
        "nearest": "INTER_NEAREST", "bilinear": "INTER_LINEAR", "bicubic": "INTER_CUBIC",
        "area": "INTER_AREA", "lanczos": None
    }
}

# --- Internal Core Utilities ---

def _ensure_rgb(img: np.ndarray) -> np.ndarray:
    """Standardizes image to 3-channel RGB (H, W, 3)."""
    if len(img.shape) == 2: return np.stack([img] * 3, axis=-1)
    if img.shape[-1] == 4: return img[..., :3]
    return img

def _normalize_image(img: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "uint8": return img.astype(np.uint8)
    img_f = img.astype(np.float32) / 255.0
    return img_f.astype(np.float16) if dtype == "float16" else img_f

def _apply_lossless_geometry(img: np.ndarray, action: str) -> np.ndarray:
    """Geometric transformations via axis permutations (lossless)."""
    if action == "rot90":  return np.rot90(img, k=1, axes=(0, 1))
    if action == "rot180": return np.rot90(img, k=2, axes=(0, 1))
    if action == "rot270": return np.rot90(img, k=3, axes=(0, 1))
    if action == "flip_lr": return np.fliplr(img)
    if action == "flip_ud": return np.flipud(img)
    return img

def _resize_with_letterbox(img, target_hw, resize_fn, interpolation, fill):
    h, w = img.shape[:2]
    th, tw = target_hw
    scale = min(th / h, tw / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    img = resize_fn(img, (nh, nw), interpolation)
    pad_h, pad_w = th - nh, tw - nw
    top, left, out = pad_h // 2, pad_w // 2, np.full((th, tw, 3), fill, dtype=img.dtype)
    out[top:top+nh, left:left+nw] = img
    return out

class BackendStrategy:
    @staticmethod
    def get_pil():
        from PIL import Image
        def decode(p): return np.asarray(Image.open(p).convert("RGB"))
        def resize(img, size, interp):
            m = getattr(Image, _BACKEND_CONFIG["PIL"][interp])
            return np.asarray(Image.fromarray(img).resize((size[1], size[0]), resample=m))
        return decode, resize

    @staticmethod
    def get_opencv():
        import cv2
        def decode(p):
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None: raise ValueError(f"OpenCV failed to read: {p}")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        def resize(img, size, interp):
            m = getattr(cv2, _BACKEND_CONFIG["OPENCV"][interp])
            return cv2.resize(img, (size[1], size[0]), interpolation=m)
        return decode, resize

# --- Main API ---

def get_tensor(
    img_paths: Union[str, Path, List],
    tensor_type: str = "numpy",
    dtype: str = "float32",
    image_layer: str = "PIL",
    n_jobs: int = 4,
    output_format: str = "tensor",
    tfrecord_path: Optional[str] = None,
    num_shards: int = 1,
    resize: Optional[Tuple[int, int]] = None,
    interpolation: Optional[str] = None,
    preserve_aspect_ratio: bool = False,
    letterbox_color: Tuple[int, int, int] = (0, 0, 0),
    augmentation: Optional[bool] = None,
    augmentation_seed: Optional[int] = None,
):
    # 1. Normalization & Fail-Fast
    is_single = isinstance(img_paths, (str, Path))
    paths = [img_paths] if is_single else list(img_paths)
    layer = image_layer.upper().replace("CV2", "OPENCV")
    if preserve_aspect_ratio and resize is None: raise ValueError("preserve_aspect_ratio requires resize.")
    if (resize or augmentation) and not interpolation: interpolation = "bicubic"
    
    # 2. Pipeline Setup
    decode_fn, resize_fn = BackendStrategy.get_pil() if layer == "PIL" else BackendStrategy.get_opencv()
    rng = random.Random(augmentation_seed)
    actions = ["none", "rot90", "rot180", "rot270", "flip_lr", "flip_ud"]
    payloads = [(p, rng.choice(actions)) for p in paths]

    def pipeline(item):
        path, action = item
        try:
            img = _ensure_rgb(decode_fn(path))
            if augmentation and action != "none": img = _apply_lossless_geometry(img, action)
            if resize:
                img = _resize_with_letterbox(img, resize, resize_fn, interpolation, letterbox_color) if preserve_aspect_ratio else resize_fn(img, resize, interpolation)
            return img
        except Exception as e:
            logger.error(f"Failed {path}: {e}")
            raise

    # 3. TFRecord Sharded Output
    if output_format.lower() == "tfrecord":
        import tensorflow as tf
        if not tfrecord_path: raise ValueError("tfrecord_path required.")
        base_path = Path(tfrecord_path)
        
        # Create writers for each shard
        writers = [tf.io.TFRecordWriter(str(base_path.with_name(f"{base_path.stem}-{i:04d}-of-{num_shards:04d}.tfrecord"))) for i in range(num_shards)]
        
        def process_and_serialize(item):
            img = _normalize_image(pipeline(item), dtype)
            return tf.train.Example(features=tf.train.Features(feature={
                "data": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tobytes()])),
                "shape": tf.train.Feature(int64_list=tf.train.Int64List(value=list(img.shape))),
                "dtype": tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(img.dtype).encode()])),
            })).SerializeToString()

        chunk_size = max(n_jobs * 2, 16)
        for i in range(0, len(payloads), chunk_size):
            chunk = payloads[i : i + chunk_size]
            serialized = pqdm(chunk, process_and_serialize, n_jobs=n_jobs, disable=True)
            for idx, ex in enumerate(serialized):
                writers[(i + idx) % num_shards].write(ex)
        
        for w in writers: w.close()
        return {"status": "success", "shards": num_shards}

    # 4. In-Memory Execution
    sample = pipeline(payloads[0])
    safe_batch = max(1, int((psutil.virtual_memory().available * _MEMORY_THRESHOLD) // sample.nbytes))
    all_chunks = []
    for i in range(0, len(payloads), safe_batch):
        chunk = payloads[i : i + safe_batch]
        all_chunks.append(_normalize_image(np.stack(pqdm(chunk, pipeline, n_jobs=n_jobs), axis=0), dtype))
    
    arr = np.concatenate(all_chunks, axis=0)
    if tensor_type.lower() == "pytorch":
        import torch
        res = torch.from_numpy(arr.transpose(0, 3, 1, 2)).contiguous()
        return res[0] if is_single else res
    return arr[0] if is_single else arr
