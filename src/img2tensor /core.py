import numpy as np
from pathlib import Path
import os
import io
from pqdm.threads import pqdm


# -----------------------------
# Image decoders
# -----------------------------

def _decode_pil(path):
    from PIL import Image
    img = Image.open(path).convert("RGB")
    return np.asarray(img)  # HWC, RGB

def _decode_cv2(path):
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cv2 failed to read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # HWC, RGB

# -----------------------------
# Streaming TFRecord writer
# -----------------------------

def _write_tfrecord_stream(paths, decode_fn, tfrecord_path, dtype, n_jobs):
    import tensorflow as tf
    from pqdm.threads import pqdm

    def to_example(path):
        img = decode_fn(path)

        if dtype == "float32":
            img = img.astype(np.float32) / 255.0
        elif dtype == "float16":
            img = img.astype(np.float16) / 255.0
        elif dtype == "uint8":
            img = img.astype(np.uint8)
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

        features = {
            "data": tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[img.tobytes()])
            ),
            "shape": tf.train.Feature(
                int64_list=tf.train.Int64List(value=list(img.shape))
            ),
            "dtype": tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[str(img.dtype).encode()])
            ),
        }
        example = tf.train.Example(
            features=tf.train.Features(feature=features)
        )
        return example.SerializeToString()

    chunk_size = max(n_jobs * 2, 16)

    with tf.io.TFRecordWriter(str(tfrecord_path)) as writer:
        for i in range(0, len(paths), chunk_size):
            chunk = paths[i : i + chunk_size]
            if len(chunk) > 1 and n_jobs > 1:
                serialized = pqdm(chunk, to_example, n_jobs=n_jobs)
            else:
                serialized = [to_example(p) for p in chunk]

            for ex in serialized:
                writer.write(ex)

# -----------------------------
# Main API
# -----------------------------

def get_tensor(
    img_paths,
    tensor_type="numpy",
    dtype="float32",
    image_layer="PIL",
    n_jobs=4,
    output_format="tensor",
    tfrecord_path=None,
):
    """
    Unified image → tensor / TFRecord utility.

    - tensor output: NumPy / TensorFlow / PyTorch
    - tfrecord output: streaming, memory-bounded
    """

    # ---- Normalize inputs ----
    is_single = isinstance(img_paths, (str, Path))
    paths = [img_paths] if is_single else list(img_paths)

    tensor_type = tensor_type.lower()
    image_layer = image_layer.upper()
    output_format = output_format.lower()

    # ---- Validation ----
    if tensor_type not in ("numpy", "tensorflow", "pytorch"):
        raise ValueError("tensor_type must be 'numpy', 'tensorflow', or 'pytorch'")
    if image_layer not in ("PIL", "CV2"):
        raise ValueError("image_layer must be 'PIL' or 'CV2'")
    if output_format not in ("tensor", "tfrecord"):
        raise ValueError("output_format must be 'tensor' or 'tfrecord'")
    if output_format == "tfrecord" and not tfrecord_path:
        raise ValueError("tfrecord_path is required for tfrecord output")

    decode = _decode_pil if image_layer == "PIL" else _decode_cv2

    # ---- TFRecord path (streaming) ----
    if output_format == "tfrecord":
        _write_tfrecord_stream(paths, decode, tfrecord_path, dtype, n_jobs)
        return {
            "status": "success",
            "path": str(tfrecord_path),
            "count": len(paths),
            "dtype": dtype,
        }

    # ---- In-memory tensor path ----
    if len(paths) > 1 and n_jobs > 1:
        from pqdm.threads import pqdm
        arrays = pqdm(paths, decode, n_jobs=n_jobs)
    else:
        arrays = [decode(p) for p in paths]

    # Enforce consistent shapes
    try:
        arr = np.stack(arrays, axis=0)  # NHWC
    except ValueError as e:
        shapes = {a.shape for a in arrays}
        raise ValueError(
            f"Batch contains inconsistent shapes: {shapes}. "
            "All images must match."
        ) from e

    # ---- Dtype handling ----
    if dtype == "float32":
        arr = arr.astype(np.float32) / 255.0
    elif dtype == "float16":
        arr = arr.astype(np.float16) / 255.0
    elif dtype == "uint8":
        arr = arr.astype(np.uint8)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    # ---- Tensor conversion ----
    if tensor_type == "numpy":
        res = arr
    elif tensor_type == "tensorflow":
        import tensorflow as tf
        res = tf.convert_to_tensor(arr)
    else:  # pytorch
        import torch
        res = torch.from_numpy(arr.transpose(0, 3, 1, 2)).contiguous()

    # ---- Squeeze single image ----
    return res[0] if is_single else res
