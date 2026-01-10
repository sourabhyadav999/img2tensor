# img2tensor

A unified, high-performance utility to convert images into training-ready tensors for **NumPy**, **PyTorch**, and **TensorFlow**, or stream them directly into **TFRecords**.

`img2tensor` handles the standard deep learning data ingestion "papercuts": BGR/RGB swaps, memory layouts (NHWC vs NCHW), and dtype scaling in a single, lightweight function.

---


## ✨ Key Features

* **Multi-Framework Support:** Automatic conversion to `np.ndarray`, `torch.Tensor` (NCHW), or `tf.Tensor`.
* **Lossless Augmentations:** Geometric transformations (orthogonal rotations and flips) via pure NumPy axis permutations to avoid interpolation drift.
* **High-Fidelity Resizing:** Support for standard and aspect-ratio-preserving (letterboxed) resizing with synchronized interpolation across PIL and OpenCV backends.
* **Deterministic Parallelism:** Thread-safe execution with per-image seeding to guarantee reproducible results across runs.
* **Automatic Memory Management:** Internal RAM monitoring (70% threshold) to auto-calculate batch sizes and prevent OOM (Out-Of-Memory) crashes.
* **Production Streaming:** Native sharded TFRecord output for massive datasets, enabling parallel I/O during training.

---




## 🚀 Installation

pip install img2tensor


---

## 📖 Usage

### 1. Single Image (In-Memory)
Returns a 3D tensor ($C, H, W$ for PyTorch).

import img2tensor

# Returns: torch.Tensor of shape (3, 224, 224)
tensor = img2tensor.get_tensor("cat.jpg", tensor_type="pytorch")

### 2. Batch Loading (In-Memory)
Returns a 4D tensor ($N, H, W, C$ for NumPy/TF).

# Returns: np.ndarray of shape (32, 224, 224, 3)
batch = img2tensor.get_tensor(list_of_paths, n_jobs=8)

### 3. Production Pipeline (TFRecord)
Writes to disk using a chunked streaming approach to save RAM.

img2tensor.get_tensor(
    img_paths=large_list_of_paths,
    output_format="tfrecord",
    tfrecord_path="dataset.tfrecord",
    n_jobs=12
)


### 4. High-Fidelity Resizing (Letterboxed)
Resize images while maintaining the original aspect ratio using high-quality bicubic interpolation.


import img2tensor

# Returns: torch.Tensor of shape (3, 224, 224)
# Pads with black (default) to keep the original image proportions
tensor = img2tensor.get_tensor(
    "input.jpg", 
    tensor_type="pytorch", 
    resize=(224, 224),
    preserve_aspect_ratio=True
)
---


## 🧠 Resizing and Augmentation Logic

Our `get_tensor` utility implements a **"Quality-First"** approach to data preparation. When features are enabled without specific parameters, the following internal defaults are applied to ensure scientific reproducibility and high signal-to-noise ratios.

### 1. High-Fidelity Resizing
Resizing often involves interpolation, which can introduce artifacts or blurriness if not managed carefully.

* **Default Interpolation (Bicubic):** If `resize` is provided but `interpolation` is `None`, the system defaults to **Bicubic** interpolation. This method uses a $4 \times 4$ pixel neighborhood for calculation, resulting in sharper edges and better detail preservation than the standard Bilinear method.
* **Backend Parity:** The function synchronizes interpolation flags across **PIL** and **OpenCV**. This ensures that "Bicubic" resizing yields numerically consistent results regardless of the underlying decoder.
* **Aspect Ratio Preservation:** When `preserve_aspect_ratio=True` is set, the image is scaled to fit the target dimensions without stretching. Any remaining space is filled using **Letterboxing** with a default `letterbox_color` (black).



### 2. Lossless Geometric Augmentations
Standard rotations (e.g., $15^\circ$) require interpolation that "guesses" new pixel values, creating blur. `img2tensor` enforces a **Lossless Philosophy**.

* **D4 Symmetry Group:** When `augmentation=True` is enabled, the utility randomly selects from bit-perfect orthogonal transformations, including $90^\circ, 180^\circ, 270^\circ$ rotations and horizontal/vertical flips.
* **Pure NumPy Permutations:** These operations are executed using `np.rot90` and `np.flip`. Because these are memory-address rearrangements (swapping axes), they are mathematically lossless—no new pixels are generated and zero information is lost.



### 3. Internal Safety Defaults

| Parameter | Internal Default | Rationale |
| :--- | :--- | :--- |
| **`interpolation`** | `bicubic` | Prioritizes higher image quality for model training over faster, blurrier methods. |
| **`augmentation_seed`** | `None` | If provided, generates a unique but **deterministic** seed per image path to ensure experiments are 100% reproducible. |
| **`Memory Threshold`** | `0.7` | Automatically monitors available RAM and caps usage at **70%** to prevent system-wide OOM (Out-of-Memory) crashes. |
| **Channel Sync** | `RGB` | Automatically replicates 1-channel Grayscale to 3-channels and strips Alpha from RGBA to maintain uniform batch shapes. |

---




## 🛠 API Reference: `get_tensor()`

### Inputs

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `img_paths` | `str \| Path \| list` | **Required** | Single path or list of paths to image files. |
| `tensor_type` | `str` | `"numpy"` | Target framework: `"numpy"`, `"pytorch"`, or `"tensorflow"`. |
| `dtype` | `str` | `"float32"` | Target type: `"float32"`, `"float16"`, `"uint8"`. Floats are auto-scaled (1/255). |
| `image_layer` | `str` | `"PIL"` | Backend decoder: `"PIL"` or `"OpenCV"`. |
| `n_jobs` | `int` | `4` | Number of threads for parallel processing and decoding. |
| `output_format` | `str` | `"tensor"` | `"tensor"` (returns object) or `"tfrecord"` (writes to disk). |
| `tfrecord_path` | `str \| Path` | `None` | Required if `output_format='tfrecord'`. |
| `num_shards` | `int` | `1` | Number of shards to split TFRecord output into. |
| `resize` | `tuple` | `None` | `(H, W)` target size. Defaults to **Bicubic** interpolation if set. |
| `interpolation` | `str` | `None` | `nearest`, `bilinear`, `bicubic`, `area`, or `lanczos` (PIL only). |
| `preserve_aspect_ratio`| `bool` | `False` | Uses **Letterboxing** (padding) to maintain original aspect ratio. |
| `augmentation` | `bool` | `None` | Enables **Lossless** geometric augmentations (D4 symmetry group). |
| `augmentation_angles` | `list` | `[90, 180, 270]` | Specific orthogonal angles to select from when `augmentation=True`. |
| `augmentation_seed` | `int` | `None` | Seed for deterministic and reproducible augmentation results. |

### Outputs

* **Single Path Input:** Returns a **3D Tensor** ($H, W, C$ for NumPy/TF; $C, H, W$ for PyTorch).
* **List Input:** Returns a **4D Tensor** ($N, H, W, C$ for NumPy/TF; $N, C, H, W$ for PyTorch).
* **TFRecord Mode:** Returns a success dictionary containing shard metadata and file counts.

---

## 🧠 Design Philosophy

### NCHW vs NHWC

One of the most frequent bugs in Computer Vision pipelines is passing the wrong channel layout. `img2tensor` detects your framework and adjusts automatically:
* **PyTorch:** Returns $N \times C \times H \times W$ (and ensures memory is `.contiguous()`).
* **NumPy/TF:** Returns $N \times H \times W \times C$.



## 📄 License
MIT
