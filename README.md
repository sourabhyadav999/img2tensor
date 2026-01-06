# img2tensor

A unified, high-performance utility to convert images into training-ready tensors for **NumPy**, **PyTorch**, and **TensorFlow**, or stream them directly into **TFRecords**.

`img2tensor` handles the standard deep learning data ingestion "papercuts": BGR/RGB swaps, memory layouts (NHWC vs NCHW), and dtype scaling in a single, lightweight function.

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

---

## 🛠 API Reference: `get_tensor()`

### Inputs

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `img_paths` | `str | Path | list` | **Required** | Single path or list of paths to image files. |
| `tensor_type` | `str` | `"numpy"` | Target framework: `"numpy"`, `"pytorch"`, or `"tensorflow"`. |
| `dtype` | `str` | `"float32"` | Target type: `"float32"`, `"float16"`, `"uint8"`. Floats are auto-scaled (1/255). |
| `image_layer` | `str` | `"PIL"` | Backend decoder: `"PIL"` or `"CV2"`. |
| `n_jobs` | `int` | `4` | Number of threads for parallel decoding. |
| `output_format` | `str` | `"tensor"` | `"tensor"` (returns object) or `"tfrecord"` (writes to disk). |
| `tfrecord_path` | `str | Path` | `None` | Required if `output_format='tfrecord'`. |

### Outputs

* **Single Path Input:** Returns a **3D Tensor** ($H, W, C$ for NumPy/TF; $C, H, W$ for PyTorch).
* **List Input:** Returns a **4D Tensor** ($N, H, W, C$ for NumPy/TF; $N, C, H, W$ for PyTorch).
* **TFRecord Mode:** Returns a `dict` with file metadata (path, sample count, dtype).

---

## 🧠 Design Philosophy

### NCHW vs NHWC

One of the most frequent bugs in Computer Vision pipelines is passing the wrong channel layout. `img2tensor` detects your framework and adjusts automatically:
* **PyTorch:** Returns $N \times C \times H \times W$ (and ensures memory is `.contiguous()`).
* **NumPy/TF:** Returns $N \times H \times W \times C$.

### What it Does NOT Do (By Design)
* **No Resizing:** We believe silent resizing is dangerous as it introduces artifacts. If your images are inconsistent sizes, `get_tensor` will throw a `ValueError` identifying the offending file.
* **No Augmentation:** This is a pure loader. Use specialized libraries like `albumentations` or `torchvision` for data manipulation.
* **No Heavy Dependencies:** Using lazy imports, the library won't crash if you don't have TensorFlow installed but only use NumPy or Torch.

## 📄 License
MIT
