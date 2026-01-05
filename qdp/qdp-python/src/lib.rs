//
// Licensed to the Apache Software Foundation (ASF) under one or more
// contributor license agreements.  See the NOTICE file distributed with
// this work for additional information regarding copyright ownership.
// The ASF licenses this file to You under the Apache License, Version 2.0
// (the "License"); you may not use this file except in compliance with
// the License.  You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyRuntimeError;
use pyo3::ffi;
use pyo3::prelude::*;
use qdp_core::dlpack::DLManagedTensor;
use qdp_core::{Precision, QdpEngine as CoreEngine};

/// Quantum tensor wrapper implementing DLPack protocol
///
/// This class wraps a GPU-allocated quantum state vector and implements
/// the DLPack protocol for zero-copy integration with PyTorch and other
/// array libraries.
///
/// Example:
///     >>> engine = QdpEngine(device_id=0)
///     >>> qtensor = engine.encode([1.0, 2.0, 3.0], num_qubits=2, encoding_method="amplitude")
///     >>> torch_tensor = torch.from_dlpack(qtensor)
#[pyclass]
struct QuantumTensor {
    ptr: *mut DLManagedTensor,
    consumed: bool,
}

#[pymethods]
impl QuantumTensor {
    /// Implements DLPack protocol - returns PyCapsule for PyTorch
    ///
    /// This method is called by torch.from_dlpack() to get the GPU memory pointer.
    /// The capsule can only be consumed once to prevent double-free errors.
    ///
    /// Args:
    ///     stream: Optional CUDA stream pointer (for DLPack 0.8+)
    ///
    /// Returns:
    ///     PyCapsule containing DLManagedTensor pointer
    ///
    /// Raises:
    ///     RuntimeError: If the tensor has already been consumed
    #[pyo3(signature = (stream=None))]
    fn __dlpack__<'py>(&mut self, py: Python<'py>, stream: Option<i64>) -> PyResult<Py<PyAny>> {
        let _ = stream; // Suppress unused variable warning
        if self.consumed {
            return Err(PyRuntimeError::new_err(
                "DLPack tensor already consumed (can only be used once)",
            ));
        }

        if self.ptr.is_null() {
            return Err(PyRuntimeError::new_err("Invalid DLPack tensor pointer"));
        }

        // Mark as consumed to prevent double-free
        self.consumed = true;

        // Create PyCapsule using FFI
        // PyTorch will call the deleter stored in DLManagedTensor.deleter
        // Use a static C string for the capsule name to avoid lifetime issues
        const DLTENSOR_NAME: &[u8] = b"dltensor\0";

        unsafe {
            // Create PyCapsule without a destructor
            // PyTorch will manually call the deleter from DLManagedTensor
            let capsule_ptr = ffi::PyCapsule_New(
                self.ptr as *mut std::ffi::c_void,
                DLTENSOR_NAME.as_ptr() as *const i8,
                None, // No destructor - PyTorch handles it
            );

            if capsule_ptr.is_null() {
                return Err(PyRuntimeError::new_err("Failed to create PyCapsule"));
            }

            Ok(Py::from_owned_ptr(py, capsule_ptr))
        }
    }

    /// Returns DLPack device information
    ///
    /// Returns:
    ///     Tuple of (device_type, device_id) where device_type=2 for CUDA
    fn __dlpack_device__(&self) -> PyResult<(i32, i32)> {
        if self.ptr.is_null() {
            return Err(PyRuntimeError::new_err("Invalid DLPack tensor pointer"));
        }

        unsafe {
            let tensor = &(*self.ptr).dl_tensor;
            // device_type is an enum, convert to integer
            // kDLCUDA = 2, kDLCPU = 1
            // Ref: https://github.com/dmlc/dlpack/blob/6ea9b3eb64c881f614cd4537f95f0e125a35555c/include/dlpack/dlpack.h#L76-L80
            let device_type = match tensor.device.device_type {
                qdp_core::dlpack::DLDeviceType::kDLCUDA => 2,
                qdp_core::dlpack::DLDeviceType::kDLCPU => 1,
            };
            // Read device_id from DLPack tensor metadata
            Ok((device_type, tensor.device.device_id))
        }
    }
}

impl Drop for QuantumTensor {
    fn drop(&mut self) {
        // Only free if not consumed by __dlpack__
        // If consumed, PyTorch/consumer will call the deleter
        if !self.consumed && !self.ptr.is_null() {
            unsafe {
                // Defensive check: qdp-core always provides a deleter
                debug_assert!(
                    (*self.ptr).deleter.is_some(),
                    "DLManagedTensor from qdp-core should always have a deleter"
                );

                // Call the DLPack deleter to free memory
                if let Some(deleter) = (*self.ptr).deleter {
                    deleter(self.ptr);
                }
            }
        }
    }
}

// Safety: QuantumTensor can be sent between threads
// The DLManagedTensor pointer management is thread-safe via Arc in the deleter
unsafe impl Send for QuantumTensor {}
unsafe impl Sync for QuantumTensor {}

/// Helper to detect PyTorch tensor
fn is_pytorch_tensor(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    let type_obj = obj.get_type();
    let name = type_obj.name()?;
    if name != "Tensor" {
        return Ok(false);
    }
    let module = type_obj.module()?;
    let module_name = module.to_str()?;
    Ok(module_name == "torch")
}

/// Helper to validate tensor
fn validate_tensor(tensor: &Bound<'_, PyAny>) -> PyResult<()> {
    if !is_pytorch_tensor(tensor)? {
        return Err(PyRuntimeError::new_err("Object is not a PyTorch Tensor"));
    }
    Ok(())
}

struct DlpackInputGuard {
    ptr: *mut DLManagedTensor,
}

impl Drop for DlpackInputGuard {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }
        unsafe {
            if let Some(deleter) = (*self.ptr).deleter {
                deleter(self.ptr);
            }
        }
    }
}

/// PyO3 wrapper for QdpEngine
///
/// Provides Python bindings for GPU-accelerated quantum state encoding.
#[pyclass]
struct QdpEngine {
    engine: CoreEngine,
    device_id: usize,
}

#[pymethods]
impl QdpEngine {
    /// Initialize QDP engine on specified GPU device
    ///
    /// Args:
    ///     device_id: CUDA device ID (typically 0)
    ///     precision: Output precision ("float32" default, or "float64")
    ///
    /// Returns:
    ///     QdpEngine instance
    ///
    /// Raises:
    ///     RuntimeError: If CUDA device initialization fails
    #[new]
    #[pyo3(signature = (device_id=0, precision="float32"))]
    fn new(device_id: usize, precision: &str) -> PyResult<Self> {
        let precision = match precision.to_ascii_lowercase().as_str() {
            "float32" | "f32" | "float" => Precision::Float32,
            "float64" | "f64" | "double" => Precision::Float64,
            other => {
                return Err(PyRuntimeError::new_err(format!(
                    "Unsupported precision '{}'. Use 'float32' (default) or 'float64'.",
                    other
                )));
            }
        };

        let engine = CoreEngine::new_with_precision(device_id, precision)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to initialize: {}", e)))?;
        Ok(Self { engine, device_id })
    }

    /// Encode classical data into quantum state
    ///
    /// Args:
    ///     data: Input data as list of floats
    ///     num_qubits: Number of qubits for encoding
    ///     encoding_method: Encoding strategy ("amplitude", "angle", or "basis")
    ///
    /// Returns:
    ///     QuantumTensor: DLPack-compatible tensor for zero-copy PyTorch integration
    ///         Shape: [1, 2^num_qubits]
    ///
    /// Raises:
    ///     RuntimeError: If encoding fails
    ///
    /// Example:
    ///     >>> engine = QdpEngine(device_id=0)
    ///     >>> qtensor = engine.encode([1.0, 2.0, 3.0, 4.0], num_qubits=2, encoding_method="amplitude")
    ///     >>> torch_tensor = torch.from_dlpack(qtensor)
    ///
    /// TODO: Use numpy array input (`PyReadonlyArray1<f64>`) for zero-copy instead of `Vec<f64>`.
    fn encode(
        &self,
        data: Vec<f64>,
        num_qubits: usize,
        encoding_method: &str,
    ) -> PyResult<QuantumTensor> {
        let ptr = self
            .engine
            .encode(&data, num_qubits, encoding_method)
            .map_err(|e| PyRuntimeError::new_err(format!("Encoding failed: {}", e)))?;
        Ok(QuantumTensor {
            ptr,
            consumed: false,
        })
    }

    /// Encode a batch of samples from NumPy array (zero-copy, most efficient)
    ///
    /// Args:
    ///     batch_data: 2D NumPy array of shape [num_samples, sample_size] with dtype float64
    ///     num_qubits: Number of qubits for encoding
    ///     encoding_method: Encoding strategy ("amplitude", "angle", or "basis")
    ///
    /// Returns:
    ///     QuantumTensor: DLPack tensor containing all encoded states
    ///         Shape: [num_samples, 2^num_qubits]
    ///
    /// Example:
    ///     >>> engine = QdpEngine(device_id=0)
    ///     >>> batch = np.random.randn(64, 4).astype(np.float64)
    ///     >>> qtensor = engine.encode_batch(batch, 2, "amplitude")
    ///     >>> torch_tensor = torch.from_dlpack(qtensor)  # Shape: [64, 4]
    fn encode_batch(
        &self,
        batch_data: PyReadonlyArray2<f64>,
        num_qubits: usize,
        encoding_method: &str,
    ) -> PyResult<QuantumTensor> {
        let shape = batch_data.shape();
        let num_samples = shape[0];
        let sample_size = shape[1];

        // Get contiguous slice from numpy array (zero-copy if already contiguous)
        let data_slice = batch_data
            .as_slice()
            .map_err(|_| PyRuntimeError::new_err("NumPy array must be contiguous (C-order)"))?;

        let ptr = self
            .engine
            .encode_batch(
                data_slice,
                num_samples,
                sample_size,
                num_qubits,
                encoding_method,
            )
            .map_err(|e| PyRuntimeError::new_err(format!("Batch encoding failed: {}", e)))?;
        Ok(QuantumTensor {
            ptr,
            consumed: false,
        })
    }

    /// Encode from PyTorch Tensor
    ///
    /// Args:
    ///     tensor: PyTorch Tensor (must be on CPU)
    ///     num_qubits: Number of qubits for encoding
    ///     encoding_method: Encoding strategy
    ///
    /// Returns:
    ///     QuantumTensor: DLPack-compatible tensor
    fn encode_tensor(
        &self,
        tensor: &Bound<'_, PyAny>,
        num_qubits: usize,
        encoding_method: &str,
    ) -> PyResult<QuantumTensor> {
        validate_tensor(tensor)?;

        let device = tensor.getattr("device")?;
        let device_type: String = device.getattr("type")?.extract()?;

        // CUDA path: consume DLPack to get a device pointer (zero-copy).
        if device_type == "cuda" {
            let device_index: Option<i64> = device.getattr("index")?.extract()?;
            if let Some(idx) = device_index {
                if idx < 0 {
                    return Err(PyRuntimeError::new_err("Invalid CUDA device index"));
                }
                if idx as usize != self.device_id {
                    return Err(PyRuntimeError::new_err(format!(
                        "Tensor is on CUDA device {}, but engine was initialized for device {}",
                        idx, self.device_id
                    )));
                }
            }

            let is_contiguous: bool = tensor.call_method0("is_contiguous")?.extract()?;
            if !is_contiguous {
                return Err(PyRuntimeError::new_err(
                    "CUDA tensor must be contiguous for zero-copy encode_tensor()",
                ));
            }

            // Request a DLPack capsule from PyTorch, on the current CUDA stream.
            // This lets PyTorch synchronize producer work onto our consumer stream.
            let py = tensor.py();
            let torch = py
                .import("torch")
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to import torch: {}", e)))?;
            let cuda = torch.getattr("cuda")?;
            let stream_obj = if let Some(idx) = device_index {
                cuda.call_method1("current_stream", (idx,))?
            } else {
                cuda.call_method0("current_stream")?
            };
            let stream_u64: u64 = stream_obj.getattr("cuda_stream")?.extract()?;
            if stream_u64 > i64::MAX as u64 {
                return Err(PyRuntimeError::new_err(
                    "CUDA stream pointer does not fit in i64",
                ));
            }
            let stream_i64 = stream_u64 as i64;

            let capsule = tensor
                .call_method1("__dlpack__", (Some(stream_i64),))
                .map_err(|e| {
                    PyRuntimeError::new_err(format!(
                        "Failed to obtain DLPack capsule from tensor: {}",
                        e
                    ))
                })?;

            // Validate and extract DLManagedTensor* from the capsule.
            let name = c"dltensor";
            let capsule_ptr = capsule.as_ptr();

            let is_valid = unsafe { ffi::PyCapsule_IsValid(capsule_ptr, name.as_ptr()) };
            if is_valid == 0 {
                return Err(PyRuntimeError::new_err(
                    "Invalid DLPack capsule (expected name 'dltensor')",
                ));
            }

            let managed_ptr = unsafe {
                ffi::PyCapsule_GetPointer(capsule_ptr, name.as_ptr()) as *mut DLManagedTensor
            };

            if managed_ptr.is_null() {
                return Err(PyRuntimeError::new_err(
                    "Failed to extract DLManagedTensor pointer from capsule",
                ));
            }

            // Mark capsule as consumed to prevent accidental reuse.
            let used_name = c"used_dltensor";
            unsafe {
                let _ = ffi::PyCapsule_SetName(capsule_ptr, used_name.as_ptr());
            }

            // Ensure deleter is called even on early-return.
            let _guard = DlpackInputGuard { ptr: managed_ptr };

            // Validate DLPack metadata and compute length.
            let (dtype_bits, input_ptr, len) = unsafe {
                let dl = &(*managed_ptr).dl_tensor;

                // Device must be CUDA
                match &dl.device.device_type {
                    qdp_core::dlpack::DLDeviceType::kDLCUDA => {}
                    other => {
                        let other_code = match other {
                            qdp_core::dlpack::DLDeviceType::kDLCPU => 1,
                            qdp_core::dlpack::DLDeviceType::kDLCUDA => 2,
                        };
                        return Err(PyRuntimeError::new_err(format!(
                            "DLPack tensor is not CUDA (device_type={})",
                            other_code
                        )));
                    }
                }

                if dl.device.device_id < 0 {
                    return Err(PyRuntimeError::new_err("Invalid DLPack device_id"));
                }
                if dl.device.device_id as usize != self.device_id {
                    return Err(PyRuntimeError::new_err(format!(
                        "DLPack tensor is on CUDA device {}, but engine was initialized for device {}",
                        dl.device.device_id, self.device_id
                    )));
                }

                if dl.data.is_null() {
                    return Err(PyRuntimeError::new_err("DLPack data pointer is null"));
                }

                // Support float32/float64 input (minimal).
                if dl.dtype.code != qdp_core::dlpack::DL_FLOAT || dl.dtype.lanes != 1 {
                    return Err(PyRuntimeError::new_err(format!(
                        "Unsupported CUDA tensor dtype for encode_tensor: code={}, bits={}, lanes={}",
                        dl.dtype.code, dl.dtype.bits, dl.dtype.lanes
                    )));
                }

                if dl.dtype.bits != 32 && dl.dtype.bits != 64 {
                    return Err(PyRuntimeError::new_err(format!(
                        "Unsupported CUDA tensor dtype bits for encode_tensor: {}",
                        dl.dtype.bits
                    )));
                }

                let ndim = dl.ndim as usize;
                if ndim == 0 || dl.shape.is_null() {
                    return Err(PyRuntimeError::new_err("Invalid DLPack shape metadata"));
                }

                let shape = std::slice::from_raw_parts(dl.shape, ndim);

                // Compute element count (support 1D or contiguous 2D).
                let elem_count: usize = if ndim == 1 {
                    shape[0]
                        .try_into()
                        .map_err(|_| PyRuntimeError::new_err("Negative shape dimension"))?
                } else if ndim == 2 {
                    let rows: usize = shape[0]
                        .try_into()
                        .map_err(|_| PyRuntimeError::new_err("Negative shape dimension"))?;
                    let cols: usize = shape[1]
                        .try_into()
                        .map_err(|_| PyRuntimeError::new_err("Negative shape dimension"))?;
                    // If strides are provided, enforce row-major contiguous.
                    if !dl.strides.is_null() {
                        let strides = std::slice::from_raw_parts(dl.strides, ndim);
                        if strides[1] != 1 || strides[0] != shape[1] {
                            return Err(PyRuntimeError::new_err(
                                "Only row-major contiguous 2D CUDA tensors are supported",
                            ));
                        }
                    }
                    rows.saturating_mul(cols)
                } else {
                    return Err(PyRuntimeError::new_err(
                        "Only 1D or 2D CUDA tensors are supported for encode_tensor",
                    ));
                };

                if elem_count == 0 {
                    return Err(PyRuntimeError::new_err("Input tensor cannot be empty"));
                }

                let byte_offset = dl.byte_offset as usize;
                let elem_size = (dl.dtype.bits as usize) / 8;
                if elem_size == 0 {
                    return Err(PyRuntimeError::new_err("Invalid DLPack dtype size"));
                }
                if byte_offset % elem_size != 0 {
                    return Err(PyRuntimeError::new_err(
                        "DLPack byte_offset is not aligned for tensor dtype",
                    ));
                }

                let base = (dl.data as *const u8).add(byte_offset);
                (dl.dtype.bits, base, elem_count)
            };

            // Encode directly from device pointer on the same CUDA stream.
            let stream_ptr = stream_u64 as *mut std::ffi::c_void;
            let ptr = if dtype_bits == 64 {
                let input_ptr_f64 = input_ptr as *const f64;
                unsafe {
                    self.engine
                        .encode_device_f64_ptr(
                            input_ptr_f64,
                            len,
                            num_qubits,
                            encoding_method,
                            stream_ptr,
                        )
                        .map_err(|e| PyRuntimeError::new_err(format!("Encoding failed: {}", e)))?
                }
            } else {
                let input_ptr_f32 = input_ptr as *const f32;
                unsafe {
                    self.engine
                        .encode_device_f32_ptr(
                            input_ptr_f32,
                            len,
                            num_qubits,
                            encoding_method,
                            stream_ptr,
                        )
                        .map_err(|e| PyRuntimeError::new_err(format!("Encoding failed: {}", e)))?
                }
            };

            return Ok(QuantumTensor {
                ptr,
                consumed: false,
            });
        }

        // CPU path (existing): flatten -> tolist -> Vec<f64>.
        if device_type != "cpu" {
            return Err(PyRuntimeError::new_err(format!(
                "Unsupported tensor device type: {}",
                device_type
            )));
        }

        // NOTE(perf): `tolist()` + `extract()` makes extra copies (Tensor -> Python list -> Vec).
        // TODO: follow-up PR can use `numpy()`/buffer protocol (and possibly pinned host memory)
        // to reduce copy overhead.
        let data: Vec<f64> = tensor
            .call_method0("flatten")?
            .call_method0("tolist")?
            .extract()?;

        let ptr = self
            .engine
            .encode(&data, num_qubits, encoding_method)
            .map_err(|e| PyRuntimeError::new_err(format!("Encoding failed: {}", e)))?;

        Ok(QuantumTensor {
            ptr,
            consumed: false,
        })
    }

    /// Encode from Parquet file
    ///
    /// Args:
    ///     path: Path to Parquet file
    ///     num_qubits: Number of qubits for encoding
    ///     encoding_method: Encoding strategy (currently only "amplitude")
    ///
    /// Returns:
    ///     QuantumTensor: DLPack tensor containing all encoded states
    ///
    /// Example:
    ///     >>> engine = QdpEngine(device_id=0)
    ///     >>> batched = engine.encode_from_parquet("data.parquet", 16, "amplitude")
    ///     >>> torch_tensor = torch.from_dlpack(batched)  # Shape: [200, 65536]
    fn encode_from_parquet(
        &self,
        path: &str,
        num_qubits: usize,
        encoding_method: &str,
    ) -> PyResult<QuantumTensor> {
        let ptr = self
            .engine
            .encode_from_parquet(path, num_qubits, encoding_method)
            .map_err(|e| PyRuntimeError::new_err(format!("Encoding from parquet failed: {}", e)))?;
        Ok(QuantumTensor {
            ptr,
            consumed: false,
        })
    }

    /// Encode from Arrow IPC file
    ///
    /// Args:
    ///     path: Path to Arrow IPC file (.arrow or .feather)
    ///     num_qubits: Number of qubits for encoding
    ///     encoding_method: Encoding strategy (currently only "amplitude")
    ///
    /// Returns:
    ///     QuantumTensor: DLPack tensor containing all encoded states
    ///
    /// Example:
    ///     >>> engine = QdpEngine(device_id=0)
    ///     >>> batched = engine.encode_from_arrow_ipc("data.arrow", 16, "amplitude")
    ///     >>> torch_tensor = torch.from_dlpack(batched)
    fn encode_from_arrow_ipc(
        &self,
        path: &str,
        num_qubits: usize,
        encoding_method: &str,
    ) -> PyResult<QuantumTensor> {
        let ptr = self
            .engine
            .encode_from_arrow_ipc(path, num_qubits, encoding_method)
            .map_err(|e| {
                PyRuntimeError::new_err(format!("Encoding from Arrow IPC failed: {}", e))
            })?;
        Ok(QuantumTensor {
            ptr,
            consumed: false,
        })
    }

    /// Encode from NumPy .npy file
    ///
    /// Args:
    ///     path: Path to NumPy .npy file
    ///     num_qubits: Number of qubits for encoding
    ///     encoding_method: Encoding strategy ("amplitude", "angle", or "basis")
    ///
    /// Returns:
    ///     QuantumTensor: DLPack tensor containing all encoded states
    ///
    /// Example:
    ///     >>> engine = QdpEngine(device_id=0)
    ///     >>> batched = engine.encode_from_numpy("states.npy", 10, "amplitude")
    ///     >>> torch_tensor = torch.from_dlpack(batched)
    fn encode_from_numpy(
        &self,
        path: &str,
        num_qubits: usize,
        encoding_method: &str,
    ) -> PyResult<QuantumTensor> {
        let ptr = self
            .engine
            .encode_from_numpy(path, num_qubits, encoding_method)
            .map_err(|e| PyRuntimeError::new_err(format!("Encoding from NumPy failed: {}", e)))?;
        Ok(QuantumTensor {
            ptr,
            consumed: false,
        })
    }
}

/// Mahout QDP Python module
///
/// GPU-accelerated quantum data encoding with DLPack integration.
#[pymodule]
fn mahout_qdp(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<QdpEngine>()?;
    m.add_class::<QuantumTensor>()?;
    Ok(())
}
