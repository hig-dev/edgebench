import numpy as np
from numpy.typing import DTypeLike
import torch
from typing import Optional
from executorch.runtime import Runtime
from .base_interpreter import BaseInterpreter

class ExecuTorchInterpreter(BaseInterpreter):
    def __init__(self, model_path: str):
        print(f"Initializing ExecuTorchInterpreter with model at {model_path}.")
        self.runtime = Runtime.get()
        self.program = self.runtime.load_program(model_path)
        self.method = self.program.load_method("forward")
        self.input: Optional[np.ndarray] = None
        self.output: Optional[np.ndarray] = None

    def get_input_shape(self) -> tuple[int, ...]:
        method_meta = self.method.metadata
        tensor_info = method_meta.input_tensor_meta(0)
        shape = tensor_info.sizes()
        return shape
    
    def get_input_dtype(self) -> DTypeLike:
        method_meta = self.method.metadata
        tensor_info = method_meta.input_tensor_meta(0)
        dtype_scalar: int = tensor_info.dtype()
        if dtype_scalar == 0:
            return np.uint8
        elif dtype_scalar == 1:
            return np.int8
        elif dtype_scalar == 2:
            return np.int16
        elif dtype_scalar == 3:
            return np.int32
        elif dtype_scalar == 4:
            return np.int64
        elif dtype_scalar == 6:
            return np.float32
        else:
            raise ValueError(f"Unsupported dtype scalar {dtype_scalar} in ExecuTorchInterpreter.")

    def set_input(self, input_data: np.ndarray):
        self.input = torch.from_numpy(input_data.copy())
        if self.input.shape != self.get_input_shape():
            raise ValueError(f"Input shape {self.input.shape} does not match expected shape {self.get_input_shape()}")

    def get_output(self) -> Optional[np.ndarray]:
        return self.output.detach().numpy()

    def invoke(self):
        self.output = self.method.execute([self.input])[0]
