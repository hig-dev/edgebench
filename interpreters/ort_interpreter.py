from interpreters.base_interpreter import BaseInterpreter
import numpy as np
from numpy.typing import DTypeLike
from typing import Any, Optional, Sequence
import onnxruntime as ort


class ORTInterpreter(BaseInterpreter):
    def __init__(self, onnx_path: str, exection_providers: Sequence[str | tuple[str, dict[Any, Any]]]):
        print(f"Initializing ORTInterpreter with ONNX model at {onnx_path} and execution providers {exection_providers}")
        self.session = ort.InferenceSession(onnx_path, providers=exection_providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = tuple(self.session.get_inputs()[0].shape)
        self.output_shape = tuple(self.session.get_outputs()[0].shape)
        onnx_type = self.session.get_inputs()[0].type
        if 'float' in onnx_type:
            self.input_dtype = np.float32
        elif 'int64' in onnx_type:
            self.input_dtype = np.int64
        elif 'int32' in onnx_type:
            self.input_dtype = np.int32
        elif 'int8' in onnx_type:
            self.input_dtype = np.int8
        elif 'uint8' in onnx_type:
            self.input_dtype = np.uint8
        else:
            raise ValueError(f"Unsupported ONNX input type: {onnx_type}")
        self.input: Optional[np.ndarray] = None
        self.output: Optional[np.ndarray] = None
        print(f"ORTInterpreter initialized with input name: {self.input_name}, output name: {self.output_name}, "
              f"input shape: {self.input_shape}, output shape: {self.output_shape}")

    def get_input_shape(self) -> tuple[int, ...]:
        return self.input_shape
    
    def get_input_dtype(self) -> DTypeLike:
        return self.input_dtype

    def set_input(self, input_data: np.ndarray):
        self.input = input_data.copy()
        if self.input.shape != self.get_input_shape():
            raise ValueError(f"Input shape {self.input.shape} does not match expected shape {self.get_input_shape()}")
        if self.input.dtype != self.get_input_dtype():
            raise ValueError(f"Input dtype {self.input.dtype} does not match expected dtype {self.get_input_dtype()}")

    def get_output(self) -> Optional[np.ndarray]:
        return self.output

    def invoke(self):
        self.output = self.session.run([self.output_name], {self.input_name: self.input})[0]
