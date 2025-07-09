import numpy as np
from numpy.typing import DTypeLike
from typing import Optional
from interpreters.base_interpreter import BaseInterpreter
import os
import shutil
import tempfile
from dlr import DLRModel

class DLRInterpreter(BaseInterpreter):
    def __init__(self, model_path: str, input_shape = (1, 256, 256, 3), output_shape = (1, 64, 64, 16)):
        if model_path.endswith('.zip'):
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model zip file {model_path} does not exist.")
            # Create a temporary directory to extract the model files
            self.model_path = tempfile.mkdtemp()
            shutil.unpack_archive(model_path, self.model_path)
        elif os.path.isdir(model_path):
            self.model_path = model_path
        else:
            raise ValueError(f"Invalid model path: {model_path}. It should be a zip file or a directory.")
        self.model = DLRModel(self.model_path, "cpu")
        self.input_name = self.model.get_input_name(0)
        self.output_name = self.model.get_output_name(0)
        print(f"Input name: {self.input_name}, Output Name: {self.output_name}")
        self.input_dtype = self.model.get_input_dtype(0)
        self.output_dtype = self.model.get_output_dtype(0)
        print(f"Input dtype: {self.input_dtype}, Output dtype: {self.output_dtype}")
        self.input_shape = input_shape
        self.output_shape = output_shape
        print(f"Input shape: {self.input_shape}, Output shape: {self.output_shape}")
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

    def get_output(self) -> Optional[np.ndarray]:
        return self.output

    def invoke(self):
        self.output = self.model.run({self.input_name: self.input})
