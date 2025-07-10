from enum import IntEnum
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from numpy.typing import DTypeLike

class InterpreterType(IntEnum):
    TFLITE = 0
    ORT = 1
    HAILO = 2
    HHB = 3
    DLR = 4
    EXECUTORCH = 5
    TVM = 6

    def model_file_extension(self):
        if self == InterpreterType.TFLITE:
            return ".tflite"
        elif self == InterpreterType.ORT:
            return ".onnx"
        elif self == InterpreterType.HAILO:
            return ".hef"
        elif self == InterpreterType.HHB or self == InterpreterType.DLR:
            return ".zip"
        elif self == InterpreterType.EXECUTORCH:
            return ".pte"
        elif self == InterpreterType.TVM:
            return ".onnx"
        else:
            raise ValueError("Unknown interpreter type")
        
class BaseInterpreter(ABC):
    @abstractmethod
    def get_input_shape(self) -> tuple[int, ...]:
        pass

    @abstractmethod
    def get_input_dtype(self) -> DTypeLike:
        pass

    @abstractmethod
    def set_input(self, input_data: np.ndarray):
        pass

    @abstractmethod
    def get_output(self) -> Optional[np.ndarray]:
        pass

    @abstractmethod
    def invoke(self):
        pass

    def set_input_from_bytes(self, input_bytes: bytes):
        input_shape = self.get_input_shape()
        input_dtype = self.get_input_dtype()
        input_data = np.frombuffer(input_bytes, dtype=input_dtype)
        input_data = input_data.reshape(input_shape)
        self.set_input(input_data)

    def get_output_as_bytes(self) -> Optional[bytes]:
        output_data = self.get_output()
        if output_data is not None:
            return output_data.flatten().tobytes()
        return None
