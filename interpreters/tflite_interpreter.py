from typing import Optional

from numpy import ndarray
from numpy.typing import DTypeLike
from interpreters.base_interpreter import BaseInterpreter
from ai_edge_litert.interpreter import Interpreter

class TFLiteInterpreter(BaseInterpreter):
    def __init__(self, model_path: str, num_threads: Optional[int] = None):
        super().__init__(model_path)
        self.interpreter = Interpreter(model_path, num_threads=num_threads)
        self.interpreter.allocate_tensors()
        self.output = None

    def get_input_shape(self) -> tuple[int, ...]:
        return self.interpreter.get_input_details()[0]['shape']
    
    def get_input_dtype(self) -> DTypeLike:
        return self.interpreter.get_input_details()[0]['dtype']

    def set_input(self, input_data: ndarray):
        input_details = self.interpreter.get_input_details()
        self.interpreter.set_tensor(input_details[0]["index"], input_data)

    def get_output(self) -> Optional[ndarray]:
        output_details = self.interpreter.get_output_details()
        return self.interpreter.get_tensor(output_details[0]["index"])

    def invoke(self):
        self.interpreter.invoke()