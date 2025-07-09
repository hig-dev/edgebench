from base_interpreter import BaseInterpreter
from typing import Optional
import numpy as np
from numpy.typing import DTypeLike
import tvm
from tvm import relay, transform
from tvm.relay.op.contrib.arm_compute_lib import partition_for_arm_compute_lib
import onnx
import time


class TVMInterpreter(BaseInterpreter):
    def __init__(self, model_path: str, use_arm_compute_lib: bool = False):
        print(f"Initializing TVMInterpreter with model at {model_path}.")
        if not model_path.endswith(".onnx"):
            raise ValueError(f"Model path must end with .onnx, got {model_path}")
        self.executor = self.compile_model(model_path, use_arm_compute_lib)
        self.input: Optional[np.ndarray] = None
        self.output: Optional[np.ndarray] = None

    def compile_model(self, model_path: str, use_arm_compute_lib: bool = False):
        onnx_model = onnx.load(model_path)
        self.input_name = onnx_model.graph.input[0].name
        self.input_shape = tuple(dim.dim_value for dim in onnx_model.graph.input[0].type.tensor_type.shape.dim)
        onnx_dtype = onnx_model.graph.input[0].type.tensor_type.elem_type
        self.input_dtype = onnx.helper.tensor_dtype_to_np_dtype(onnx_dtype)
        print(f"Input name: {self.input_name}, Input shape: {self.input_shape}, Input dtype: {self.input_dtype}")
        mod, params = relay.frontend.from_onnx(
            onnx_model, shape={self.input_name: self.input_shape}
        )
            
        if use_arm_compute_lib:
            print("Using ARM Compute Library for partitioning.")
            mod = partition_for_arm_compute_lib(mod)

        target = "llvm -mattr=+neon" if use_arm_compute_lib else "llvm"
        with transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
            executor = relay.build_module.create_executor(
                "graph",
                mod,
                tvm.cpu(0),
                target=target,
                params=params,
            ).evaluate()

        return executor

    def get_input_shape(self) -> tuple[int, ...]:
        return self.input_shape

    def get_input_dtype(self) -> DTypeLike:
        return self.input_dtype

    def set_input(self, input_data: np.ndarray):
        if input_data.shape != self.get_input_shape():
            raise ValueError(
                f"Input shape {input_data.shape} does not match expected shape {self.get_input_shape()}"
            )
        if input_data.dtype != self.get_input_dtype():
            raise ValueError(
                f"Input dtype {input_data.dtype} does not match expected dtype {self.get_input_dtype()}"
            )
        self.input = tvm.nd.array(input_data.astype(self.get_input_dtype()))

    def get_output(self) -> Optional[np.ndarray]:
        return self.output.numpy()

    def invoke(self):
        self.output = self.executor(self.input)
