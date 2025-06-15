import ctypes
from typing import Optional
import numpy as np
import os
import shutil
import tempfile


class HHBInterpreter:
    def __init__(self, model_zip_path: str):
        if not os.path.exists(model_zip_path):
            raise FileNotFoundError(f"Model zip file {model_zip_path} does not exist.")
        # Create a temporary directory to extract the model files
        self.temp_dir = tempfile.mkdtemp()
        shutil.unpack_archive(model_zip_path, self.temp_dir)
        model_path = os.path.join(self.temp_dir, "hhb.bm")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file {model_path} does not exist in the unpacked directory."
            )
        model_wrapper_lib_path = os.path.join(self.temp_dir, "model_wrapper.so")
        if not os.path.exists(model_wrapper_lib_path):
            raise FileNotFoundError(
                f"Model wrapper library {model_wrapper_lib_path} does not exist in the unpacked directory."
            )
        self.model_lib = ctypes.CDLL(model_wrapper_lib_path)
        self.session = self.load_model(model_path)
        self.input: Optional[np.ndarray] = None
        self.output: Optional[np.ndarray] = None

    def get_input_shape(self) -> tuple:
        # TODO: Replace with actual input shape retrieval logic
        return (1, 3, 256, 256)
    def get_output_shape(self) -> tuple:
        shape_np: np.ndarray = self.get_output_shape_by_index(0)
        shape_tuple = tuple(shape_np.astype(np.int32))
        return shape_tuple

    def set_input(self, input_data: np.ndarray):
        self.input = input_data.copy()
        if self.input.shape != self.get_input_shape():
            raise ValueError(
                f"Input shape {self.input.shape} does not match expected shape {self.get_input_shape()}"
            )

    def get_output(self) -> Optional[np.ndarray]:
        return self.get_output_by_index(0)

    def invoke(self):
        self.session_run([self.input])

    def load_model(self, model_path):
        self.model_lib.load_model.argtypes = [ctypes.c_char_p]
        self.model_lib.load_model.restype = ctypes.c_void_p

        if not os.path.isabs(model_path):
            raise ValueError("Model path must be an absolute path.")

        model_path = bytes(model_path, encoding="utf8")
        model_path = ctypes.create_string_buffer(model_path, size=(len(model_path) + 1))
        return self.model_lib.load_model(model_path)

    def get_input_number(self):
        self.model_lib.get_input_number.argtypes = [ctypes.c_void_p]
        self.model_lib.get_input_number.restype = ctypes.c_int

        in_num = self.model_lib.get_input_number(self.session)

        return in_num

    def get_output_number(self):
        self.model_lib.get_output_number.argtypes = [ctypes.c_void_p]
        self.model_lib.get_output_number.restype = ctypes.c_int

        out_num = self.model_lib.get_output_number(self.session)

        return out_num

    def get_output_size_by_index(self, index):
        self.model_lib.get_output_size_by_index.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        self.model_lib.get_output_size_by_index.restype = ctypes.c_int

        res = self.model_lib.get_output_size_by_index(self.session, index)
        return res

    def get_output_dim_num_by_index(self, index):
        self.model_lib.get_output_dim_num_by_index.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        self.model_lib.get_output_dim_num_by_index.restype = ctypes.c_int

        res = self.model_lib.get_output_dim_num_by_index(self.session, index)
        return res

    def session_run(self, inputs):
        if not isinstance(inputs, (tuple, list)):
            raise ValueError("Inputs needs: [ndarray, ndarray, ...]")

        actual_in_num = self.get_input_number()
        assert actual_in_num == len(
            inputs
        ), "Actual input number: {}, but get {}".format(actual_in_num, len(inputs))

        self.model_lib.session_run.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int,
        ]
        self.model_lib.session_run.restype = None
        f_ptr = ctypes.POINTER(ctypes.c_float)
        data = (f_ptr * len(inputs))(
            *[single.ctypes.data_as(f_ptr) for single in inputs]
        )

        self.model_lib.session_run(self.session, data, len(inputs))

    def get_output_by_index(self, index):
        out_size = self.get_output_size_by_index(index)

        self.model_lib.get_output_by_index.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.model_lib.get_output_by_index.restype = None

        out = np.zeros(out_size, np.float32)
        self.model_lib.get_output_by_index(
            self.session, index, out.ctypes.data_as(ctypes.c_void_p)
        )

        return out

    def get_output_shape_by_index(self, index):
        out_dim_num = self.get_output_dim_num_by_index(index)

        self.model_lib.get_output_shape_by_index.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.model_lib.get_output_shape_by_index.restype = None

        out = np.zeros(out_dim_num, np.int32)
        self.model_lib.get_output_shape_by_index(
            self.session, index, out.ctypes.data_as(ctypes.c_void_p)
        )

        return out
