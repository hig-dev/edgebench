import numpy as np
from typing import Optional
from hailo_platform import (
    HEF,
    VDevice,
    HailoStreamInterface,
    InferVStreams,
    ConfigureParams,
    InputVStreamParams,
    OutputVStreamParams,
    FormatType,
)

class HailoInterpreter:
    def __init__(self, hef_path: str):
        self.target = VDevice()
        self.hef = HEF(hef_path)
        self.configure_params = ConfigureParams.create_from_hef(hef=self.hef, interface=HailoStreamInterface.PCIe)
        self.network_groups = self.target.configure(self.hef, self.configure_params)
        self.network_group = self.network_groups[0]
        self.network_group_params = self.network_group.create_params()
        self.input_vstreams_params = InputVStreamParams.make(self.network_group, format_type=FormatType.FLOAT32)
        self.output_vstreams_params = OutputVStreamParams.make(self.network_group, format_type=FormatType.FLOAT32)
        self.input_vstream_info = self.hef.get_input_vstream_infos()[0]
        self.output_vstream_info = self.hef.get_output_vstream_infos()[0]
        self.input: Optional[np.ndarray] = None
        self.output: Optional[np.ndarray] = None

    def get_input_shape(self, batch_size=1) -> tuple:
        shape =  self.input_vstream_info.shape
        return (batch_size, *shape)
    
    def get_output_shape(self, batch_size=1) -> tuple:
        shape = self.output_vstream_info.shape
        return (batch_size, *shape)

    def set_input(self, input_data: np.ndarray):
        self.input = input_data.copy()
        if self.input.shape != self.get_input_shape():
            raise ValueError(f"Input shape {self.input.shape} does not match expected shape {self.get_input_shape()}")
        if self.input.dtype != np.float32:
            raise ValueError(f"Input dtype {self.input.dtype} is not float32")

    def get_output(self) -> Optional[np.ndarray]:
        return self.output

    def invoke(self):
        with InferVStreams(self.network_group, self.input_vstreams_params, self.output_vstreams_params) as infer_pipeline:
            input_data = {self.input_vstream_info.name: self.input}
            with self.network_group.activate(self.network_group_params):
                infer_results = infer_pipeline.infer(input_data)
                self.output = infer_results[self.output_vstream_info.name]