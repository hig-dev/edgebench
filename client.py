import argparse
from typing import Optional
import uuid
import time
import paho.mqtt.client as mqtt
import queue
import tempfile

from interpreters.base_interpreter import InterpreterType, BaseInterpreter
from shared import TestMode, ClientStatus, Command, Topic, Logger

BYTE_ORDER = "big"

class EdgeBenchClient:
    def __init__(
        self,
        device_id,
        broker_host="127.0.0.1",
        broker_port=1883,
        threads=1,
        interpreter_type=InterpreterType.TFLITE,
        ort_execution_providers=[],
    ):
        self.device_id = device_id
        self.threads = threads
        self.interpreter_type = interpreter_type
        self.ort_execution_providers = ort_execution_providers
        self.logger = Logger("EdgeBenchClient")
        self.topic = Topic(device_id)
        self.client = mqtt.Client(
            client_id=str(uuid.uuid4()),
            clean_session=True,
            protocol=mqtt.MQTTv311,
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        )
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.iterations = 0
        self.mode = TestMode.NONE
        self.interpreter: Optional[BaseInterpreter] = None
        self.latency_input_bytes = None
        self.message_queue = queue.Queue()

        # Setup callbacks
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

    def on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code == 0:
            self.logger.log("Connected to MQTT broker")
            # Subscribe to topics
            self.client.subscribe(self.topic.CONFIG_MODE(), qos=1)
            self.client.subscribe(self.topic.CONFIG_ITERATIONS(), qos=1)
            self.client.subscribe(self.topic.MODEL(), qos=1)
            self.client.subscribe(self.topic.INPUT_LATENCY(), qos=1)
            self.client.subscribe(self.topic.INPUT_ACCURACY(), qos=1)
            self.client.subscribe(self.topic.CMD(), qos=1)
        else:
            self.logger.log(
                f"Failed to connect to MQTT broker with code: {reason_code}"
            )

    def on_message(self, client, userdata, msg: mqtt.MQTTMessage):
        self.logger.log(f"Received message on topic {msg.topic}")
        self.message_queue.put(msg)

    def connect(self):
        self.logger.log(
            f"Connecting to MQTT broker at {self.broker_host}:{self.broker_port}"
        )
        self.client.connect(self.broker_host, self.broker_port)
        self.client.loop_start()

    def disconnect(self):
        self.client.loop_stop()
        self.client.disconnect()
        self.logger.log("Disconnected from MQTT broker")

    def wait_for_message(self, timeout=None) -> mqtt.MQTTMessage:
        return self.message_queue.get(block=True, timeout=timeout)

    def send_status(self, status: ClientStatus):
        self.client.publish(
            self.topic.STATUS(), status.to_bytes(length=1, byteorder=BYTE_ORDER), qos=1
        )
        self.logger.log(f"Status sent: {status.name}")

    def send_result(self, elapsed_time_ms: int):
        self.client.publish(
            self.topic.RESULT_LATENCY(),
            elapsed_time_ms.to_bytes(length=4, byteorder=BYTE_ORDER),
            qos=1,
        )
        self.logger.log(f"Result sent: {elapsed_time_ms} ms")

    def start_latency_test(self):
        self.logger.log(f"Running {self.iterations} iterations...")
        start_time = time.time()
        for i in range(self.iterations):
            self.interpreter.invoke()
        end_time = time.time()
        self.send_status(ClientStatus.DONE)
        elapsed_time = end_time - start_time
        self.logger.log(
            "Run completed, elapsed time: {:.2f} seconds".format(elapsed_time)
        )
        elapsed_time_ms = int(elapsed_time * 1000)
        self.send_result(elapsed_time_ms)

    def setup_interpreter(self, model_bytes: bytes):
        model_file_type = self.interpreter_type.model_file_extension()
        with tempfile.NamedTemporaryFile(suffix=model_file_type) as temp_file:
            model_path = temp_file.name
            with open(model_path, "wb") as f:
                f.write(model_bytes)
            self.logger.log(f"Model saved to {model_path}")
        
        if self.interpreter_type == InterpreterType.ORT:
            from interpreters.ort_interpreter import ORTInterpreter

            if (
                not self.ort_execution_providers
                or len(self.ort_execution_providers) == 0
            ):
                raise ValueError(
                    "ORT execution providers must be specified for ORT interpreter"
                )

            self.interpreter = ORTInterpreter(
                onnx_path=model_path,
                exection_providers=self.ort_execution_providers,
            )
        elif self.interpreter_type == InterpreterType.DLR:
            from interpreters.dlr_interpreter import DLRInterpreter
            self.interpreter = DLRInterpreter(model_path)
        elif self.interpreter_type == InterpreterType.HAILO:
            from interpreters.hailo_interpreter import HailoInterpreter
            self.interpreter = HailoInterpreter(model_path)
        elif self.interpreter_type == InterpreterType.HHB:
            from interpreters.hhb_interpreter import HHBInterpreter
            self.interpreter = HHBInterpreter(model_path)
        elif self.interpreter_type == InterpreterType.EXECUTORCH:
            from interpreters.executorch_interpreter import ExcecutorchInterpreter
            self.interpreter = ExcecutorchInterpreter(model_path)
        elif self.interpreter_type == InterpreterType.TFLITE:
            from interpreters.tflite_interpreter import TFLiteInterpreter
            self.interpreter = TFLiteInterpreter(
                model_path=model_path, num_threads=self.threads
            )
        else:
            raise ValueError(f"Unsupported interpreter type: {self.interpreter_type}")
        self.logger.log(
            f"Interpreter set up with type: {self.interpreter_type.name}, model path: {model_path}"
        )


    def run(self):
        l = self.logger
        t = self.topic
        sent_ready_for_model = False
        sent_ready_for_input = False
        sent_ready_for_task = False
        self.connect()
        l.log(f"Connected as {self.device_id}")
        self.send_status(ClientStatus.STARTED)

        while True:
            msg = self.wait_for_message()

            topic = msg.topic
            payload = msg.payload

            if topic == t.CONFIG_MODE():
                self.mode = TestMode.from_bytes(payload, byteorder=BYTE_ORDER)
                l.log(f"Mode set: {self.mode.name}")
            elif topic == t.CONFIG_ITERATIONS():
                self.iterations = int.from_bytes(payload, byteorder=BYTE_ORDER)
                l.log(f"Iterations set: {self.iterations}")
            elif topic == t.MODEL():
                model_bytes = payload
                l.log(f"Received model, size: {len(model_bytes)} bytes")
                self.setup_interpreter(model_bytes)
            elif topic == t.INPUT_LATENCY():
                self.latency_input_bytes = payload
                l.log(f"Received input, size: {len(payload)} bytes")
                self.interpreter.set_input_from_bytes(self.latency_input_bytes)

            elif topic == t.INPUT_ACCURACY():
                input_bytes = payload
                l.log(f"Received input, size: {len(input_bytes)} bytes")
                self.interpreter.set_input_from_bytes(input_bytes)
                self.interpreter.invoke()
                output_bytes = self.interpreter.get_output_as_bytes()
                self.client.publish(self.topic.RESULT_ACCURACY(), output_bytes, qos=1)
            elif topic == t.CMD():
                cmd = Command.from_bytes(payload, byteorder=BYTE_ORDER)
                if cmd == Command.START_LATENCY_TEST:
                    l.log("Received START_LATENCY_TEST command")
                    self.start_latency_test()
                elif cmd == Command.STOP:
                    l.log("Received STOP command")
                    self.disconnect()
                    break
                elif cmd == Command.RESET:
                    l.log("Received RESET command")
                    sent_ready_for_model = False
                    sent_ready_for_input = False
                    sent_ready_for_task = False
                    self.iterations = 0
                    self.mode = TestMode.NONE
                    self.interpreter = None
                    self.latency_input_bytes = None
                    continue

            latency_config_ready = self.mode == TestMode.LATENCY and self.iterations > 0
            accuracy_config_ready = self.mode == TestMode.ACCURACY
            config_ready = latency_config_ready or accuracy_config_ready
            interpreter_ready = self.interpreter is not None
            input_ready = (
                self.latency_input_bytes is not None or self.mode == TestMode.ACCURACY
            )

            if not sent_ready_for_model and config_ready:
                sent_ready_for_model = True
                l.log("All config received, requesting model")
                self.send_status(ClientStatus.READY_FOR_MODEL)

            if (
                not sent_ready_for_input
                and config_ready
                and interpreter_ready
                and self.mode == TestMode.LATENCY
            ):
                sent_ready_for_input = True
                l.log("Interpreter ready, waiting for input")
                self.send_status(ClientStatus.READY_FOR_INPUT)

            if (
                not sent_ready_for_task
                and config_ready
                and interpreter_ready
                and input_ready
            ):
                sent_ready_for_task = True
                l.log("Ready for task, waiting for input or command")
                self.send_status(ClientStatus.READY_FOR_TASK)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True, help="Device ID")
    parser.add_argument("--threads", type=int, default=1, help="Number of threads")
    parser.add_argument("--broker-host", default="127.0.0.1", help="MQTT broker host")
    parser.add_argument(
        "--broker-port", type=int, default=1883, help="MQTT broker port"
    )
    parser.add_argument("--hailo", action="store_true", help="Use Hailo device")
    parser.add_argument("--hhb", action="store_true", help="Use HHB")
    parser.add_argument("--executorch", action="store_true", help="Use Excecutorch")
    parser.add_argument("--dlr", action="store_true", help="Use NEO AI DLR Runtime")
    parser.add_argument("--ort", action="store_true", help="Use ONNX Runtime")
    parser.add_argument(
        "--ort-execution-providers",
        nargs="+",
        default=[],
        help="ONNX Runtime execution providers (e.g., CPUExecutionProvider, ShlExecutionProvider)",
    )
    args = parser.parse_args()
    if args.hailo:
        interpreter_type = InterpreterType.HAILO
    elif args.hhb:
        interpreter_type = InterpreterType.HHB
    elif args.executorch:
        interpreter_type = InterpreterType.EXECUTORCH
    elif args.dlr:
        interpreter_type = InterpreterType.DLR
    elif args.ort:
        interpreter_type = InterpreterType.ORT
        if not args.ort_execution_providers:
            raise ValueError(
                "ORT execution providers must be specified when using ONNX Runtime"
            )
    else:
        interpreter_type = InterpreterType.TFLITE

    client = EdgeBenchClient(
        args.device,
        args.broker_host,
        args.broker_port,
        args.threads,
        interpreter_type,
        args.ort_execution_providers,
    )
    client.run()


if __name__ == "__main__":
    main()
