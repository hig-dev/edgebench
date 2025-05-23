import argparse
import time
import os
import json
import queue
import uuid
import numpy as np
import ai_edge_litert.interpreter as tflite
from io import BytesIO
import paho.mqtt.client as mqtt
from pck_eval import PckEvaluator
from shared import TestMode, ClientStatus, Command, Topic, Logger, Model
from typing import Dict

BYTE_ORDER = "big"

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")
SAMPLE_INPUT_NCHW_PATH = os.path.join(DATA_DIR, "sample_input_nchw.npy")
TEST_INPUTS_PATH = os.path.join(DATA_DIR, "model_inputs.npy")


class EdgeBenchManager:
    def __init__(
        self,
        device_id: str,
        is_micro: bool,
        model_paths: list[str],
        skip: bool = False,
        broker_host="127.0.0.1",
        broker_port=1883,
    ):
        self.device_id = device_id
        self.is_micro = is_micro
        self.model_paths = model_paths
        self.skip = skip
        self.client = mqtt.Client(
            client_id=str(uuid.uuid4()),
            clean_session=True,
            protocol=mqtt.MQTTv311,
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        )
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.logger = Logger("EdgeBenchManager")
        self.topic = Topic(device_id)
        self.message_queue = queue.Queue()

        # Get model input details
        self.model_input_details: Dict[str, Dict] = {}
        self.model_output_details: Dict[str, Dict] = {}
        for model_path in model_paths:
            interpreter = tflite.Interpreter(model_path=model_path)
            self.model_input_details[model_path] = interpreter.get_input_details()[0]
            self.model_output_details[model_path] = interpreter.get_output_details()[0]

        # Setup callbacks
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

    def on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code == 0:
            self.logger.log("Connected to MQTT broker")
            # Subscribe to topics
            self.client.subscribe(self.topic.STATUS(), qos=1)
            self.client.subscribe(self.topic.RESULT_LATENCY(), qos=1)
            self.client.subscribe(self.topic.RESULT_ACCURACY(), qos=1)
        else:
            self.logger.log(
                f"Failed to connect to MQTT broker with code: {reason_code }"
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

    def set_mode(self, mode: TestMode):
        self.logger.log(f"Setting mode to {mode.name}")
        self.client.publish(
            self.topic.CONFIG_MODE(),
            mode.to_bytes(length=1, byteorder=BYTE_ORDER),
            qos=1,
        )

    def set_model(self, model_path: str):
        model = Model.UNKNOWN
        model_name = os.path.basename(model_path)
        
        if "DeitSmall" in model_name:
            model = Model.DEIT_SMALL
        elif "DeitTiny" in model_name:
            model = Model.DEIT_TINY
        elif "efficientvit_b0" in model_name:
            model = Model.EFFICIENT_VIT_B0
        elif "efficientvit_b1" in model_name:
            model = Model.EFFICIENT_VIT_B1
        elif "efficientvit_b2" in model_name:
            model = Model.EFFICIENT_VIT_B2
        elif "mobileone_s0" in model_name:
            model = Model.MOBILEONE_S0
        elif "mobileone_s1" in model_name:
            model = Model.MOBILEONE_S1
        elif "mobileone_s4" in model_name:
            model = Model.MOBILEONE_S4
        else:
            print(
                f"Unknown model name: {model_name}. Please check the model name."
            )

        self.logger.log(f"Setting model to {model.name}")
        self.client.publish(
            self.topic.CONFIG_MODEL(),
            model.to_bytes(length=4, byteorder=BYTE_ORDER),
            qos=1,
        )

    def set_iterations(self, iterations: int):
        self.logger.log(f"Setting iterations to {iterations}")
        self.client.publish(
            self.topic.CONFIG_ITERATIONS(),
            iterations.to_bytes(length=4, byteorder=BYTE_ORDER),
            qos=1,
        )

    def send_model(self, model_path: str):
        if not model_path:
            raise ValueError("Model file is required for not micro clients.")
        self.logger.log(f"Sending model from {model_path}")
        with open(model_path, "rb") as f:
            model_bytes = f.read()
        self.logger.log(f"Model size: {len(model_bytes)} bytes")
        self.client.publish(self.topic.MODEL(), model_bytes, qos=1)

    def send_input(self, model_path: str, input_data: np.ndarray, topic: str):
        input_details = self.model_input_details[model_path]
        input_shape = input_details["shape"]
        input_dtype = input_details["dtype"]
        if input_data.ndim != len(input_shape):
            raise ValueError(
                f"Input data must have {len(input_shape)} dimensions, but got {input_data.ndim}"
            )
        if input_shape[3] == 3:
            input_data = np.transpose(input_data, (0, 2, 3, 1))
        if input_dtype == np.int8:
            input_scale, input_zero_point = input_details["quantization"]
            input_data = (input_data / input_scale) + input_zero_point
            input_data = np.around(input_data).astype(np.int8)
        input_data_bytes = input_data.flatten().tobytes()
        self.logger.log(
            f"Sending input (shape: {input_data.shape}, dtype: {input_data.dtype}, bytes: {len(input_data_bytes)})"
        )
        self.client.publish(topic, input_data_bytes, qos=1)

    def send_command(self, command: Command):
        self.logger.log(f"Sending command {command.name}")
        self.client.publish(
            self.topic.CMD(), command.to_bytes(length=1, byteorder=BYTE_ORDER), qos=1
        )

    def wait_for_message(self, timeout=None) -> mqtt.MQTTMessage:
        return self.message_queue.get(block=True, timeout=timeout)

    def wait_for_status(self, status: ClientStatus):
        ready = False
        self.logger.log(f"Waiting for client status {status.name}")
        while not ready:
            message = self.wait_for_message()
            if (
                message
                and message.topic == self.topic.STATUS()
                and ClientStatus.from_bytes(message.payload, byteorder=BYTE_ORDER)
                == status
            ):
                ready = True
                self.logger.log(f'Client status for "{self.device_id}" changed to {status.name}')

    def run_latency(
        self,
        iterations: int,
        with_energy_measurement: bool = False,
        already_connected: bool = False,
        disconnect_after: bool = True,
    ):
        l = self.logger
        t = self.topic
        device_id = self.device_id
        l.log(f"Running LatencyTest on {device_id} for {iterations} iterations.")

        if not already_connected:
            self.connect()
            self.wait_for_status(ClientStatus.STARTED)

        for model_index, model_path in enumerate(self.model_paths):
            latency_test_report_path = os.path.join(
                REPORTS_DIR,
                f"{device_id}/{os.path.basename(model_path)}_latency.json",
            )
            if self.skip and os.path.exists(latency_test_report_path):
                l.log(
                    f"Skipping {model_path} as report already exists at {latency_test_report_path}"
                )
                continue
            l.log(
                f"Running for model {model_path} ({model_index + 1}/{len(self.model_paths)})"
            )
            self.set_mode(TestMode.LATENCY)
            self.set_model(model_path)
            self.set_iterations(iterations)
            input_data = np.load(SAMPLE_INPUT_NCHW_PATH)
            self.send_input(model_path, input_data, t.INPUT_LATENCY())
            self.wait_for_status(ClientStatus.READY_FOR_MODEL)
            self.send_model(model_path)
            self.wait_for_status(ClientStatus.READY_FOR_TASK)
            
            if with_energy_measurement:
                l.log("Ready for energy measurement?")
                input("Press Enter to continue...")
            
            self.send_command(Command.START_LATENCY_TEST)
            l.log("Waiting for clients to finish...")

            start_time = time.time()
            end_time = None
            done = False
            elapsed_ms_from_client = None
            while not done or elapsed_ms_from_client is None:
                message = self.wait_for_message()
                if message:
                    if (
                        message.topic == t.STATUS()
                        and ClientStatus.from_bytes(
                            message.payload, byteorder=BYTE_ORDER
                        )
                        == ClientStatus.DONE
                    ):
                        done = True
                        end_time = time.time()
                        l.log(f"{device_id} is done. Timing stopped.")
                    elif message.topic == t.RESULT_LATENCY():
                        elapsed_ms_from_client = int.from_bytes(
                            message.payload, byteorder=BYTE_ORDER
                        )

            elapsed_ms = (end_time - start_time) * 1000
            avg_latency_ms = elapsed_ms / iterations
            avg_latency_ms_from_client = float(elapsed_ms_from_client) / iterations
            measured_energy_input_mWh = 0.0
            if with_energy_measurement:
                l.log("Please enter measured energy in mWh:")
                while measured_energy_input_mWh <= 0:
                    try:
                        measured_energy_input_mWh = float(input())
                    except ValueError:
                        l.log("Invalid input. Please enter a positive number.")

            latency_test_report = {
                "device_id": device_id,
                "model_name": os.path.basename(model_path),
                "iterations": iterations,
                "avg_latency_ms": avg_latency_ms,
                "avg_latency_ms_from_client": avg_latency_ms_from_client,
                "energy_mWh": measured_energy_input_mWh,
            }

            os.makedirs(os.path.dirname(latency_test_report_path), exist_ok=True)
            with open(latency_test_report_path, "w") as f:
                json.dump(latency_test_report, f, indent=4)
            l.log(f"Latency test report saved to {latency_test_report_path}")
            l.log(
                f"Average latency: {avg_latency_ms:.2f} ms (from client: {avg_latency_ms_from_client:.2f} ms)"
            )
            l.log(
                f"Elapsed time: {elapsed_ms:.2f} ms (from client: {elapsed_ms_from_client:.2f} ms)"
            )

            is_last_model = model_index >= len(self.model_paths) - 1
            if is_last_model and disconnect_after:
                self.send_command(Command.STOP)
                self.disconnect()
            else:
                self.send_command(Command.RESET)

    def run_accuracy(
        self,
        limit: int = 0,
        already_connected: bool = False,
        disconnect_after: bool = True,
    ):
        l = self.logger
        t = self.topic
        device_id = self.device_id
        l.log(f"Running AccuracyTest on {device_id}.")

        if not already_connected:
            self.connect()
            self.wait_for_status(ClientStatus.STARTED)

        for model_index, model_path in enumerate(self.model_paths):
            accuracy_test_report_path = os.path.join(
                REPORTS_DIR,
                f"{device_id}/{os.path.basename(model_path)}_accuracy.json",
            )
            if self.skip and os.path.exists(accuracy_test_report_path):
                l.log(
                    f"Skipping {model_path} as report already exists at {accuracy_test_report_path}"
                )
                continue

            l.log(
                f"Running for model {model_path} ({model_index + 1}/{len(self.model_paths)})"
            )
            pck_evaluator = PckEvaluator(
                DATA_DIR, self.model_output_details[model_path], limit=limit
            )
            self.set_mode(TestMode.ACCURACY)
            self.set_model(model_path)

            self.wait_for_status(ClientStatus.READY_FOR_MODEL)
            self.send_model(model_path)
            self.wait_for_status(ClientStatus.READY_FOR_TASK)

            # Load test data
            test_data = np.load(TEST_INPUTS_PATH)
            if limit > 0:
                test_data = test_data[:limit]
            test_data_length = test_data.shape[0]
            for index, input_data in enumerate(test_data):
                l.log(f"Sending input {index + 1}/{test_data_length}")
                input_data = np.expand_dims(input_data, axis=0)
                self.send_input(model_path, input_data, t.INPUT_ACCURACY())
                result = None
                while result is None:
                    message = self.wait_for_message()
                    if message and message.topic == t.RESULT_ACCURACY():
                        result = np.frombuffer(
                            message.payload,
                            dtype=self.model_output_details[model_path]["dtype"],
                        ).reshape(self.model_output_details[model_path]["shape"])
                        l.log(
                            f"Received result for input {index + 1}/{test_data_length}"
                        )
                        pck_evaluator.process(result)

            pck_metrics = pck_evaluator.evaluate()
            accuracy_test_report = {
                "device_id": device_id,
                "model_name": os.path.basename(model_path),
            }
            accuracy_test_report.update(pck_metrics)

            os.makedirs(os.path.dirname(accuracy_test_report_path), exist_ok=True)
            with open(accuracy_test_report_path, "w") as f:
                json.dump(accuracy_test_report, f, indent=4)

            l.log(f"Accuracy test report saved to {accuracy_test_report_path}")
            l.log(f"PCK: {accuracy_test_report['PCK']:.2f}")
            l.log(f"PCK-AUC: {accuracy_test_report['PCK-AUC']:.2f}")
            l.log("AccuracyTest complete.")

            is_last_model = model_index >= len(self.model_paths) - 1
            if is_last_model and disconnect_after:
                self.send_command(Command.STOP)
                self.disconnect()
            else:
                self.send_command(Command.RESET)


def main():
    parser = argparse.ArgumentParser(description="EdgeBenchManager CLI")
    # common arguments
    parser.add_argument("--broker-host", default="127.0.0.1", help="MQTT broker host")
    parser.add_argument(
        "--broker-port", type=int, default=1883, help="MQTT broker port"
    )
    parser.add_argument("--device", required=True, help="Device ID")
    parser.add_argument("--micro", action="store_true", help="Micro client")
    parser.add_argument("--quick", action="store_true", help="Quick test")
    parser.add_argument("--skip", action="store_true", help="Skips if report exists")
    parser.add_argument("--energy", action="store_true", help="With energy measurement")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="*",
        help="Model filename in models/ or '*' for all",
    )
    parser.add_argument(
        "--exlude_quantized", action="store_true", help="Exclude quantized models"
    )
    parser.add_argument(
        "--exlude_classic", action="store_true", help="Exclude models with classic head"
    )
    parser.add_argument(
        "-n",
        "--iterations",
        type=int,
        default=100,
        help="Number of iterations for latency test",
    )
    parser.add_argument("--latency", action="store_true", help="Run latency test")
    parser.add_argument("--accuracy", action="store_true", help="Run accuracy test")
    args = parser.parse_args()

    if not args.latency and not args.accuracy:
        parser.error("At least one of --latency or --accuracy must be specified")

    model_paths = []
    if args.model and args.model != "*":
        model_paths.append(os.path.join(MODELS_DIR, args.model))
    else:
        model_paths = [
            os.path.join(MODELS_DIR, f)
            for f in os.listdir(MODELS_DIR)
            if f.endswith(".tflite")
        ]

    if args.exlude_quantized:
        model_paths = [
            path for path in model_paths if "_int8" not in os.path.basename(path)
        ]

    if args.exlude_classic:
        model_paths = [
            path for path in model_paths if "-classic" not in os.path.basename(path)
        ]

    print(f"Found {len(model_paths)} models: {model_paths}")

    manager = EdgeBenchManager(
        args.device,
        args.micro,
        model_paths,
        args.skip,
        args.broker_host,
        args.broker_port,
    )

    if args.latency:
        manager.run_latency(
            iterations=2 if args.quick else args.iterations,
            disconnect_after=not args.accuracy,
            with_energy_measurement=args.energy,
        )

    if args.accuracy:
        manager.run_accuracy(
            limit=2 if args.quick else 0, already_connected=args.latency
        )


if __name__ == "__main__":
    main()
