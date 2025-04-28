import argparse
import uuid
import tflite_runtime.interpreter as tflite
import numpy as np
import time
import paho.mqtt.client as mqtt
import queue
import tempfile
from io import BytesIO

from shared import TestMode, ClientStatus, Command, Topic, Logger

BYTE_ORDER = "big"


class EdgeBenchClient:
    def __init__(self, device_id, broker_host="127.0.0.1", broker_port=1883):
        self.device_id = device_id
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
        self.iterations = None
        self.mode = None
        self.model = None
        self.input = None
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
            self.model.invoke()
        end_time = time.time()
        self.send_status(ClientStatus.DONE)
        elapsed_time = end_time - start_time
        self.logger.log(
            "Run completed, elapsed time: {:.2f} seconds".format(elapsed_time)
        )
        elapsed_time_ms = int(elapsed_time * 1000)
        self.send_result(elapsed_time_ms)

    def run(self):
        l = self.logger
        t = self.topic
        sent_ready = False
        self.connect()
        l.log(f"Connected as {self.device_id}")
        self.send_status(ClientStatus.STARTED)

        while True:
            msg = self.wait_for_message()

            topic = msg.topic
            payload = msg.payload

            if topic == t.CONFIG_MODE():
                self.mode = TestMode.from_bytes(payload, byteorder=BYTE_ORDER)
                l.log(f"Mode set: {self.mode}")
            elif topic == t.CONFIG_ITERATIONS():
                self.iterations = int.from_bytes(payload, byteorder=BYTE_ORDER)
                l.log(f"Iterations set: {self.iterations}")
            elif topic == t.MODEL():
                model_bytes = payload
                l.log(f"Received model, size: {len(model_bytes)} bytes")
                with tempfile.NamedTemporaryFile(suffix=".tflite") as temp_file:
                    model_path = temp_file.name
                    with open(model_path, "wb") as f:
                        f.write(model_bytes)
                    l.log(f"Model saved to {model_path}")
                    self.model = tflite.Interpreter(model_path=model_path)
                    self.model.allocate_tensors()
                    l.log("Model loaded and tensors allocated")
            elif topic == t.INPUT_LATENCY():
                input_bytes = payload
                l.log(f"Received input, size: {len(input_bytes)} bytes")
                self.input = np.load(BytesIO(input_bytes))
            elif topic == t.INPUT_ACCURACY():
                input_bytes = payload
                l.log(f"Received input, size: {len(input_bytes)} bytes")
                input = np.load(BytesIO(input_bytes))
                input_details = self.model.get_input_details()
                output_details = self.model.get_output_details()
                self.model.set_tensor(input_details[0]["index"], input)
                self.model.invoke()
                output_data = self.model.get_tensor(output_details[0]["index"])
                output_bytes_io = BytesIO()
                np.save(output_bytes_io, output_data)
                self.client.publish(
                    self.topic.RESULT_ACCURACY(), output_bytes_io.getvalue(), qos=1
                )
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
                    sent_ready = False
                    self.iterations = None
                    self.mode = None
                    self.model = None
                    self.input = None
                    continue
            if (
                not sent_ready
                and self.mode == TestMode.LATENCY
                and self.iterations is not None
                and self.model is not None
                and self.input is not None
            ):
                sent_ready = True
                l.log("All configurations received, loading model and input data")
                # Prepare the model and input data
                input_details = self.model.get_input_details()
                self.model.set_tensor(input_details[0]["index"], self.input)
                self.send_status(ClientStatus.READY)
            elif (
                not sent_ready
                and self.mode == TestMode.ACCURACY
                and self.model is not None
            ):
                sent_ready = True
                l.log("All configurations received, loading model and input data")
                self.send_status(ClientStatus.READY)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True, help="Device ID")
    parser.add_argument("--broker-host", default="127.0.0.1", help="MQTT broker host")
    parser.add_argument(
        "--broker-port", type=int, default=1883, help="MQTT broker port"
    )
    args = parser.parse_args()
    client = EdgeBenchClient(args.device, args.broker_host, args.broker_port)
    client.run()


if __name__ == "__main__":
    main()
