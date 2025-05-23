from enum import IntEnum


class TestMode(IntEnum):
    NONE = 0
    LATENCY = 1
    ACCURACY = 2


class ClientStatus(IntEnum):
    NONE = 0
    STARTED = 1
    READY_FOR_MODEL = 2
    READY_FOR_TASK = 3
    DONE = 4


class Command(IntEnum):
    NONE = 0
    START_LATENCY_TEST = 1
    STOP = 2
    RESET = 3

class Model(IntEnum):
    UNKNOWN = 0
    DEIT_SMALL = 1
    DEIT_TINY = 2
    EFFICIENT_VIT_B0 = 3
    EFFICIENT_VIT_B1 = 4
    EFFICIENT_VIT_B2 = 5
    MOBILEONE_S0 = 6
    MOBILEONE_S1 = 7
    MOBILEONE_S4 = 8

class Topic:
    def __init__(self, device_id):
        self.device_id = device_id

    def CONFIG_ITERATIONS(self):
        return f"bench/{self.device_id}/config/iterations"

    def CONFIG_MODE(self):
        return f"bench/{self.device_id}/config/mode"
    
    def CONFIG_MODEL(self):
        return f"bench/{self.device_id}/config/model"

    def MODEL(self):
        return f"bench/{self.device_id}/model"

    def INPUT_LATENCY(self):
        return f"bench/{self.device_id}/input/latency"

    def INPUT_ACCURACY(self):
        return f"bench/{self.device_id}/input/accuracy"

    def CMD(self):
        return f"bench/{self.device_id}/cmd"

    def STATUS(self):
        return f"bench/{self.device_id}/status"

    def RESULT_LATENCY(self):
        return f"bench/{self.device_id}/result/latency"

    def RESULT_ACCURACY(self):
        return f"bench/{self.device_id}/result/accuracy"


class Logger:
    def __init__(self, name):
        self.name = name

    def log(self, message):
        print(f"[{self.name}] {message}")
