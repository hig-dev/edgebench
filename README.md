# EdgeBench

EdgeBench is a benchmarking framework for evaluating edge AI models, specifically focused on pose estimation models. The framework enables performance testing across various devices and model configurations with metrics for both latency and accuracy.

## Overview

EdgeBench provides a client-server architecture using MQTT communication for benchmarking TensorFlow Lite models on edge devices. It supports:

- Latency testing with customizable iteration counts
- Accuracy testing using PCK (Percentage of Correct Keypoints) metrics
- Testing on both standard clients and microcontroller devices
- Automated reporting and analysis

## Project Structure

```
edgebench/
├── analyze.py           # Analysis tools for benchmark results
├── client.py            # Python client implementation
├── manager.py           # Test manager implementation
├── mmpose_eval.py       # MMPose-specific evaluation helpers
├── pck_eval.py          # PCK metrics evaluation
├── requirements.txt     # Python dependencies
├── shared.py            # Shared utilities and constants
├── data/                # Test data and ground truth
├── micro_client/        # Microcontroller client implementations
├── models/              # TFLite models
└── reports/             # Generated test reports
```

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Running the Manager

The manager controls the testing process and communicates with clients:

```bash
python manager.py --device [DEVICE_ID] --broker-host [MQTT_HOST] --broker-port [MQTT_PORT] [OPTIONS]
```

Options:
- `--device`: Device ID (required)
- `--broker-host`: MQTT broker host (default: 127.0.0.1)
- `--broker-port`: MQTT broker port (default: 1883)
- `--micro`: Specify if using a micro client
- `--latency`: Run latency test
- `--accuracy`: Run accuracy test
- `--model`: Specific model to test or "*" for all models
- `--iterations`: Number of iterations for latency test (default: 100)
- `--quick`: Run a quick test (reduced iterations/samples)
- `--energy`: Enable energy measurement

### Running a Client

Clients execute the models and report results back to the manager:

```bash
python client.py --device [DEVICE_ID] --broker-host [MQTT_HOST] --broker-port [MQTT_PORT]
```

### Analyzing Results

After running tests, analyze the results with:

```bash
python analyze.py
```

This will generate comparison reports between different models based on metrics like latency, accuracy, and resource utilization.

## Testing Process

1. The manager connects to the MQTT broker
2. Clients connect and subscribe to configuration topics
3. The manager sends test configuration (mode, iterations, model)
4. Clients load models and prepare for testing
5. The manager initiates tests and clients execute them
6. Results are collected and saved to the reports directory
7. Optional analysis can be performed on the collected data