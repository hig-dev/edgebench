# EdgeBench

EdgeBench is a benchmarking framework for evaluating Edge AI models, specifically focused on pose estimation models. The framework enables performance testing across various devices and model configurations with metrics for latency, accuracy, and energy consumption.

## Overview

EdgeBench provides a host-client architecture using MQTT communication for benchmarking neural network models on Edge AI devices.

- **Inference Frameworks**: TensorFlow Lite, ONNX Runtime, Executorch, Apache TVM, Hailo RT, Neo AI-DLR, and HHB
- **Latency Testing**: Customizable iterations count
- **Accuracy Testing**: PCK (Percentage of Correct Keypoints) metrics for pose estimation
- **Energy Measurement**: Allows manual input for energy consumption
- **Multi-device Support**: Python clients and microcontroller implementations

## Project Structure

```
edgebench/
├── analyze.py           # Analysis and visualization for benchmark results
├── client.py            # Python client implementation with multi-framework support
├── manager.py           # Test manager implementation and orchestration
├── mmpose_eval.py       # MMPose-specific evaluation helpers
├── pck_eval.py          # PCK metrics evaluation for pose estimation
├── requirements.txt     # Python dependencies
├── shared.py            # Shared utilities, constants, and communication protocols
├── interpreters/        # Inference framework implementations
│   ├── base_interpreter.py       # Base interpreter interface
│   ├── tflite_interpreter.py     # TensorFlow Lite interpreter
│   ├── ort_interpreter.py        # ONNX Runtime interpreter
│   ├── executorch_interpreter.py # Executorch interpreter
│   ├── tvm_interpreter.py        # Apache TVM interpreter
│   ├── hailo_interpreter.py      # Hailo AI accelerator interpreter
│   ├── dlr_interpreter.py        # NEO AI DLR interpreter
│   └── hhb_interpreter.py        # HHB (Heterogeneous Honey Badger) interpreter
├── micro_client/        # Microcontroller client implementations
│   ├── coralmicro/         # Google Coral Dev Board Micro
│   ├── esp32/              # ESP32-based implementation
│   ├── grove_vision_ai_v2/ # Seeed Grove Vision AI V2
│   └── shared/             # Shared microcontroller utilities
├── test_data/           # Test data and sample inputs
├── reports/             # Generated test reports (runtime)
├── final_reports/       # Final benchmark results of master thesis
└── plots/               # Generated visualization plots
```

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd edgebench
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up MQTT broker (if not using external broker):
   ```bash
   # Install mosquitto or use Docker
   docker run -it -p 1883:1883 eclipse-mosquitto
   ```

## Supported Devices and Frameworks

### Client Devices
- **Generic devices with Python support**: Any device that supports the inference library and Python.
- **Single Board Computers**: Raspberry Pi 5, BeagleV-Ahead, BeagleY-AI, etc.
- **AI Accelerators**: Raspberry Pi AI HAT+ (Hailo-8)
- **Microcontrollers**: ESP32-S3, Grove Vision AI V2, Coral Dev Board Micro

### Inference Frameworks
- **TensorFlow Lite**
- **ONNX Runtime**
- **ExecuTorch**
- **Apache TVM**: Deep learning compiler stack
- **Hailo RT**: Inference runtime for Hailo NPUs
- **Neo-AI-DLR**: NEO AI Deep Learning Runtime
- **HHB**: HHB (Heterogeneous Honey Badger) for BeagleV-Ahead

## Usage

### Running the Manager

The manager orchestrates testing across devices and collects results:

```bash
python manager.py --device [DEVICE_ID] [OPTIONS]
```

**Required Arguments:**
- `--device`: Device identifier (required)
- `--latency` or `--accuracy`: At least one test mode must be specified

**Connection Options:**
- `--broker-host`: MQTT broker host (default: 127.0.0.1)
- `--broker-port`: MQTT broker port (default: 1883)

**Test Configuration:**
- `--iterations`, `-n`: Number of iterations for latency test (default: 100)
- `--limit`, `-l`: Limit for accuracy test samples (0 for no limit)
- `--energy`: Enable energy measurement during testing

**Model Selection:**
- `--models-dir`: Models directory (default: ./models)
- `--file-type`: Model file extension (default: .tflite)
- `--filter`, `-f`: Include models containing all specified strings
- `--not-filter`, `-nf`: Exclude models containing any specified strings

**Data and Reports:**
- `--data-dir`: Test data directory (default: ./test_data)
- `--reports-dir`: Output reports directory (default: ./reports)

**Advanced Options:**
- `--skip`: Skip testing if report already exists
- `--chunked`: Transfer large models in chunks
- `--skip-send-model`: Skip model transfer

### Running a Client

Clients execute models and report results back to the manager:

```bash
python client.py --device [DEVICE_ID] [FRAMEWORK_OPTIONS]
```

**Basic Options:**
- `--device`: Device identifier (must match manager)
- `--threads`: Number of inference threads (default: 1) (used only for TensorFlow Lite inference)
- `--broker-host`: MQTT broker host (default: 127.0.0.1)
- `--broker-port`: MQTT broker port (default: 1883)

**Framework Selection (choose one):**
- `--hailo`: Use Hailo AI accelerator
- `--hhb`: Use HHB framework
- `--executorch`: Use ExecuTorch
- `--tvm`: Use Apache TVM
- `--dlr`: Use NEO AI DLR Runtime
- `--ort`: Use ONNX Runtime
- `--ort-execution-providers`: ONNX Runtime execution providers (e.g., CPUExecutionProvider)
- (default): TensorFlow Lite

### Analysis and Visualization

Generate comprehensive analysis reports and visualizations:

```bash
python analyze.py
```

This script automatically:
- Processes all benchmark results from `final_reports/`
- Generates comparison tables and charts
- Creates bar plots for latency, accuracy, and energy metrics
- Outputs visualization to `plots/` directory

## Testing Process

The benchmarking workflow follows these steps:

1. The manager connects to the MQTT broker
2. Clients connect and subscribe to configuration topics
3. The manager sends test configuration (mode, iterations, model)
4. Clients load models and prepares for testing
5. The manager initiates tests and clients execute them
6. Results are collected and saved to the reports directory
7. Optional analysis can be performed on the collected data

## Additional Information
This benchmarking framework was created as part of the master thesis *Benchmarking Computer Vision Tasks on Edge AI Hardware*.

- Compatible models: https://github.com/hig-dev/edgebench-models
- Model configs and export scripts: https://github.com/hig-dev/mmpose