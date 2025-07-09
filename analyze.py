from io import StringIO
import os
import os.path as osp
import json
from typing import Dict, Optional, Tuple, Callable
from cycler import cycler
from prettytable import PrettyTable
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

models = [
    "mobileone_s0",
    "mobileone_s1",
    "mobileone_s4",
    "efficientvit_b0",
    "efficientvit_b1",
    "efficientvit_b2",
    "DeitTiny",
    "DeitSmall",
]

device_names = [
    "rtx5090", # NVIDIA RTX 5090 as a high-end desktop GPU
    "esp32s3",
    "coralmicro",
    "grove_vision_ai_v2",
    "rp5",
    "beaglevahead",
    "beagleyai",
    "hailo",
]

display_names = {
    "rtx5090": "RTX 5090",
    "esp32s3": "XIAO ESP32S3 Sense",
    "coralmicro": "Coral Dev Board Micro",
    "grove_vision_ai_v2": "Grove Vision AI V2",
    "rp5": "Raspberry Pi 5",
    "beaglevahead": "BeagleV-Ahead",
    "beagleyai": "BeagleY-AI",
    "hailo": "Raspberry Pi AI HAT+",
}

REPORTS_DIR = osp.join(
    osp.dirname(osp.abspath(__file__)),
    "final_reports",
)


TORCH_MODEL_SUMMARY_JSON_FILE = osp.join(REPORTS_DIR, "model_summary.json")
IDLE_ENERGY_JSON_FILE = osp.join(REPORTS_DIR, "idle_energy.json")


with open(TORCH_MODEL_SUMMARY_JSON_FILE, "r") as f:
    torch_model_summary = json.load(f)

with open(IDLE_ENERGY_JSON_FILE, "r") as f:
    idle_energy = json.load(f)

def get_latency_report(device_name: str, model_name: str):
    device_reports_dir = osp.join(REPORTS_DIR, device_name)
    report_path = [
        x
        for x in os.listdir(device_reports_dir)
        if model_name in x and "_latency" in x
    ]
    report_path = (
        osp.join(device_reports_dir, report_path[0]) if report_path else None
    )
    if report_path and osp.exists(report_path):
        with open(report_path, "r") as f:
            return json.load(f)
    return None

def get_accuracy_report(device_name: str, model_name: str):
    device_reports_dir = osp.join(REPORTS_DIR, device_name)
    report_path = [
        x
        for x in os.listdir(device_reports_dir)
        if model_name in x and "_accuracy" in x
    ]
    report_path = (
        osp.join(device_reports_dir, report_path[0]) if report_path else None
    )
    if report_path and osp.exists(report_path):
        with open(report_path, "r") as f:
            return json.load(f)
    return None


def compare_metric(
    metric_key: str,
    metric_formatter: Callable[[float], str],
    report_mapper: Callable[[str], str],
    reports_a: Tuple[str, Dict],
    reports_b: Tuple[str, Dict],
    model_name_mapper: Optional[Callable[[Dict, Dict], str]] = None,
    group_by_conversion_method: bool = False,
):
    if len(reports_a[1]) == 0 or len(reports_b[1]) == 0:
        print(f"No reports found for {reports_a[0]} or {reports_b[0]}")
        return
    table = PrettyTable()
    table.align = "l"
    table.field_names = ["Model", reports_a[0], reports_b[0], "Diff"]
    groups = (
        [(reports_a, reports_b)]
        if not group_by_conversion_method
        else [
            (
                (
                    reports_a[0],
                    {k: v for k, v in reports_a[1].items() if "_onnx2tf" in k},
                ),
                (
                    reports_b[0],
                    {k: v for k, v in reports_b[1].items() if "_onnx2tf" in k},
                ),
            ),
            (
                (
                    reports_a[0],
                    {k: v for k, v in reports_a[1].items() if "_aiedgetorch" in k},
                ),
                (
                    reports_b[0],
                    {k: v for k, v in reports_b[1].items() if "_aiedgetorch" in k},
                ),
            ),
        ]
    )
    for reports_a, reports_b in groups:
        values_a = []
        values_b = []
        for model_a, report_a in reports_a[1].items():
            model_b = report_mapper(model_a)
            if model_b not in reports_b[1]:
                continue
            report_b = reports_b[1][model_b]
            report_a_value = report_a[metric_key]
            report_b_value = report_b[metric_key]
            diff_percentage = (report_a_value - report_b_value) / report_b_value * 100
            values_a.append(report_a_value)
            values_b.append(report_b_value)

            table.add_row(
                [
                    (
                        model_name_mapper(report_a, report_b)
                        if model_name_mapper
                        else model_a
                    ),
                    metric_formatter(report_a_value),
                    metric_formatter(report_b_value),
                    f"{diff_percentage:.2f}%",
                ]
            )
        avg_a = sum(values_a) / len(values_a) if len(values_a) > 0 else 0
        avg_b = sum(values_b) / len(values_b) if len(values_b) > 0 else 0
        diff_avg = (avg_a - avg_b) / avg_b * 100 if avg_b > 0 else 0
        table.add_divider()
        table.add_row(
            [
                "Average",
                metric_formatter(avg_a),
                metric_formatter(avg_b),
                f"{diff_avg:.2f}%",
            ]
        )
        table.add_divider()
    print(table)


def format_latency(latency: float) -> str:
    return f"{latency:.2f} ms" if latency >= 0 else "DNF"


def format_accuracy(accuracy: float) -> str:
    return f"{accuracy:.2f}" if accuracy >= 0 else "DNF"


def format_peak_memory(peak_memory: float) -> str:
    return f"{peak_memory:.2f} MB"


def format_flops(flops: float) -> str:
    return f"{flops/1e9:.2f} G"


def format_params(params: float) -> str:
    return f"{params/1e6:.2f} M"


print("\nPeak memory usage comparison between light head and classic head:")
compare_metric(
    "peak_memory_mb",
    format_peak_memory,
    lambda x: x.replace("-light", "-classic"),
    (
        "light_head",
        dict([(k, v) for k, v in torch_model_summary.items() if "-light" in k]),
    ),
    (
        "classic_head",
        dict([(k, v) for k, v in torch_model_summary.items() if "-classic" in k]),
    ),
    lambda report_a, report_b: report_a["model_name"].replace("-light", ""),
)

print("\nFLOPs comparison between light head and classic head:")
compare_metric(
    "flops",
    format_flops,
    lambda x: x.replace("-light", "-classic"),
    (
        "light_head",
        dict([(k, v) for k, v in torch_model_summary.items() if "-light" in k]),
    ),
    (
        "classic_head",
        dict([(k, v) for k, v in torch_model_summary.items() if "-classic" in k]),
    ),
    lambda report_a, report_b: report_a["model_name"].replace("-light", ""),
)
print("\nParams comparison between light head and classic head:")
compare_metric(
    "params",
    format_params,
    lambda x: x.replace("-light", "-classic"),
    (
        "light_head",
        dict([(k, v) for k, v in torch_model_summary.items() if "-light" in k]),
    ),
    (
        "classic_head",
        dict([(k, v) for k, v in torch_model_summary.items() if "-classic" in k]),
    ),
    lambda report_a, report_b: report_a["model_name"].replace("-light", ""),
)
print("\nLatency comparison between light head and classic head:")
compare_metric(
    "avg_latency_ms",
    format_latency,
    lambda x: x.replace("-light", "-classic"),
    (
        "light_head",
        dict([(k, v) for k, v in torch_model_summary.items() if "-light" in k]),
    ),
    (
        "classic_head",
        dict([(k, v) for k, v in torch_model_summary.items() if "-classic" in k]),
    ),
    lambda report_a, report_b: report_a["model_name"].replace("-light", ""),
)
print("\nPCK comparison between light head and classic head:")
compare_metric(
    "PCK",
    format_accuracy,
    lambda x: x.replace("-light", "-classic"),
    (
        "light_head",
        dict([(k, v) for k, v in torch_model_summary.items() if "-light" in k]),
    ),
    (
        "classic_head",
        dict([(k, v) for k, v in torch_model_summary.items() if "-classic" in k]),
    ),
    lambda report_a, report_b: report_a["model_name"].replace("-light", ""),
)
print("\nPCK-AUC comparison between light head and classic head:")
compare_metric(
    "PCK-AUC",
    format_accuracy,
    lambda x: x.replace("-light", "-classic"),
    (
        "light_head",
        dict([(k, v) for k, v in torch_model_summary.items() if "-light" in k]),
    ),
    (
        "classic_head",
        dict([(k, v) for k, v in torch_model_summary.items() if "-classic" in k]),
    ),
    lambda report_a, report_b: report_a["model_name"].replace("-light", ""),
)


print(f"\nAvailable devices: {device_names}")
for model_name in models:
    print(f"\nComparison for {model_name} across devices:")
    table = PrettyTable()
    table.align = "l"
    table.field_names = ["Device", "Latency (ms)", "PCK", "PCK-AUC", "Model File"]
    for device_name in device_names:
        latency_report = get_latency_report(device_name, model_name)
        accuracy_report = get_accuracy_report(device_name, model_name)

        if not latency_report:
            latency_report = {"avg_latency_ms_from_client": -1}

        if not accuracy_report:
            accuracy_report = {"PCK": -1, "PCK-AUC": -1}

        model_file_name = latency_report.get("model_name", "DNF")

        table.add_row(
            [
                device_name,
                format_latency(latency_report["avg_latency_ms_from_client"]),
                format_accuracy(accuracy_report["PCK"]),
                format_accuracy(accuracy_report["PCK-AUC"]),
                model_file_name,
            ]
        )
    print(table)

for metric_key, metric_name in [
    ("avg_latency_ms_from_client", "Latency (ms)"),
    ("PCK", "PCK"),
    ("PCK-AUC", "PCK-AUC"),
    ("energy_mWh", "Energy (mWh)"),
]:
    print(f"\n{metric_name} comparison across devices:")
    table = PrettyTable()
    table.align = "l"
    table.field_names = ["Device"] + models

    for device_name in device_names:
        row = [device_name]
        for model_name in models:
            if metric_key == "avg_latency_ms_from_client" or metric_key == "energy_mWh":
                report = get_latency_report(device_name, model_name)
            else:
                report = get_accuracy_report(device_name, model_name)
            
            if report:
                value = report.get(metric_key, -1)
            else:
                value = -1
            
            if metric_key == "avg_latency_ms_from_client":
                row.append(format_latency(value))
            elif metric_key == "energy_mWh":
                if value <= 0:
                    row.append("DNF")
                else:
                    iterations = report.get("iterations", -1)
                    if iterations <= 0:
                        raise ValueError(
                            f"Invalid iterations count {iterations}"
                        )
                    row.append(f"{(value / iterations)*1000:.0f} μWh")
            elif metric_key in ["PCK", "PCK-AUC"]:
                # Find the torch model summary for the current model by model name
                # The key should contain the model name
                torch_report = next(
                    (
                        (k, v)
                        for k, v in torch_model_summary.items()
                        if model_name in k and "light" in k
                    ),
                    None,
                )
                torch_accuracy = torch_report[1][metric_key] if torch_report else -1
                if torch_accuracy < 0:
                    raise ValueError(
                        f"Could not find torch model summary for {model_name} with key {metric_key}"
                    )
                accuracy_drop_percent = (
                    (torch_accuracy - value) / torch_accuracy * 100
                    if torch_accuracy > 0
                    else 0
                )
                if value <= 0:
                    row.append("DNF")
                else:
                    row.append(
                        f"{format_accuracy(value)} ({accuracy_drop_percent:.2f}%)"
                    )

            else:
                raise ValueError(f"Unknown metric key: {metric_key}")
        table.add_row(row)
    print(table)

print("\nPower adjustment grove_vision_ai_v2 (with carrier board):")
for model_name in models:
    carrier_idle_watt = idle_energy.get("esp32s3", -1)
    report = get_latency_report("grove_vision_ai_v2", model_name)
    if not report:
        print(f"Report for {model_name} not found in grove_vision_ai_v2")
        continue

    iterations = report.get("iterations", 1)
    duration_ms = report.get("avg_latency_ms_from_client", -1) * iterations
    energy_Wh = report.get("energy_mWh", -1) * 0.001
    if duration_ms <= 0 or energy_Wh <= 0:
        raise ValueError(
            f"Invalid duration {duration_ms} ms or energy {energy_Wh} mWh in report"
        )
    carrier_energy_Wh = carrier_idle_watt * (duration_ms / 1000) / 3600
    adjusted_energy_Wh = energy_Wh + carrier_energy_Wh
    adjusted_energy_per_inference_Wh = adjusted_energy_Wh / iterations
    adjusted_energy_per_inference_microWh = adjusted_energy_per_inference_Wh * 1e6
    print(
        f"{model_name}: {adjusted_energy_per_inference_microWh:.0f} μWh (+{((carrier_energy_Wh/iterations)*1e6):.2f})"
    )

    

print("\nPower comparison across devices:")
table = PrettyTable()
table.align = "l"
table.field_names = [
    "Device",
    "Idle Power (W)",
    "Avg Inference Power (W)",
    "Variance",
    "Std",
    "Coefficient of Variation"
] + models
for device_name in device_names:
    idle_energy_value = idle_energy.get(device_name, -1)
    # Calculate avg inference power
    total_energy_used_Wh_by_model = { m : 0 for m in models }
    total_time_seconds_by_model = { m : 0 for m in models }
    for model_name in models:
        report = get_latency_report(device_name, model_name)
        if report:
            avg_latency_ms = report.get("avg_latency_ms_from_client", -1)
            iterations = report.get("iterations", 1)
            energy_mWh = report.get("energy_mWh", 0)
            if avg_latency_ms > 0 and iterations > 0 and energy_mWh > 0:
                avg_latency_seconds = avg_latency_ms / 1000
                energy_Wh = energy_mWh / 1000
                total_time_seconds_by_model[model_name] += avg_latency_seconds * iterations
                total_energy_used_Wh_by_model[model_name] += energy_Wh

    def get_avg_inference_power(energy_used_Wh: float, time_seconds: float) -> float:
        if time_seconds > 0:
            return energy_used_Wh * 3600 / time_seconds
        else:
            return 0

    avg_inference_power_by_model = {
        model_name: get_avg_inference_power(
            total_energy_used_Wh_by_model[model_name],
            total_time_seconds_by_model[model_name]
        )
        for model_name in models
    }
    total_energy_used_Wh = sum(total_energy_used_Wh_by_model.values())
    total_time_seconds = sum(total_time_seconds_by_model.values())
    average_inference_power = get_avg_inference_power(total_energy_used_Wh, total_time_seconds)
    # Calculate variance using only valid models (non-zero avg inference power)
    valid_models = [model_name for model_name in models if avg_inference_power_by_model[model_name] > 0]
    if valid_models:
        valid_avg = sum(avg_inference_power_by_model[m] for m in valid_models) / len(valid_models)
        variance = sum(
            (avg_inference_power_by_model[m] - valid_avg) ** 2 for m in valid_models
        ) / len(valid_models)
    else:
        variance = 0
    std = variance ** 0.5
    coefficient_of_variation = std / average_inference_power if average_inference_power > 0 else 0
    table.add_row(
        [
            device_name,
            f"{idle_energy_value:.2f} W",
            f"{average_inference_power:.2f} W",
            f"{variance:.4f} W^2",
            f"{std:.2f} W",
            f"{coefficient_of_variation:.4f}",
        ] +
        [f"{avg_inference_power_by_model[model_name]:.2f} W" for model_name in models]
    )

print(table)

print("\nScores:")
table = PrettyTable()
table.align = "l"
table.field_names = [
    "Device",
    "Compatibility",
    "PCK",
    "PCK-AUC",
    "Accuracy",
    "Inference Latency",
    "Energy Efficiency",
    "Total Score",
    "Best Model",
]

for device_name in device_names:
    total_models = len(models)
    failed_models = 0
    best_model = None

    reference_pck = 88.22534478272183
    reference_pck_auc = 29.465925082304175
    reference_device_name = "rtx5090"

    energy_reference_device_name = "grove_vision_ai_v2"
    energy_reference_model_name = "mobileone_s0"

    best_pck = 0
    best_pck_auc = 0

    if "rtx5090" in device_name:
            best_pck = reference_pck
            best_pck_auc = reference_pck_auc
            best_model = "efficientvit_l2"
    else:
        for model_name in models:
            latency_report = get_latency_report(device_name, model_name)
            accuracy_report = get_accuracy_report(device_name, model_name)

            if not latency_report or not accuracy_report:
                failed_models += 1
                continue
            
            
            pck = accuracy_report.get("PCK", 0)
            pck_auc = accuracy_report.get("PCK-AUC", 0)

            reference_accuracy_same_model_report = get_accuracy_report(reference_device_name, model_name)
            reference_pck_same_model = reference_accuracy_same_model_report.get("PCK", 0)
            pck_drop =  (reference_pck_same_model - pck) / reference_pck_same_model
                    
            if pck_drop > 0.5:
                failed_models += 1
                continue

            if pck > best_pck or pck_auc > best_pck_auc:
                best_pck = pck
                best_pck_auc = pck_auc
                best_model = model_name

    if not best_model:
        raise ValueError(
            f"No valid model found for device {device_name} with reports"
        )
    
    reference_latency_report = get_latency_report(reference_device_name, best_model)
    device_latency_report = get_latency_report(device_name, best_model)
    assert reference_latency_report, f"Reference latency report for {reference_device_name} not found"
    assert device_latency_report, f"Device latency report for {device_name} not found"

    reference_latency = reference_latency_report.get("avg_latency_ms_from_client", 0)
    device_latency = device_latency_report.get("avg_latency_ms_from_client", 0)
    assert reference_latency > 0, f"Reference latency for {reference_device_name} is not valid"
    assert device_latency > 0, f"Device latency for {device_name} is not valid"

    energy_reference_latency_report = get_latency_report(energy_reference_device_name, energy_reference_model_name)
    energy_device_latency_report = get_latency_report(device_name, energy_reference_model_name)
    assert energy_reference_latency_report, f"Energy reference latency report for {energy_reference_device_name} not found"
    assert energy_device_latency_report, f"Energy device latency report for {device_name} not found"

    reference_energy = energy_reference_latency_report.get("energy_mWh", 0) / energy_reference_latency_report.get("iterations", 1)
    device_energy = energy_device_latency_report.get("energy_mWh", 0) / energy_device_latency_report.get("iterations", 1)
    assert reference_energy > 0, f"Reference energy for {energy_reference_device_name} is not valid"
    assert device_energy > 0, f"Device energy for {device_name} is not valid"

    compatibility_score = (total_models - failed_models) / total_models * 100
    pck_score = (best_pck / reference_pck) * 100
    pck_auc_score = (best_pck_auc / reference_pck_auc) * 100
    accuracy_score = (pck_score + pck_auc_score) / 2
    latency_score = (reference_latency / device_latency) * 100
    energy_score = (reference_energy / device_energy) * 100
    total_score = compatibility_score + accuracy_score + latency_score + energy_score
    table.add_row([
        display_names.get(device_name, device_name),
        f"{compatibility_score:.0f}",
        f"{pck_score:.2f}",
        f"{pck_auc_score:.2f}",
        f"{accuracy_score:.0f}",
        f"{latency_score:.2f}",
        f"{energy_score:.0f}",
        f"{total_score:.0f}",
        best_model,
    ])

print(table)
print(table.get_latex_string())
def create_plot(df: pd.DataFrame, file_path: str, metrics: list, x_label: str = "Device", y_label: str = "Score", stacked: bool = True):
    plt.rcParams['axes.prop_cycle'] = cycler(color=['#FFC000', '#FF6600', '#FF0033', '#FF33CC'])
    plt.rcParams['axes.axisbelow'] = True
    plt.rcParams['grid.color'] = 'gray'
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.linewidth'] = 0.5
    plt.rcParams['grid.alpha'] = 0.7

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(df))
    bottom = np.zeros(len(df))
    width = 0.25

    if stacked:
        for metric in metrics:
            ax.bar(x, df[metric], bottom=bottom, label=metric)
            bottom += df[metric]
    else:
        for i, metric in enumerate(metrics):
            ax.bar(x + i*width, df[metric], width, label=metric)

    ax.grid(axis='y')
    ax.set_xticks(x) if stacked else ax.set_xticks(x + width)
    ax.set_xticklabels(df[x_label], rotation=45, ha="right")
    ax.set_ylabel(y_label)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(file_path)

df = pd.read_csv(StringIO(table.get_csv_string()))
metrics = ["Compatibility", "Accuracy", "Inference Latency", "Energy Efficiency"]
df['Total Score'] = df[metrics].sum(axis=1)
df_sorted = df.sort_values("Total Score", ascending=False).reset_index(drop=True)
create_plot(df_sorted, "plots/scores-bar-chart.png", metrics)

print("\nNeural architecture comparison:")
cnn_repr_model = "mobileone_s1"
transformer_repr_model = "DeitTiny"
hybrid_repr_model = "efficientvit_b1"

table_latency = PrettyTable()
table_latency.align = "l"
table_latency.field_names = [
    "Device",
    "CNN-based model",
    "Hybrid model",
    "Transformer-based model",
    "Lat. Hybrid/CNN (%)",
    "Lat. Transformer/CNN (%)",
]
table_energy = PrettyTable()
table_energy.align = "l"
table_energy.field_names = [
    "Device",
    "CNN-based model",
    "Hybrid model",
    "Transformer-based model",
    "Energy Hybrid/CNN (%)",
    "Energy Transformer/CNN (%)",
]
for device_name in device_names:
    if "rtx5090" in device_name:
        continue
    cnn_report = get_latency_report(device_name, cnn_repr_model)
    transformer_report = get_latency_report(device_name, transformer_repr_model)
    hybrid_report = get_latency_report(device_name, hybrid_repr_model)

    if not cnn_report:
        cnn_report = {"avg_latency_ms_from_client": -1, "energy_mWh": -1, "iterations": 1}
    if not transformer_report:
        transformer_report = {"avg_latency_ms_from_client": -1, "energy_mWh": -1, "iterations": 1}
    if not hybrid_report:
        hybrid_report = {"avg_latency_ms_from_client": -1, "energy_mWh": -1, "iterations": 1}

    cnn_latency = cnn_report.get("avg_latency_ms_from_client", -1)
    transformer_latency = transformer_report.get("avg_latency_ms_from_client", -1)
    hybrid_latency = hybrid_report.get("avg_latency_ms_from_client", -1)

    cnn_energy = cnn_report.get("energy_mWh", -1) / cnn_report.get("iterations", 1)
    transformer_energy = transformer_report.get("energy_mWh", -1) / transformer_report.get("iterations", 1)
    hybrid_energy = hybrid_report.get("energy_mWh", -1) / hybrid_report.get("iterations", 1)

    if transformer_latency < 0 and hybrid_latency < 0:
        continue

    increased_latency_hybrid = (
        hybrid_latency / cnn_latency * 100
        if cnn_latency > 0 else -1
    )
    increased_latency_transformer = (
        transformer_latency / cnn_latency * 100
        if cnn_latency > 0 else -1
    )
    increased_energy_hybrid = (
        hybrid_energy / cnn_energy * 100
        if cnn_energy > 0 else -1
    )
    increased_energy_transformer = (
        transformer_energy / cnn_energy * 100
        if cnn_energy > 0 else -1
    )

    table_latency.add_row([
        display_names.get(device_name, device_name),
        cnn_latency if cnn_latency >= 0 else "0",
        hybrid_latency if hybrid_latency >= 0 else "0",
        transformer_latency if transformer_latency >= 0 else "0",
        f"{increased_latency_hybrid:.2f}%" if hybrid_latency >= 0 else "DNF",
        f"{increased_latency_transformer:.2f}%" if transformer_latency >= 0 else "DNF",
    ])
    table_energy.add_row([
        display_names.get(device_name, device_name),
        f"{(cnn_energy * 1000):.0f}" if cnn_energy > 0 else "0",
        f"{(hybrid_energy * 1000):.0f}" if hybrid_energy > 0 else "0",
        f"{(transformer_energy * 1000):.0f}" if transformer_energy > 0 else "0",
        f"{increased_energy_hybrid:.2f}%" if hybrid_energy >= 0 else "DNF",
        f"{increased_energy_transformer:.2f}%" if transformer_energy >= 0 else "DNF",
    ])

print(table_latency)
df = pd.read_csv(StringIO(table_latency.get_csv_string()))
metrics = ["CNN-based model", "Hybrid model", "Transformer-based model"]
create_plot(df, "plots/latency-bar-chart.png", metrics, y_label="Inference Latency (ms)", stacked=False)

print(table_energy)
df = pd.read_csv(StringIO(table_energy.get_csv_string()))
metrics = ["CNN-based model", "Hybrid model", "Transformer-based model"]
create_plot(df, "plots/energy-bar-chart.png", metrics, y_label="Energy Efficiency (μWh)", stacked=False)