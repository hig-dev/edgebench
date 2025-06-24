import os
import os.path as osp
import json
from typing import Dict, Optional, Tuple, Callable
from prettytable import PrettyTable


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

all_device_names = sorted([
    x for x in os.listdir(REPORTS_DIR) if os.path.isdir(os.path.join(REPORTS_DIR, x))
])
all_device_names = ["esp32s3","coralmicro","grove_vision_ai_v2","rp5","beaglevahead", "beagleyai", "hailo"]


print(f"\nAvailable devices: {all_device_names}")
for model_name in models:
    print(f"\nComparison for {model_name} across devices:")
    table = PrettyTable()
    table.align = "l"
    table.field_names = ["Device", "Latency (ms)", "PCK", "PCK-AUC", "Model File"]
    for device_name in all_device_names:
        device_reports_dir = osp.join(REPORTS_DIR, device_name)
        latency_report_path = [
            x
            for x in os.listdir(device_reports_dir)
            if model_name in x and "_latency" in x
        ]
        latency_report_path = (
            osp.join(device_reports_dir, latency_report_path[0])
            if latency_report_path
            else None
        )
        accuracy_report_path = [
            x
            for x in os.listdir(device_reports_dir)
            if model_name in x and "_accuracy" in x
        ]
        accuracy_report_path = (
            osp.join(device_reports_dir, accuracy_report_path[0])
            if accuracy_report_path
            else None
        )

        if latency_report_path and osp.exists(latency_report_path):
            with open(latency_report_path, "r") as f:
                latency_report = json.load(f)
            model_file_name = osp.basename(latency_report_path).replace(
                "_latency.json", ""
            )
        else:
            latency_report = {"avg_latency_ms_from_client": -1}
            model_file_name = "DNF"

        if accuracy_report_path and osp.exists(accuracy_report_path):
            with open(accuracy_report_path, "r") as f:
                accuracy_report = json.load(f)
        else:
            accuracy_report = {"PCK": -1, "PCK-AUC": -1}

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
    
    for device_name in all_device_names:
        row = [device_name]
        for model_name in models:
            device_reports_dir = osp.join(REPORTS_DIR, device_name)
            report_identifier = "_latency" if metric_key == "avg_latency_ms_from_client" or metric_key == "energy_mWh"  else "_accuracy"
            report_path = [
                x
                for x in os.listdir(device_reports_dir)
                if model_name in x and report_identifier in x
            ]
            report_path = (
                osp.join(device_reports_dir, report_path[0])
                if report_path
                else None
            )
            if report_path and osp.exists(report_path):
                with open(report_path, "r") as f:
                    report = json.load(f)
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
                            f"Invalid iterations count {iterations} in report {report_path}"
                        )
                    row.append(f"{(value / iterations)*1000:.0f} μWh")
            elif metric_key in ["PCK", "PCK-AUC"]:
                # Find the torch model summary for the current model by model name
                # The key should contain the model name
                torch_report = next((
                    (k, v)
                    for k, v in torch_model_summary.items()
                    if model_name in k and "light" in k
                ), None)
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
                    row.append(f"{format_accuracy(value)} ({accuracy_drop_percent:.2f}%)")
                
            else:
                raise ValueError(f"Unknown metric key: {metric_key}")
        table.add_row(row)
    print(table)

print("\nPower comparison across devices:")
table = PrettyTable()
table.align = "l"
table.field_names = ["Device", "Idle Power (W)", "Inference Power (W)"]
for device_name in all_device_names:
    idle_energy_value = idle_energy.get(device_name, -1)
    # Calculate avg inference power
    total_energy_used_Wh = 0
    total_time_seconds = 0
    for model_name in models:
        device_reports_dir = osp.join(REPORTS_DIR, device_name)
        report_path = [
            x
            for x in os.listdir(device_reports_dir)
            if model_name in x and "_latency" in x
        ]
        report_path = (
            osp.join(device_reports_dir, report_path[0])
            if report_path
            else None
        )
        if report_path and osp.exists(report_path):
            with open(report_path, "r") as f:
                report = json.load(f)
            avg_latency_ms = report.get("avg_latency_ms_from_client", -1)
            iterations = report.get("iterations", 1)
            energy_mWh = report.get("energy_mWh", 0)
            if avg_latency_ms > 0 and iterations > 0 and energy_mWh > 0:
                avg_latency_seconds = avg_latency_ms / 1000
                energy_Wh = energy_mWh / 1000
                total_time_seconds += avg_latency_seconds * iterations
                total_energy_used_Wh += energy_Wh
    if total_time_seconds > 0:
        avg_inference_power_W = total_energy_used_Wh * 3600 / total_time_seconds
    else:
        avg_inference_power_W = 0
    table.add_row([device_name, f"{idle_energy_value:.2f} W", f"{avg_inference_power_W:.2f} W"])

print(table)