import os
import os.path as osp
import json
from typing import Dict, Optional, Tuple, Callable
from prettytable import PrettyTable


device_name = "test"

REPORTS_DIR = osp.join(
    osp.dirname(osp.abspath(__file__)),
    "reports",
)

DEVICE_REPORTS_DIR = osp.join(REPORTS_DIR, device_name)

TORCH_MODEL_SUMMARY_JSON_FILE = osp.join(REPORTS_DIR, "model_summary.json")

latency_report_json_paths = [
    osp.join(DEVICE_REPORTS_DIR, json_file)
    for json_file in os.listdir(DEVICE_REPORTS_DIR)
    if json_file.endswith(".json") and "_latency" in json_file
]
accuracy_report_json_paths = [
    osp.join(DEVICE_REPORTS_DIR, json_file)
    for json_file in os.listdir(DEVICE_REPORTS_DIR)
    if json_file.endswith(".json") and "_accuracy" in json_file
]

with open(TORCH_MODEL_SUMMARY_JSON_FILE, "r") as f:
    torch_model_summary = json.load(f)

combined_reports = {}
for latency_report_json_path in latency_report_json_paths:
    with open(latency_report_json_path, "r") as f:
        latency_report = json.load(f)
    model_name = latency_report["model_name"]
    if model_name not in combined_reports:
        combined_reports[model_name] = {}
    combined_reports[model_name].update(latency_report)

for accuracy_report_json_path in accuracy_report_json_paths:
    with open(accuracy_report_json_path, "r") as f:
        accuracy_report = json.load(f)
    model_name = accuracy_report["model_name"]
    if model_name not in combined_reports:
        combined_reports[model_name] = {}
    combined_reports[model_name].update(accuracy_report)

for model_name, report in combined_reports.items():
    torch_model_name = model_name.replace("_int8", "")
    torch_model_name = torch_model_name.replace("_from_onnx", "")
    torch_model_name = torch_model_name.replace(".tflite", "")
    if torch_model_name in torch_model_summary:
        report["torch"] = torch_model_summary[torch_model_name]
    else:
        raise ValueError(f"Model {torch_model_name} not found in torch model summary")

combined_reports = dict(sorted(combined_reports.items()))

def compare_metric(
    metric_key: str,
    metric_formatter: Callable[[float], str],
    report_mapper: Callable[[str], str],
    reports_a: Tuple[str, Dict],
    reports_b: Tuple[str, Dict],
    model_name_mapper: Optional[Callable[[Dict, Dict], str]] = None,
    group_by_conversion_method: bool = False,
):
    table = PrettyTable()
    table.align = "l"
    table.field_names = ["Model", reports_a[0], reports_b[0], "Diff"]
    groups = [(reports_a, reports_b)] if not group_by_conversion_method else [
        ((reports_a[0], {k: v for k, v in reports_a[1].items() if "_from_onnx" in k}),
         (reports_b[0], {k: v for k, v in reports_b[1].items() if "_from_onnx" in k})),
        ((reports_a[0], {k: v for k, v in reports_a[1].items() if "_from_onnx" not in k}),
         (reports_b[0], {k: v for k, v in reports_b[1].items() if "_from_onnx" not in k})),
    ]
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
                    model_name_mapper(report_a, report_b) if model_name_mapper else model_a,
                    metric_formatter(report_a_value),
                    metric_formatter(report_b_value),
                    f"{diff_percentage:.2f}%",
                ]
            )
        avg_a = sum(values_a) / len(values_a)
        avg_b = sum(values_b) / len(values_b)
        diff_avg = (avg_a - avg_b) / avg_b * 100
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
    return f"{latency:.2f} ms"


def format_accuracy(accuracy: float) -> str:
    return f"{accuracy:.2f}"


def format_peak_memory(peak_memory: float) -> str:
    return f"{peak_memory:.2f} MB"


def format_flops(flops: float) -> str:
    return f"{flops/1e9:.2f} G"


def format_params(params: float) -> str:
    return f"{params/1e6:.2f} M"


# Compare latency and accuracy of tflite conversion method
# Models with _from_onnx in the name are converted using onnx2tf
# Models without _from_onnx in are converted using ai_edge_torch
print("Latency comparison between onnx2tf and ai_edge_torch (non-quantized):")
compare_metric(
    "avg_latency_ms_from_client",
    format_latency,
    lambda x: x.replace("_from_onnx", ""),
    ("onnx2tf", {k: v for k, v in combined_reports.items() if "_from_onnx" in k and "_int8" not in k}),
    ("ai_edge_torch", {k: v for k, v in combined_reports.items() if "_from_onnx" not in k and "_int8" not in k}),
    lambda report_a, report_b: report_a["torch"]["model_name"],
)

print("\nLatency comparison between onnx2tf and ai_edge_torch (quantized):")
compare_metric(
    "avg_latency_ms_from_client",
    format_latency,
    lambda x: x.replace("_from_onnx", ""),
    ("onnx2tf", {k: v for k, v in combined_reports.items() if "_from_onnx" in k and "_int8" in k}),
    ("ai_edge_torch", {k: v for k, v in combined_reports.items() if "_from_onnx" not in k and "_int8" in k}),
    lambda report_a, report_b: report_a["torch"]["model_name"],
)

print("\nAccuracy comparison between onnx2tf and ai_edge_torch (non-quantized):")
compare_metric(
    "PCK-AUC",
    format_accuracy,
    lambda x: x.replace("_from_onnx", ""),
    ("onnx2tf", {k: v for k, v in combined_reports.items() if "_from_onnx" in k and "_int8" not in k}),
    ("ai_edge_torch", {k: v for k, v in combined_reports.items() if "_from_onnx" not in k and "_int8" not in k}),
    lambda report_a, report_b: report_a["torch"]["model_name"],
)

print("\nAccuracy comparison between onnx2tf and ai_edge_torch (quantized):")
compare_metric(
    "PCK-AUC",
    format_accuracy,
    lambda x: x.replace("_from_onnx", ""),
    ("onnx2tf", {k: v for k, v in combined_reports.items() if "_from_onnx" in k and "_int8" in k}),
    ("ai_edge_torch", {k: v for k, v in combined_reports.items() if "_from_onnx" not in k and "_int8" in k}),
    lambda report_a, report_b: report_a["torch"]["model_name"],
)


# Compare latency, accuracy, peak memory usage, flops, params of classic and light head
# Models with -light in the name are light head
# Models without -classic in the name are classic head

print("\nLatency comparison between light head and classic head (non-quantized):")
compare_metric(
    "avg_latency_ms_from_client",
    format_latency,
    lambda x: x.replace("-light", "-classic"),
    ("light_head", {k: v for k, v in combined_reports.items() if "-light" in k and "_int8" not in k}),
    ("classic_head", {k: v for k, v in combined_reports.items() if "-classic" in k and "_int8" not in k}),
    lambda report_a, report_b: report_a["model_name"].replace("-light", ""),
    group_by_conversion_method=True,
)

print("\nLatency comparison between light head and classic head (quantized):")
compare_metric(
    "avg_latency_ms_from_client",
    format_latency,
    lambda x: x.replace("-light", "-classic"),
    ("light_head", {k: v for k, v in combined_reports.items() if "-light" in k and "_int8" in k}),
    ("classic_head", {k: v for k, v in combined_reports.items() if "-classic" in k and "_int8" in k}),
    lambda report_a, report_b: report_a["model_name"].replace("-light", ""),
    group_by_conversion_method=True,
)

print("\nAccuracy comparison between light head and classic head: (non-quantized)")
compare_metric(
    "PCK-AUC",
    format_accuracy,
    lambda x: x.replace("-light", "-classic"),
    ("light_head", {k: v for k, v in combined_reports.items() if "-light" in k and "_int8" not in k}),
    ("classic_head", {k: v for k, v in combined_reports.items() if "-classic" in k and "_int8" not in k}),
    lambda report_a, report_b: report_a["model_name"].replace("-light", ""),
    group_by_conversion_method=True,
)

print("\nAccuracy comparison between light head and classic head: (quantized)")
compare_metric(
    "PCK-AUC",
    format_accuracy,
    lambda x: x.replace("-light", "-classic"),
    ("light_head", {k: v for k, v in combined_reports.items() if "-light" in k and "_int8" in k}),
    ("classic_head", {k: v for k, v in combined_reports.items() if "-classic" in k and "_int8" in k}),
    lambda report_a, report_b: report_a["model_name"].replace("-light", ""),
    group_by_conversion_method=True,
)

print("\nPeak memory usage comparison between light head and classic head:")
compare_metric(
    "peak_memory_mb",
    format_peak_memory,
    lambda x: x.replace("-light", "-classic"),
    ("light_head", dict([(k, v) for k, v in torch_model_summary.items() if "-light" in k])),
    ("classic_head", dict([(k, v) for k, v in torch_model_summary.items() if "-classic" in k])),
    lambda report_a, report_b: report_a["model_name"].replace("-light", ""),
)

print("\nFLOPs comparison between light head and classic head:")
compare_metric(
    "flops",
    format_flops,
    lambda x: x.replace("-light", "-classic"),
    ("light_head", dict([(k, v) for k, v in torch_model_summary.items() if "-light" in k])),
    ("classic_head", dict([(k, v) for k, v in torch_model_summary.items() if "-classic" in k])),
    lambda report_a, report_b: report_a["model_name"].replace("-light", ""),
)
print("\nParams comparison between light head and classic head:")
compare_metric(
    "params",
    format_params,
    lambda x: x.replace("-light", "-classic"),
    ("light_head", dict([(k, v) for k, v in torch_model_summary.items() if "-light" in k])),
    ("classic_head", dict([(k, v) for k, v in torch_model_summary.items() if "-classic" in k])),
    lambda report_a, report_b: report_a["model_name"].replace("-light", ""),
)

# Compare accuracy between original torch model and tflite model
# Differentiate between quantized and non-quantized models
print("\nAccuracy comparison between tflite and torch (non-quantized):")
compare_metric(
    "PCK-AUC",
    format_accuracy,
    lambda x: x,
    ("tflite", {k: v for k, v in combined_reports.items() if "_int8" not in k}),
    ("torch", dict([(k, v["torch"]) for k, v in combined_reports.items() if "_int8" not in k])),
    lambda report_a, report_b: report_a["model_name"],
    group_by_conversion_method=True,
)

print("\nAccuracy comparison between tflite and torch (quantized):")
compare_metric(
    "PCK-AUC",
    format_accuracy,
    lambda x: x,
    ("tflite", dict([(k, v) for k, v in combined_reports.items() if "_int8" in k])),
    ("torch", dict([(k, v["torch"]) for k, v in combined_reports.items() if "_int8" in k])),
    lambda report_a, report_b: report_a["model_name"],
    group_by_conversion_method=True,
)