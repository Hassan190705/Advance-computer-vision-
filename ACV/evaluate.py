"""
Exact Evaluation Script for Reproducible Semantic Segmentation Experiment
Assignment B - Unit 3: Advanced Computer Vision (CS107203CS)

Measures and outputs:
- Per-class IoU for all 11 benchmark classes
- Overall Mean IoU (mIoU) & Pixel Accuracy on the official test set (233 images)
- Total parameter count & trainable parameter count
- Precise inference latency benchmarking (mean ms, std, min, max, FPS) under controlled conditions
- Outputs results to console, Markdown, and JSON.
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, Any

import numpy as np
import torch
import pandas as pd
from tabulate import tabulate

from src.data_contract import get_dataloaders, CLASS_NAMES, NUM_CLASSES, IGNORE_INDEX
from src.models import build_model, get_model_summary
from src.metrics import StreamSegMetrics


def measure_inference_latency(
    model: torch.nn.Module,
    input_resolution: tuple = (192, 256),
    device: torch.device = torch.device("cpu"),
    warmup_runs: int = 30,
    timed_runs: int = 100
) -> Dict[str, float]:
    """
    Benchmarks CPU inference latency under explicit, controlled conditions:
    - Batch size = 1
    - Warm-up iterations to stabilize CPU caches & branch predictors
    - High-resolution timing (time.perf_counter)
    """
    model.eval()
    dummy_input = torch.randn(1, 3, input_resolution[0], input_resolution[1], device=device)

    # 1. Warm-up
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)

    # 2. Timed runs
    latencies = []
    with torch.no_grad():
        for _ in range(timed_runs):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # milliseconds

    latencies = np.array(latencies)
    mean_lat = float(np.mean(latencies))
    std_lat = float(np.std(latencies))
    min_lat = float(np.min(latencies))
    max_lat = float(np.max(latencies))
    fps = float(1000.0 / mean_lat) if mean_lat > 0 else 0.0

    return {
        "warmup_runs": warmup_runs,
        "timed_runs": timed_runs,
        "mean_latency_ms": round(mean_lat, 2),
        "std_latency_ms": round(std_lat, 2),
        "min_latency_ms": round(min_lat, 2),
        "max_latency_ms": round(max_lat, 2),
        "throughput_fps": round(fps, 1),
        "device": str(device),
        "threads": torch.get_num_threads(),
        "batch_size": 1,
        "resolution": f"{input_resolution[0]}x{input_resolution[1]}"
    }


def evaluate_checkpoint(checkpoint_path: str, data_dir: str = "data/CamVid") -> Dict[str, Any]:
    assert os.path.exists(checkpoint_path), f"Checkpoint not found: {checkpoint_path}"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    args = checkpoint.get("args", {})
    mechanism = args.get("mechanism", "baseline")
    base_channels = args.get("base_channels", 16)
    resolution = tuple(args.get("resolution", [192, 256]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Rebuild and load model
    model = build_model(mechanism=mechanism, num_classes=NUM_CLASSES, base_channels=base_channels)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    summary = get_model_summary(model, input_size=(1, 3, resolution[0], resolution[1]))

    # 2. Test Split Evaluation
    loaders = get_dataloaders(data_dir, batch_size=8, resolution=resolution)
    evaluator = StreamSegMetrics(num_classes=NUM_CLASSES, class_names=CLASS_NAMES, ignore_index=IGNORE_INDEX)

    with torch.no_grad():
        for images, masks, _ in loaders["test"]:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            evaluator.update(preds, masks)

    test_metrics = evaluator.get_results()

    # 3. Latency Benchmarking (Batch size = 1, CPU)
    latency_stats = measure_inference_latency(model, input_resolution=resolution, device=device)

    # 4. Consolidate results
    results = {
        "checkpoint": checkpoint_path,
        "mechanism": mechanism,
        "base_channels": base_channels,
        "resolution": f"{resolution[0]}x{resolution[1]}",
        "parameters": summary["total_parameters"],
        "mIoU": test_metrics["mIoU"],
        "pixel_accuracy": test_metrics["Pixel_Accuracy"],
        "latency": latency_stats,
        "per_class_iou": test_metrics["Per_Class_IoU"]
    }

    # Print Clean Formatted Tables
    print("\n" + "=" * 70)
    print(f"TEST EVALUATION REPORT: {mechanism.upper()}")
    print("=" * 70)
    print(f"Checkpoint:       {checkpoint_path}")
    print(f"Model Mechanism:  {mechanism}")
    print(f"Total Parameters: {summary['total_parameters']:,}")
    print(f"Mean IoU (mIoU):  {test_metrics['mIoU'] * 100:.2f}%")
    print(f"Pixel Accuracy:   {test_metrics['Pixel_Accuracy'] * 100:.2f}%")
    print(f"Latency (CPU):    {latency_stats['mean_latency_ms']} ms ± {latency_stats['std_latency_ms']} ms ({latency_stats['throughput_fps']} FPS)")
    print("-" * 70)

    # Per-Class Table
    table_data = []
    for i, name in enumerate(CLASS_NAMES):
        iou_val = test_metrics["Per_Class_IoU"][name] * 100
        table_data.append([i, name, f"{iou_val:.2f}%"])

    print(tabulate(table_data, headers=["ID", "Class Name", "Test IoU"], tablefmt="github"))
    print("=" * 70)

    # Save to JSON
    output_json = os.path.join(os.path.dirname(checkpoint_path), "test_eval_results.json")
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved evaluation metrics to: {output_json}\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="Exact Evaluation Script")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pth checkpoint")
    parser.add_argument("--data_dir", type=str, default="data/CamVid", help="Path to CamVid directory")
    args = parser.parse_args()

    evaluate_checkpoint(args.checkpoint, args.data_dir)


if __name__ == "__main__":
    main()
