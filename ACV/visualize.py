"""
Qualitative Visualization and Failure Analysis Script
Assignment B - Unit 3: Advanced Computer Vision (CS107203CS)

Produces:
1. Side-by-side comparative panels: [Input Image | Ground Truth | Baseline | Controlled Variant]
2. Dedicated Failure Case Analysis documenting false negatives, boundary errors, and thin-object degradation.
3. Saves high-resolution figures to 'visualizations/' directory.
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from src.data_contract import CamVidDataset, CLASS_NAMES, NUM_CLASSES, RAW_IGNORE_LABEL, IGNORE_INDEX, colorize_mask, IMAGE_MEAN, IMAGE_STD
from src.models import build_model


def unnormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """Converts a normalized [3, H, W] tensor back to a displayable [H, W, 3] uint8 numpy image."""
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    mean = np.array(IMAGE_MEAN)
    std = np.array(IMAGE_STD)
    img = (img * std + mean) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def generate_visualizations(
    baseline_ckpt: str,
    variant_ckpt: str,
    data_dir: str = "data/CamVid",
    output_dir: str = "visualizations",
    num_samples: int = 5
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Baseline Model
    b_data = torch.load(baseline_ckpt, map_location="cpu", weights_only=False)
    b_args = b_data.get("args", {})
    b_model = build_model(
        mechanism=b_args.get("mechanism", "baseline"),
        num_classes=NUM_CLASSES,
        base_channels=b_args.get("base_channels", 16)
    )
    b_model.load_state_dict(b_data["model_state_dict"])
    b_model.to(device)
    b_model.eval()

    # 2. Load Variant Model
    v_data = torch.load(variant_ckpt, map_location="cpu", weights_only=False)
    v_args = v_data.get("args", {})
    v_model = build_model(
        mechanism=v_args.get("mechanism", "dilated_bottleneck"),
        num_classes=NUM_CLASSES,
        base_channels=v_args.get("base_channels", 16)
    )
    v_model.load_state_dict(v_data["model_state_dict"])
    v_model.to(device)
    v_model.eval()

    # 3. Load Test Dataset
    resolution = tuple(b_args.get("resolution", [192, 256]))
    test_dataset = CamVidDataset(data_dir, split="test", resolution=resolution, augment=False)

    # 4. Comparative Grid Generation
    indices = np.linspace(0, len(test_dataset) - 1, num_samples, dtype=int)

    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 3.2 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, 0)

    column_titles = ["Input Image", "Ground Truth", "Baseline Prediction", "Controlled (Dilated) Variant"]

    for col, title in enumerate(column_titles):
        axes[0, col].set_title(title, fontsize=12, fontweight="bold", pad=8)

    sample_errors = []

    for row, idx in enumerate(indices):
        img_tensor, mask_tensor, img_name = test_dataset[idx]
        input_batch = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            b_out = b_model(input_batch)
            b_pred = torch.argmax(b_out, dim=1).squeeze(0).cpu().numpy()

            v_out = v_model(input_batch)
            v_pred = torch.argmax(v_out, dim=1).squeeze(0).cpu().numpy()

        rgb_img = unnormalize_image(img_tensor)
        gt_mask = mask_tensor.cpu().numpy()

        gt_color = colorize_mask(gt_mask)
        b_color = colorize_mask(b_pred)
        v_color = colorize_mask(v_pred)

        axes[row, 0].imshow(rgb_img)
        axes[row, 1].imshow(gt_color)
        axes[row, 2].imshow(b_color)
        axes[row, 3].imshow(v_color)

        for col in range(4):
            axes[row, col].axis("off")

        # Track error rate for failure analysis
        valid_pixels = (gt_mask != IGNORE_INDEX)
        error_rate = np.mean(b_pred[valid_pixels] != gt_mask[valid_pixels])
        sample_errors.append((error_rate, idx, img_name))

    plt.tight_layout()
    comparison_path = os.path.join(output_dir, "comparative_predictions.png")
    plt.savefig(comparison_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved comparative predictions to: {comparison_path}")

    # 5. Dedicated Failure Case Analysis
    # Pick the 2 highest-error test samples
    sample_errors.sort(key=lambda x: x[0], reverse=True)
    worst_indices = [x[1] for x in sample_errors[:2]]

    fig, axes = plt.subplots(len(worst_indices), 5, figsize=(18, 3.5 * len(worst_indices)))
    if len(worst_indices) == 1:
        axes = np.expand_dims(axes, 0)

    fail_titles = ["Input Image", "Ground Truth", "Baseline Pred", "Dilated Pred", "Baseline Error Map (Red=Error)"]
    for col, title in enumerate(fail_titles):
        axes[0, col].set_title(title, fontsize=11, fontweight="bold", pad=8)

    for row, idx in enumerate(worst_indices):
        img_tensor, mask_tensor, img_name = test_dataset[idx]
        input_batch = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            b_pred = torch.argmax(b_model(input_batch), dim=1).squeeze(0).cpu().numpy()
            v_pred = torch.argmax(v_model(input_batch), dim=1).squeeze(0).cpu().numpy()

        rgb_img = unnormalize_image(img_tensor)
        gt_mask = mask_tensor.cpu().numpy()
        valid = (gt_mask != IGNORE_INDEX)

        # Error map: Red where wrong, Gray where correct, Black where ignored
        error_map = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
        error_map[valid & (b_pred == gt_mask)] = [100, 100, 100]  # Correct (gray)
        error_map[valid & (b_pred != gt_mask)] = [220, 20, 20]    # Error (red)

        axes[row, 0].imshow(rgb_img)
        axes[row, 1].imshow(colorize_mask(gt_mask))
        axes[row, 2].imshow(colorize_mask(b_pred))
        axes[row, 3].imshow(colorize_mask(v_pred))
        axes[row, 4].imshow(error_map)

        for col in range(5):
            axes[row, col].axis("off")

    plt.tight_layout()
    failure_path = os.path.join(output_dir, "failure_case_analysis.png")
    plt.savefig(failure_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved failure case analysis to: {failure_path}")

    return {
        "comparison_image": comparison_path,
        "failure_image": failure_path,
        "worst_test_samples": [x[2] for x in sample_errors[:2]]
    }


def main():
    parser = argparse.ArgumentParser(description="Qualitative Visualizer")
    parser.add_argument("--baseline", type=str, required=True, help="Baseline best_model.pth")
    parser.add_argument("--variant", type=str, required=True, help="Variant best_model.pth")
    parser.add_argument("--data_dir", type=str, default="data/CamVid", help="Data directory")
    parser.add_argument("--output_dir", type=str, default="visualizations", help="Output directory")
    args = parser.parse_args()

    generate_visualizations(args.baseline, args.variant, args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
