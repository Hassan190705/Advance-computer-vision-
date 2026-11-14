"""
Segmentation Metrics: Per-Class IoU, mIoU, and Confusion Matrix
Assignment B - Unit 3: Advanced Computer Vision (CS107203CS)

Computes:
- Intersection-over-Union (IoU) per semantic class
- Mean Intersection-over-Union (mIoU)
- Overall Pixel Accuracy
- Handles ignore_index (255) strictly according to the Data Contract.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import torch


class StreamSegMetrics:
    """
    Streaming segmentation evaluator that updates a confusion matrix across batches.
    """
    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None, ignore_index: int = 255):
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.total_seen_pixels = 0
        self.total_valid_pixels = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Updates the confusion matrix with predictions and ground truth targets.
        pred: [B, H, W] or [H, W] (class indices)
        target: [B, H, W] or [H, W] (class indices with potential ignore_index)
        """
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()

        pred = pred.flatten()
        target = target.flatten()

        self.total_seen_pixels += len(target)

        # Filter out ignored pixels (void)
        valid_mask = (target != self.ignore_index) & (target >= 0) & (target < self.num_classes)
        pred_valid = pred[valid_mask]
        target_valid = target[valid_mask]

        self.total_valid_pixels += len(target_valid)

        if len(target_valid) > 0:
            # Vectorized fast 2D histogram / confusion matrix update
            # bincount on target * num_classes + pred
            indices = target_valid * self.num_classes + pred_valid
            counts = np.bincount(indices, minlength=self.num_classes ** 2)
            self.confusion_matrix += counts.reshape(self.num_classes, self.num_classes)

    def get_results(self) -> Dict[str, any]:
        """
        Computes per-class IoU, mean IoU, and pixel accuracy from the confusion matrix.
        """
        hist = self.confusion_matrix
        tp = np.diag(hist)
        fp = hist.sum(axis=0) - tp
        fn = hist.sum(axis=1) - tp
        denominator = tp + fp + fn

        # Avoid division by zero for classes not present in test ground truth or prediction
        with np.errstate(divide="ignore", invalid="ignore"):
            per_class_iou = np.where(denominator > 0, tp / denominator, np.nan)

        valid_class_mask = ~np.isnan(per_class_iou)
        mean_iou = float(np.mean(per_class_iou[valid_class_mask])) if np.any(valid_class_mask) else 0.0

        total_correct = tp.sum()
        pixel_accuracy = float(total_correct / self.total_valid_pixels) if self.total_valid_pixels > 0 else 0.0

        class_iou_dict = {
            self.class_names[i]: (float(per_class_iou[i]) if not np.isnan(per_class_iou[i]) else 0.0)
            for i in range(self.num_classes)
        }

        return {
            "mIoU": mean_iou,
            "Pixel_Accuracy": pixel_accuracy,
            "Per_Class_IoU": class_iou_dict,
            "Confusion_Matrix": hist.tolist(),
            "Total_Valid_Pixels": int(self.total_valid_pixels),
            "Total_Seen_Pixels": int(self.total_seen_pixels),
        }

    def print_summary(self):
        """Prints a human-readable table of per-class metrics."""
        res = self.get_results()
        print("-" * 50)
        print(f"Overall mIoU:           {res['mIoU'] * 100:.2f}%")
        print(f"Overall Pixel Accuracy: {res['Pixel_Accuracy'] * 100:.2f}%")
        print("-" * 50)
        print(f"{'Class ID':<10}{'Class Name':<15}{'IoU (%)':<10}")
        print("-" * 50)
        for i, name in enumerate(self.class_names):
            iou_pct = res["Per_Class_IoU"][name] * 100
            print(f"{i:<10}{name:<15}{iou_pct:<10.2f}")
        print("-" * 50)


if __name__ == "__main__":
    from src.data_contract import CLASS_NAMES
    evaluator = StreamSegMetrics(num_classes=len(CLASS_NAMES), class_names=CLASS_NAMES)
    
    # Dummy test
    pred = torch.randint(0, 11, (2, 192, 256))
    target = pred.clone()
    target[0, 0:10, 0:10] = 255  # test ignore_index
    evaluator.update(pred, target)
    evaluator.print_summary()
