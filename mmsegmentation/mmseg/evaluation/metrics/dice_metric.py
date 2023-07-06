from collections import OrderedDict

import numpy as np
import torch

from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log

from prettytable import PrettyTable

from mmseg.registry import METRICS


@METRICS.register_module()
class DiceMetric(BaseMetric):
    def __init__(self, collect_device="cpu", prefix=None, **kwargs):
        super().__init__(collect_device=collect_device, prefix=prefix)

        self.classes = self.classes = [
            "finger-1",
            "finger-2",
            "finger-3",
            "finger-4",
            "finger-5",
            "finger-6",
            "finger-7",
            "finger-8",
            "finger-9",
            "finger-10",
            "finger-11",
            "finger-12",
            "finger-13",
            "finger-14",
            "finger-15",
            "finger-16",
            "finger-17",
            "finger-18",
            "finger-19",
            "Trapezium",
            "Trapezoid",
            "Capitate",
            "Hamate",
            "Scaphoid",
            "Lunate",
            "Triquetrum",
            "Pisiform",
            "Radius",
            "Ulna",
        ]

    @staticmethod
    def dice_coef(y_true, y_pred):
        y_true_f = y_true.flatten(-2)
        y_pred_f = y_pred.flatten(-2)
        print("true=", y_true_f.shape)
        print("pred=", y_pred_f.shape)
        intersection = torch.sum(y_true_f * y_pred_f, -1)

        eps = 0.0001
        return (2.0 * intersection + eps) / (
            torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps
        )

    def process(self, data_batch, data_samples):
        for data_sample in data_samples:
            pred_label = data_sample["pred_sem_seg"]["data"]

            label = data_sample["gt_sem_seg"]["data"].to(pred_label)
            self.results.append(self.dice_coef(label, pred_label))

    def compute_metrics(self, results):
        logger: MMLogger = MMLogger.get_current_instance()

        results = torch.stack(self.results, 0)
        dices_per_class = torch.mean(results, 0)
        avg_dice = torch.mean(dices_per_class)

        ret_metrics = {
            "Dice": dices_per_class.detach().cpu().numpy(),
        }
        # summary table
        ret_metrics_summary = OrderedDict(
            {
                ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
                for ret_metric, ret_metric_value in ret_metrics.items()
            }
        )

        metrics = {"mDice": torch.mean(dices_per_class).item()}

        # each class table
        ret_metrics.pop("aAcc", None)
        ret_metrics_class = OrderedDict(
            {
                ret_metric: np.round(ret_metric_value * 100, 2)
                for ret_metric, ret_metric_value in ret_metrics.items()
            }
        )
        ret_metrics_class.update({"Class": self.classes})
        ret_metrics_class.move_to_end("Class", last=False)
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)

        print_log("per class results:", logger)
        print_log("\n" + class_table_data.get_string(), logger=logger)

        return metrics
