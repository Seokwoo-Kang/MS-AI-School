# train loop
# val loop
# save model
# 평가 함수
import torch
import os
import torch.nn as nn
from collections import defaultdict
## nn?
## nn.


# 평가 함수
def calculate_acc(output, target):
    output = torch.sigmoid(output) >= 0.5
    target = target == 1.0

    return torch.true_divide((output==target).sum(dim=0), output.size(0)).item()

# save model
def save_model(model, save_dir, file_name="last.pt"):
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, file_name)
    if isinstance(model, nn.DataParallel):
        print("멀티 GPU 저장 !!")
        torch.save(model.module.state_dict(), output_path)

    else:
        print("싱글 GPU 저장 !!")
        torch.save(model.state_dict(), output_path)

class MetricMonitor:
    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.reset()
    
    def reset(self):
        self.metrics = defaultdict(lambda : {"val":0, "count":0, "avg":0})

    def update(self, metrics_name, val):
        metrics = self.metrics[metrics_name]

        metrics["val"] += val
        metrics["count"] += 1
        metrics["avg"] = metrics["val"] / metrics["count"]

    def __str__(self):
        return " | ".join([
            "{metric_name} : {avg.{float_precision}f}".format(
                metric_name = metric_name, avg=metrics["avg"],
                float_precision=self.float_precision
            )
            for(metric_name, metrics) in self.metrics.items()
        ])
