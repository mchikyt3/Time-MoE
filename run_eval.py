#!/usr/bin/env python
import argparse
import logging
import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from time_moe.datasets.benchmark_dataset import BenchmarkEvalDataset, GeneralEvalDataset
from time_moe.models.modeling_time_moe import (
    TimeMoeForPrediction,
    TimeMoeForSequenceClassification,
    TimeMoeForTokenClassification,
)


def setup_nccl(rank, world_size, master_addr="127.0.0.1", master_port=9899):
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        rank=rank,
        world_size=world_size,
    )


def count_num_tensor_elements(tensor):
    n = 1
    for s in tensor.shape:
        n = n * s
    return n


# ------------------ Metrics ------------------
class SumEvalMetric:
    def __init__(self, name, init_val: float = 0.0):
        self.name = name
        self.value = init_val

    def push(self, preds, labels, **kwargs):
        self.value += self._calculate(preds, labels, **kwargs)

    def _calculate(self, preds, labels, **kwargs):
        pass


class MSEMetric(SumEvalMetric):
    def _calculate(self, preds, labels, **kwargs):
        return torch.sum((preds - labels) ** 2)


class MAEMetric(SumEvalMetric):
    def _calculate(self, preds, labels, **kwargs):
        return torch.sum(torch.abs(preds - labels))


class AccuracyMetric(SumEvalMetric):
    def _calculate(self, preds, labels, **kwargs):
        if len(preds.shape) > 1:
            preds = torch.argmax(preds, dim=-1)
        correct = (preds == labels).sum()
        return correct.float()


class F1Metric(SumEvalMetric):
    def __init__(self, name, num_classes, init_val: float = 0.0):
        super().__init__(name, init_val)
        self.num_classes = num_classes
        self.true_positives = torch.zeros(num_classes)
        self.predicted_positives = torch.zeros(num_classes)
        self.actual_positives = torch.zeros(num_classes)

    def _calculate(self, preds, labels, **kwargs):
        if len(preds.shape) > 1:
            preds = torch.argmax(preds, dim=-1)

        # Flatten for token classification
        preds = preds.flatten()
        labels = labels.flatten()

        for c in range(self.num_classes):
            self.true_positives[c] += ((preds == c) & (labels == c)).sum()
            self.predicted_positives[c] += (preds == c).sum()
            self.actual_positives[c] += (labels == c).sum()

        return 0  # F1 calculated in get_f1_score method

    def get_f1_score(self):
        precision = self.true_positives / (self.predicted_positives + 1e-8)
        recall = self.true_positives / (self.actual_positives + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        return f1.mean().item()  # Macro F1


class TimeMoE:
    def __init__(
        self,
        model_path,
        device,
        task_type="forecasting",
        context_length=None,
        prediction_length=None,
        **kwargs,
    ):
        self.task_type = task_type

        if task_type == "forecasting":
            try:
                model = TimeMoeForPrediction.from_pretrained(
                    model_path,
                    device_map=device,
                    # attn_implementation='flash_attention_2',
                    torch_dtype="auto",
                )
            except Exception:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map=device,
                    # attn_implementation='flash_attention_2',
                    torch_dtype="auto",
                    trust_remote_code=True,
                )
        elif task_type == "sequence_classification":
            model = TimeMoeForSequenceClassification.from_pretrained(
                model_path,
                device_map=device,
                torch_dtype="auto",
            )
        elif task_type == "token_classification":
            model = TimeMoeForTokenClassification.from_pretrained(
                model_path,
                device_map=device,
                torch_dtype="auto",
            )
        else:
            raise ValueError(f"Unsupported task_type: {task_type}")

        logging.info(
            f">>> Model dtype: {model.dtype}; Attention:{model.config._attn_implementation}"
        )

        self.model = model
        self.device = device
        self.prediction_length = prediction_length
        self.model.eval()

    def predict(self, batch):
        model = self.model
        device = self.device

        if self.task_type == "forecasting":
            prediction_length = self.prediction_length
            outputs = model.generate(
                inputs=batch["inputs"].to(device).to(model.dtype),
                max_new_tokens=prediction_length,
            )
            preds = outputs[:, -prediction_length:]
            labels = batch["labels"].to(device)
            if len(preds.shape) > len(labels.shape):
                labels = labels[..., None]
        else:  # classification tasks
            inputs = batch["inputs"].to(device).to(model.dtype)
            labels = batch["labels"].to(device)

            with torch.no_grad():
                outputs = model(input_ids=inputs)
                preds = outputs.logits

        return preds, labels


def evaluate(args):
    batch_size = args.batch_size
    context_length = args.context_length
    prediction_length = args.prediction_length

    master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")
    master_port = os.getenv("MASTER_PORT", 9899)
    world_size = int(os.getenv("WORLD_SIZE") or 1)
    rank = int(os.getenv("RANK") or 0)
    local_rank = int(os.getenv("LOCAL_RANK") or 0)
    if torch.cuda.is_available():
        try:
            setup_nccl(
                rank=rank,
                world_size=world_size,
                master_addr=master_addr,
                master_port=master_port,
            )
            device = f"cuda:{local_rank}"
            is_dist = True
        except Exception as e:
            print("Error: ", f"Setup nccl fail, so set device to cpu: {e}")
            device = "cpu"
            is_dist = False
    else:
        device = "cpu"
        is_dist = False

    # evaluation
    if args.task_type == "forecasting":
        metric_list = [
            MSEMetric(name="mse"),
            MAEMetric(name="mae"),
        ]
    else:  # classification tasks
        metric_list = [
            AccuracyMetric(name="accuracy"),
        ]
        if args.num_classes:
            f1_metric = F1Metric(name="f1", num_classes=args.num_classes)
            metric_list.append(f1_metric)

    model = TimeMoE(
        args.model,
        device,
        task_type=args.task_type,
        context_length=context_length,
        prediction_length=prediction_length,
    )
    if args.data.endswith(".csv"):
        dataset = BenchmarkEvalDataset(
            args.data,
            context_length=context_length,
            prediction_length=prediction_length,
        )
    else:
        dataset = GeneralEvalDataset(
            args.data,
            context_length=context_length,
            prediction_length=prediction_length,
        )

    if torch.cuda.is_available() and dist.is_initialized():
        sampler = DistributedSampler(dataset=dataset, shuffle=False)
    else:
        sampler = None
    test_dl = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=2,
        prefetch_factor=2,
        drop_last=False,
    )

    acc_count = 0
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_dl)):
            preds, labels = model.predict(batch)

            for metric in metric_list:
                metric.push(preds, labels)

            acc_count += count_num_tensor_elements(preds)

    ret_metric = {}
    for metric in metric_list:
        ret_metric[metric.name] = metric.value / acc_count
    print(f"{rank} - {ret_metric}")

    metric_tensors = [metric.value for metric in metric_list] + [acc_count]
    if is_dist:
        stat_tensor = torch.tensor(metric_tensors).to(model.device)
        gathered_results = [torch.zeros_like(stat_tensor) for _ in range(world_size)]
        dist.all_gather(gathered_results, stat_tensor)
        all_stat = torch.stack(gathered_results, dim=0).sum(dim=0)
    else:
        all_stat = metric_tensors

    if rank == 0:
        item = {
            "model": args.model,
            "data": args.data,
            "task_type": args.task_type,
        }

        if args.task_type == "forecasting":
            item.update(
                {
                    "context_length": args.context_length,
                    "prediction_length": args.prediction_length,
                }
            )
        elif args.num_classes:
            item["num_classes"] = args.num_classes

        count = all_stat[-1]
        for i, metric in enumerate(metric_list):
            if isinstance(metric, F1Metric):
                # F1 metric needs special handling
                val = metric.get_f1_score()
            else:
                val = all_stat[i] / count
            item[metric.name] = (
                float(val.cpu().numpy()) if hasattr(val, "cpu") else float(val)
            )
        logging.info(item)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("TimeMoE Evaluate")
    parser.add_argument(
        "--model", "-m", type=str, default="Maple728/TimeMoE-50M", help="Model path"
    )
    parser.add_argument("--data", "-d", type=str, help="Benchmark data path")

    # Task type arguments
    parser.add_argument(
        "--task_type",
        type=str,
        choices=["forecasting", "sequence_classification", "token_classification"],
        default="forecasting",
        help="Type of task: forecasting (default), sequence_classification, or token_classification",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=None,
        help="Number of classes for classification tasks (required for classification)",
    )

    parser.add_argument(
        "--batch_size", "-b", type=int, default=32, help="Batch size of evaluation"
    )
    parser.add_argument("--context_length", "-c", type=int, help="Context length")
    parser.add_argument(
        "--prediction_length",
        "-p",
        type=int,
        default=96,
        help="Prediction length (for forecasting tasks)",
    )
    args = parser.parse_args()

    # Validation
    if (
        args.task_type in ["sequence_classification", "token_classification"]
        and args.num_classes is None
    ):
        parser.error(f"--num_classes is required for {args.task_type}")

    if args.context_length is None:
        if args.task_type == "forecasting":
            if args.prediction_length == 96:
                args.context_length = 512
            elif args.prediction_length == 192:
                args.context_length = 1024
            elif args.prediction_length == 336:
                args.context_length = 2048
            elif args.prediction_length == 720:
                args.context_length = 3072
            else:
                args.context_length = args.prediction_length * 4
        else:
            # For classification tasks, use a reasonable default
            args.context_length = 1024

    evaluate(args)
