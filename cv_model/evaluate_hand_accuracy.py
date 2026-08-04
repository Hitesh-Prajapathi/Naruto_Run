#!/usr/bin/env python3
"""Report per-class accuracy and calibrated rejection on the existing split."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from inference.hand_config import HAND_CONFIDENCE_THRESHOLDS, HAND_MARGIN_THRESHOLDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parent / "data/prepared_dataset/test",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).resolve().parent / "models/best_model_A.onnx",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.data.is_dir():
        raise SystemExit(f"Test split not found: {args.data}")
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    names = ast.literal_eval(session.get_modelmeta().custom_metadata_map["names"])
    classes = tuple(names[index] for index in sorted(names))
    input_name = session.get_inputs()[0].name
    confusion = np.zeros((len(classes), len(classes)), dtype=int)
    accepted_confusion = np.zeros_like(confusion)
    rejected = np.zeros(len(classes), dtype=int)

    for truth_index, truth in enumerate(classes):
        for path in sorted((args.data / truth).glob("*")):
            image = cv2.imread(str(path))
            if image is None:
                continue
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
            tensor = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))[None]
            probabilities = session.run(None, {input_name: tensor})[0][0]
            ranking = np.argsort(probabilities)[::-1]
            prediction_index, second_index = int(ranking[0]), int(ranking[1])
            confidence = float(probabilities[prediction_index])
            margin = confidence - float(probabilities[second_index])
            prediction = classes[prediction_index]
            confusion[truth_index, prediction_index] += 1
            pair_resolved = (
                prediction == "ram"
                and classes[second_index] == "rat"
                and float(probabilities[second_index]) >= 0.12
            )
            accepted_prediction_index = (
                classes.index("rat") if pair_resolved else prediction_index
            )
            accepted = pair_resolved or (
                confidence >= HAND_CONFIDENCE_THRESHOLDS[prediction]
                and margin >= HAND_MARGIN_THRESHOLDS[prediction]
            )
            if accepted:
                accepted_confusion[truth_index, accepted_prediction_index] += 1
            else:
                rejected[truth_index] += 1

    total = int(confusion.sum())
    print(f"Images: {total}")
    print(f"Raw top-1 accuracy: {np.trace(confusion) / total:.2%}")
    accepted_total = int(accepted_confusion.sum())
    accepted_correct = int(np.trace(accepted_confusion))
    print(f"Accepted coverage: {accepted_total / total:.2%}")
    print(f"Accuracy when accepted: {accepted_correct / max(accepted_total, 1):.2%}")
    print()
    print("class    raw_recall  raw_prec  accepted_recall  accepted_prec  rejected")
    for index, label in enumerate(classes):
        class_total = int(confusion[index].sum())
        raw_tp = int(confusion[index, index])
        precision = raw_tp / max(int(confusion[:, index].sum()), 1)
        accepted_tp = int(accepted_confusion[index, index])
        accepted_precision = accepted_tp / max(
            int(accepted_confusion[:, index].sum()), 1
        )
        print(
            f"{label:8s} {raw_tp / max(class_total, 1):10.2%}"
            f" {precision:9.2%} {accepted_tp / max(class_total, 1):15.2%}"
            f" {accepted_precision:14.2%}"
            f" {int(rejected[index]):8d}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
