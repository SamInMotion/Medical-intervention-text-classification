"""BiomedBERT (PubMedBERT) sequence classification for Cohen benchmark.

Mirrors the predict_proba interface used by src/models.py so it can drop into
the existing cohen_pipeline.py text-mode comparison. The four text modes from
cohen_pipeline.py (abstract / title_abstract / title_abstract_mesh /
title_abstract_automesh) work without modification — only the classifier
changes.

Note on the model name: Microsoft renamed PubMedBERT to BiomedBERT in 2024.
The canonical identifier is now microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract.
The older PubMedBERT name still redirects but may stop working at some point.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

logger = logging.getLogger(__name__)

MODEL_ID = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
MAX_LEN = 512  # BERT-base hard limit; Cohen abstracts mostly fit


@dataclass
class BertConfig:
    """Training config. Defaults tuned for Cohen-sized corpora on modest hardware."""

    model_id: str = MODEL_ID
    max_length: int = MAX_LEN
    learning_rate: float = 2e-5
    epochs: int = 3
    train_batch_size: int = 8
    eval_batch_size: int = 16
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    seed: int = 42
    fp16: bool = False  # set True on GPU
    output_dir: str = "./bert_runs"
    # Class weighting for imbalanced data. Options:
    #   None        - no weighting (default)
    #   "balanced"  - compute from train labels: n / (n_classes * count_per_class)
    #   {0: w0, 1: w1} - explicit per-class weights
    class_weight: Union[str, dict, None] = None


class _TextDataset(Dataset):
    """Wraps texts + labels for HuggingFace Trainer."""

    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors=None,
        )
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def _resolve_class_weights(
    labels: list[int],
    spec: Union[str, dict, None],
) -> Union[list[float], None]:
    """Convert class_weight spec into a [w0, w1] list, or None if no weighting.

    Mirrors sklearn's compute_class_weight('balanced', ...) formula:
        weight[c] = n_samples / (n_classes * count[c])
    """
    if spec is None:
        return None
    if isinstance(spec, dict):
        # explicit weights — preserve key order
        return [float(spec.get(0, 1.0)), float(spec.get(1, 1.0))]
    if spec == "balanced":
        n = len(labels)
        n_pos = sum(labels)
        n_neg = n - n_pos
        if n_pos == 0 or n_neg == 0:
            logger.warning("Single-class training data; skipping class weighting")
            return None
        w_neg = n / (2.0 * n_neg)
        w_pos = n / (2.0 * n_pos)
        logger.info(
            "Class weights (balanced): [neg=%.3f, pos=%.3f] from %d neg / %d pos",
            w_neg, w_pos, n_neg, n_pos,
        )
        return [w_neg, w_pos]
    raise ValueError(f"Unsupported class_weight spec: {spec!r}")


class _WeightedLossTrainer(Trainer):
    """Trainer subclass that uses weighted cross-entropy.

    Required for the Cohen benchmark because Statins (5.5% inclusion) and
    similar topics have severe class imbalance. Without weighting, BERT
    converges to the majority-class prediction and produces near-random
    discrimination at the WSS@95 threshold.
    """

    def __init__(self, *args, class_weights: Union[list[float], None] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self._class_weights is not None:
            weight = torch.tensor(
                self._class_weights,
                device=logits.device,
                dtype=logits.dtype,
            )
            loss_fct = nn.CrossEntropyLoss(weight=weight)
        else:
            loss_fct = nn.CrossEntropyLoss()

        loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


class BiomedBertClassifier:
    """Fine-tune BiomedBERT for binary screening classification.

    Usage matches existing models.py pattern:

        clf = BiomedBertClassifier(BertConfig())
        clf.fit(train_texts, train_labels)
        probs = clf.predict_proba(test_texts)  # shape (n, 2)

    Designed to drop into cohen_pipeline.py's per-fold loop. The text-mode
    decision (abstract / title+abstract / +expert MeSH / +auto MeSH) happens
    upstream — this class just consumes whatever text it's given.
    """

    def __init__(self, config: BertConfig | None = None):
        self.config = config or BertConfig()
        self._tokenizer = None
        self._model = None
        self._trainer = None

    def _ensure_loaded(self) -> None:
        if self._tokenizer is None:
            logger.info("Loading tokenizer: %s", self.config.model_id)
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)

    def fit(self, texts: list[str], labels: list[int]) -> "BiomedBertClassifier":
        self._ensure_loaded()

        # Fresh model per fit — avoids leaking weights across k-fold iterations
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_id,
            num_labels=2,
        )

        train_dataset = _TextDataset(
            texts, labels, self._tokenizer, self.config.max_length
        )

        # transformers 5.x deprecated warmup_ratio in favor of warmup_steps.
        # Compute steps from ratio so the config field stays human-readable.
        steps_per_epoch = max(
            1, (len(train_dataset) + self.config.train_batch_size - 1) // self.config.train_batch_size
        )
        total_steps = steps_per_epoch * self.config.epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)

        args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.train_batch_size,
            per_device_eval_batch_size=self.config.eval_batch_size,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_steps=warmup_steps,
            logging_steps=50,
            save_strategy="no",  # don't litter disk with checkpoints during k-fold
            report_to=[],  # disable wandb/tensorboard auto-detection
            seed=self.config.seed,
            fp16=self.config.fp16,
            disable_tqdm=False,
        )

        self._trainer = _WeightedLossTrainer(
            model=self._model,
            args=args,
            train_dataset=train_dataset,
            class_weights=_resolve_class_weights(labels, self.config.class_weight),
        )

        self._trainer.train()
        return self

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Return P(label=0), P(label=1) for each input, shape (n, 2).

        Matches sklearn convention so it's drop-in compatible with
        evaluation.py's WSS@95% computation.
        """
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Call fit() before predict_proba()")

        # Dummy labels — we only need logits, not loss
        dummy_dataset = _TextDataset(
            texts, [0] * len(texts), self._tokenizer, self.config.max_length
        )

        preds = self._trainer.predict(dummy_dataset)
        logits = torch.tensor(preds.predictions)
        probs = torch.softmax(logits, dim=-1).numpy()
        return probs

    def predict(self, texts: list[str]) -> np.ndarray:
        return np.argmax(self.predict_proba(texts), axis=1)


def smoke_test() -> None:
    """Verify the model loads and runs on a handful of examples.

    Run with: python -m src.bert_models
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logger.info("Smoke test starting...")

    train_texts = [
        "Atorvastatin reduced LDL cholesterol by 38 percent in a randomized trial of 240 patients.",
        "This case report describes a single patient with statin-induced rhabdomyolysis.",
        "Editorial: the role of statins in primary prevention remains debated.",
        "Meta-analysis of 14 RCTs of HMG-CoA reductase inhibitors for cardiovascular prevention.",
    ]
    train_labels = [1, 0, 0, 1]  # 1 = included, 0 = excluded

    test_texts = [
        "Randomized controlled trial of rosuvastatin in 1200 patients with hyperlipidemia.",
        "Letter to the editor regarding statin myopathy guidelines.",
    ]

    config = BertConfig(
        epochs=1,
        train_batch_size=2,
        class_weight="balanced",  # exercises the weighted-loss path
    )
    clf = BiomedBertClassifier(config)
    clf.fit(train_texts, train_labels)
    probs = clf.predict_proba(test_texts)

    logger.info("Test predictions (P(included)):")
    for text, p in zip(test_texts, probs):
        logger.info("  %.3f | %s", p[1], text[:80])

    logger.info("Smoke test complete. Model loads, trains, predicts.")


if __name__ == "__main__":
    smoke_test()
