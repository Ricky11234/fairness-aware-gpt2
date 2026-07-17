"""Fairness-Aware GPT-2 for Paraphrase Detection.

Modules:
    identity         Substitution lexicons, swap logic, subgroup assignment
    model            GPT-2 classifier (QQP/SST/CFIMDB) and sonnet LM
    data             Dataset loading, counterfactual augmentation, collation
    train            Paraphrase training (baseline / cda / cda_reg)
    train_sentiment  SST and CFIMDB training
    train_sonnet     Sonnet LM fine-tuning and CHRF evaluation
    evaluate         Accuracy, subgroup accuracy gap, prediction flip rate
    results          Reported-vs-reproduced results loading
"""

__version__ = "0.2.0"

__all__ = [
    "data",
    "evaluate",
    "identity",
    "model",
    "results",
    "train",
    "train_sentiment",
    "train_sonnet",
]
