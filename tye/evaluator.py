"""
Base evaluator class, which is inherited by all evaluators.
"""
from typing import List, Optional, Dict
from abc import abstractmethod, ABC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)
import numpy as np
import torch


def accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    return accuracy_score(y_true=y_true, y_pred=y_pred)


def f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    return f1_score(y_true=y_true, y_pred=y_pred)


def mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    return mean_absolute_error(y_true=y_true, y_pred=y_pred)


def mse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    return mean_squared_error(y_true=y_true, y_pred=y_pred)


def mape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    return mean_absolute_percentage_error(y_true=y_true, y_pred=y_pred)


class BaseEvaluator(ABC):
    METRIC_FUNCTIONS = {
        "accuracy": accuracy,
        "f1": f1,
        "mae": mae,
        "mse": mse,
        "mape": mape,
    }

    def __init__(
        self,
        metric_names: List[str],
        downstream_model_kwargs: Optional[dict] = None,
    ):
        self.metric_names = metric_names
        self.downstream_model_kwargs = (
            downstream_model_kwargs if downstream_model_kwargs else {}
        )

    def evaluate(self):
        raise NotImplementedError

    def compute_metrics(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        y_score: torch.Tensor,
    ) -> Dict[str, float]:
        return {
            metric_name: self.METRIC_FUNCTIONS[metric_name](
                y_true=y_true,
                y_pred=y_pred,
                y_score=y_score,
            )
            for metric_name in self._metric_names
        }

    @property
    def available_metrics(self) -> List[str]:
        return list(self.METRIC_FUNCTIONS.keys())

    @abstractmethod
    def evaluate(
        self,
        Z: torch.Tensor,
        Y: torch.Tensor,
        train_mask: torch.Tensor,
        test_mask: torch.Tensor,
        val_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        pass
