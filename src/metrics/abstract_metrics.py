import torch
from torch import Tensor
from torch.nn import functional as F
from torchmetrics import Metric, MeanSquaredError, MeanAbsoluteError


class TrainAbstractMetricsDiscrete(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, masked_pred_X, masked_pred_E, true_X, true_E, log: bool):
        pass

    def reset(self):
        pass

    def log_epoch_metrics(self):
        return None, None


class TrainAbstractMetrics(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, masked_pred_epsX, masked_pred_epsE, pred_y, true_epsX, true_epsE, true_y, log):
        pass

    def reset(self):
        pass

    def log_epoch_metrics(self):
        return None, None


class SumExceptBatchMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total_value', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, values) -> None:
        self.total_value += torch.sum(values)
        self.total_samples += values.shape[0]

    def compute(self):
        return self.total_value / self.total_samples


class SumExceptBatchMSE(MeanSquaredError):
    def __init__(self):
        super().__init__()
        
    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        assert preds.shape == target.shape
        sum_squared_error, n_obs = self._mean_squared_error_update(preds, target)

        self.sum_squared_error += sum_squared_error
        self.total += n_obs

    def _mean_squared_error_update(self, preds: Tensor, target: Tensor):
        """ Updates and returns variables required to compute Mean Squared Error. Checks for same shape of input
        tensors.
            preds: Predicted tensor
            target: Ground truth tensor
        """
        diff = preds - target
        sum_squared_error = torch.sum(diff * diff)
        n_obs = preds.shape[0]
        return sum_squared_error, n_obs
    
class SumExceptBatchMAE(MeanAbsoluteError):
    def __init__(self):
        super().__init__()
        
    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        assert preds.shape == target.shape
        sum_absolute_error, n_obs = self._mean_absolute_error_update(preds, target)

        self.sum_abs_error += sum_absolute_error
        self.total += n_obs

    def _mean_absolute_error_update(self, preds: Tensor, target: Tensor):
        """ Updates and returns variables required to compute Mean absolute Error. Checks for same shape of input
        tensors.
            preds: Predicted tensor
            target: Ground truth tensor
        """
        sum_absolute_error = torch.sum(torch.abs(preds - target))
        n_obs = target.numel()
        return sum_absolute_error, n_obs

class SumExceptBatchWasserstein(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total_value', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, pos_pred, pos_true) -> None:
        self.total_samples += pos_pred.size(0)
        # pos_pred = pred[node_mask.bool()].view(-1, 3)  # Use masked positions
        # pos_true = true[node_mask.bool()].view(-1, 3)
        
        # Pairwise distances
        dist_pred = torch.cdist(pos_pred, pos_pred, p=2)  # Predicted pairwise distances
        dist_true = torch.cdist(pos_true, pos_true, p=2)  # Ground truth pairwise distances
        
        # Wasserstein loss based on pairwise distances
        self.total_value += torch.abs(dist_pred - dist_true).mean()

    def compute(self):
        return self.total_value / self.total_samples

class SumExceptBatchKL(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total_value', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, p, q) -> None:
        self.total_value += F.kl_div(q, p, reduction='sum')
        self.total_samples += p.size(0)

    def compute(self):
        return self.total_value / self.total_samples


class CrossEntropyMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total_ce', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """ Update state with predictions and targets.
            preds: Predictions from model   (bs * n, d) or (bs * n * n, d)
            target: Ground truth values     (bs * n, d) or (bs * n * n, d). """
        target = torch.argmax(target, dim=-1)
        output = F.cross_entropy(preds, target, reduction='sum')
        self.total_ce += output
        self.total_samples += preds.size(0)

    def compute(self):
        return self.total_ce / self.total_samples


class HuberLossMetric(Metric):
    def __init__(self, delta=1.0):
        """ This metric is used to track the marginal predicted probability of a class during training. """
        super().__init__()
        self.add_state('prob', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.delta = delta

    def update(self, preds: Tensor, target: Tensor) -> None:
        residual = torch.abs(preds - target)
        mask = residual < self.delta
        output = torch.where(mask, 0.5 * residual ** 2, self.delta * residual -0.5 * self.delta ** 2)
        self.prob += output
        self.total += preds.numel()

    def compute(self):
        return self.prob / self.total


class NLL(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total_nll', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, batch_nll) -> None:
        self.total_nll += torch.sum(batch_nll)
        self.total_samples += batch_nll.numel()

    def compute(self):
        return self.total_nll / self.total_samples
