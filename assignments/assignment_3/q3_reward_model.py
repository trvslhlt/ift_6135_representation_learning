from typing import Dict, Iterable, Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel

from q3_utils import move_batch_to_device, summarize_metrics


class RewardModel(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        # STUDENT TODO START
        # Reward model on top of a pretrained GPT-2 backbone.
        # model_name: local path or HF model identifier
        # Save the transformer backbone and a scalar reward head on self.
        # ==========================
        # TODO: Write your code here
        # ==========================
        raise NotImplementedError("Implement RewardModel.__init__ in q3_reward_model.py.")
        # STUDENT TODO END

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # STUDENT TODO START
        # Score each prompt-response sequence with a scalar reward.
        # input_ids shape: (batch_size, sequence_length)
        # attention_mask shape: (batch_size, sequence_length)
        # return shape: (batch_size,)
        # ==========================
        # TODO: Write your code here
        # ==========================
        raise NotImplementedError("Implement RewardModel.forward in q3_reward_model.py.")
        # STUDENT TODO END
        return rewards


def compute_preference_loss(
    rewards_chosen: torch.Tensor,
    rewards_rejected: torch.Tensor,
) -> torch.Tensor:
    # STUDENT TODO START
    # Bradley-Terry preference loss.
    # rewards_chosen shape: (batch_size,)
    # rewards_rejected shape: (batch_size,)
    # return: scalar loss tensor
    # ==========================
    # TODO: Write your code here
    # ==========================
    raise NotImplementedError("Implement compute_preference_loss in q3_reward_model.py.")
    # STUDENT TODO END
    return loss


def compute_reward_accuracy(
    rewards_chosen: torch.Tensor,
    rewards_rejected: torch.Tensor,
) -> torch.Tensor:
    # STUDENT TODO START
    # Preference accuracy.
    # rewards_chosen shape: (batch_size,)
    # rewards_rejected shape: (batch_size,)
    # return: scalar accuracy tensor
    # ==========================
    # TODO: Write your code here
    # ==========================
    raise NotImplementedError("Implement compute_reward_accuracy in q3_reward_model.py.")
    # STUDENT TODO END
    return accuracy


class RewardModelTrainer:
    def __init__(
        self,
        model: RewardModel,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.device = torch.device(device) if device is not None else self._infer_device(model)
        self.model.to(self.device)
        self.optimizer = optimizer

    @staticmethod
    def _infer_device(model: nn.Module) -> torch.device:
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch = move_batch_to_device(batch, self.device)
        # STUDENT TODO START
        # One reward-model forward pass on a batch of chosen/rejected sequences.
        # batch contains chosen/rejected input ids and attention masks.
        # return keys: `loss` and `accuracy`
        # ==========================
        # TODO: Write your code here
        # ==========================
        raise NotImplementedError("Implement RewardModelTrainer.train_step in q3_reward_model.py.")
        # STUDENT TODO END
        return {"loss": loss, "accuracy": accuracy}

    def optimizer_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        if self.optimizer is None:
            raise ValueError("RewardModelTrainer.optimizer_step requires an optimizer.")
        self.model.train()
        metrics = self.train_step(batch)
        self.optimizer.zero_grad()
        metrics["loss"].backward()
        self.optimizer.step()
        return {key: float(value.detach().cpu().item()) for key, value in metrics.items()}

    def evaluate_loader(self, dataloader: Iterable[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        self.model.eval()
        metric_history = []
        with torch.no_grad():
            for batch in dataloader:
                metrics = self.train_step(batch)
                metric_history.append(
                    {key: float(value.detach().cpu().item()) for key, value in metrics.items()}
                )
        return summarize_metrics(metric_history)
