from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from q3_utils import move_batch_to_device, summarize_metrics


def compute_log_probs(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    # Compute sequence log-probabilities over response tokens only.
    # input_ids shape: (batch_size, sequence_length)
    # attention_mask shape: (batch_size, sequence_length)
    # response_mask shape: (batch_size, sequence_length)
    # return shape: (batch_size,)
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits  # (batch, seq_len, vocab)

    # shift: logits[i] predicts token[i+1]
    shifted_logits = logits[:, :-1, :] # drop last logit (batch, seq_len-1, vocab)
    shifted_labels = input_ids[:, 1:] # drop first label (batch, seq_len-1)
    shifted_response_mask = response_mask[:, 1:] # drop first response mask (batch, seq_len-1)

    log_probs = F.log_softmax(shifted_logits, dim=-1)  # (batch, seq_len-1, vocab)

    # gather log-prob of each target token
    token_log_probs = log_probs.gather(
        dim=-1, index=shifted_labels.unsqueeze(-1)
    ).squeeze(-1)  # (batch, seq_len-1)

    # sum over response tokens only
    sequence_log_probs = (token_log_probs * shifted_response_mask).sum(dim=-1)  # (batch,)
    return sequence_log_probs


def compute_dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # DPO loss, reward margin, and preference accuracy.
    # all log-probability inputs have shape: (batch_size,)
    # beta: scalar temperature / regularization strength
    # returns:
    #   loss: scalar tensor
    #   reward_margin: (batch_size,)
    #   accuracy: scalar tensor
    a = (policy_chosen_logps - ref_chosen_logps) - (policy_rejected_logps - ref_rejected_logps)
    loss = -F.logsigmoid(beta * a).mean()
    reward_margin = beta * a
    accuracy = (reward_margin > 0).float().mean()
    return loss, reward_margin, accuracy


def compute_implicit_reward(
    policy_logps: torch.Tensor,
    ref_logps: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    # Helper provided for analysis and notebook logging.
    implicit_rewards = beta * (policy_logps - ref_logps)
    return implicit_rewards


class DPOTrainer:
    def __init__(
        self,
        policy_model: nn.Module,
        reference_model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        beta: float = 0.1,
        device: Optional[torch.device] = None,
    ):
        self.policy_model = policy_model
        self.reference_model = reference_model
        self.optimizer = optimizer
        self.beta = beta
        self.device = torch.device(device) if device is not None else self._infer_device(policy_model)
        
        self.policy_model.to(self.device)
        self.reference_model.to(self.device)
        self.reference_model.eval()
        for parameter in self.reference_model.parameters():
            parameter.requires_grad = False

    @staticmethod
    def _infer_device(model: nn.Module) -> torch.device:
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch = move_batch_to_device(batch, self.device)
        # One DPO forward pass on a batch of preference pairs.
        # batch contains chosen/rejected sequences, attention masks, and response masks.
        # return:
        #   loss: scalar tensor
        #   metrics: dict with `reward_margin` and `accuracy`
        def _compute_logps(model, prefix):
            return compute_log_probs(
                model=model, 
                input_ids=batch[f"{prefix}_input_ids"], 
                attention_mask=batch[f"{prefix}_attention_mask"], 
                response_mask=batch[f"{prefix}_response_mask"],
            )
        
        policy_chosen_logps = _compute_logps(self.policy_model, "chosen")
        policy_rejected_logps = _compute_logps(self.policy_model, "rejected")

        with torch.no_grad():
            ref_chosen_logps = _compute_logps(self.reference_model, "chosen")
            ref_rejected_logps = _compute_logps(self.reference_model, "rejected")

        loss, reward_margin, accuracy = compute_dpo_loss(
            policy_chosen_logps, 
            policy_rejected_logps, 
            ref_chosen_logps, 
            ref_rejected_logps,
            self.beta,
        )
        metrics = {
            "reward_margin": reward_margin.mean(),
            "accuracy": accuracy,
        }
        return loss, metrics

    def optimizer_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        if self.optimizer is None:
            raise ValueError("DPOTrainer.optimizer_step requires an optimizer.")
        self.policy_model.train()
        loss, metrics = self.compute_loss(batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        output = {"loss": float(loss.detach().cpu().item())}
        output.update({key: float(value.detach().cpu().item()) for key, value in metrics.items()})
        return output

    def evaluate_loader(self, dataloader: Iterable[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        self.policy_model.eval()
        metric_history = []
        with torch.no_grad():
            for batch in dataloader:
                loss, metrics = self.compute_loss(batch)
                record = {"loss": float(loss.detach().cpu().item())}
                record.update({key: float(value.detach().cpu().item()) for key, value in metrics.items()})
                metric_history.append(record)
        return summarize_metrics(metric_history)
