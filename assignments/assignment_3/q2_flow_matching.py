from typing import Optional

import torch
from torch import nn


class FlowMatching:
    def __init__(self, velocity_model: nn.Module, device: torch.device, time_scale: int = 1000):
        super().__init__()
        self.velocity_model = velocity_model
        self.device = device
        self.time_scale = time_scale

    def _model_time(self, t: torch.Tensor, batch_size: int) -> torch.Tensor:
        if t.ndim == 0:
            t = t.expand(batch_size)
        elif t.ndim > 1:
            t = t.reshape(batch_size)
        return (t * self.time_scale).long()

    def predict_velocity(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_model = self._model_time(t.to(x_t.device), x_t.shape[0])
        return self.velocity_model(x_t, t_model)

    def sample_xt(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # STUDENT TODO START
        # Straight-line interpolation path x_t = (1 - t) x_0 + t x_1.
        # x0 shape: (batch_size, channels, height, width)
        # x1 shape: same as x0
        # t shape: (batch_size, 1, 1, 1)
        # return shape: same as x0
        # ==========================
        # TODO: Write your code here
        # ==========================
        raise NotImplementedError("Implement sample_xt in q2_flow_matching.py.")
        # STUDENT TODO END
        return x_t

    def compute_conditional_velocity(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        # STUDENT TODO START
        # Conditional velocity for the straight-line path.
        # x0 shape: (batch_size, channels, height, width)
        # x1 shape: same as x0
        # return shape: same as x0
        # ==========================
        # TODO: Write your code here
        # ==========================
        raise NotImplementedError("Implement compute_conditional_velocity in q2_flow_matching.py.")
        # STUDENT TODO END
        return u_t

    def loss(
        self, x0: torch.Tensor, noise: Optional[torch.Tensor] = None, set_seed: bool = False
    ) -> torch.Tensor:
        if set_seed:
            torch.manual_seed(42)

        batch_size = x0.shape[0]
        dim = list(range(1, x0.ndim))
        t = torch.rand((batch_size, 1, 1, 1), device=x0.device)
        if noise is None:
            noise = torch.randn_like(x0)

        # STUDENT TODO START
        # Flow Matching training loss.
        # x0 shape: (batch_size, channels, height, width)
        # noise shape: same as x0
        # return: scalar loss tensor
        # ==========================
        # TODO: Write your code here
        # ==========================
        raise NotImplementedError("Implement loss in q2_flow_matching.py.")
        # STUDENT TODO END
        return loss

    def euler_step(self, x_t: torch.Tensor, t: float, dt: float) -> torch.Tensor:
        # STUDENT TODO START
        # One Euler ODE step.
        # x_t shape: (batch_size, channels, height, width)
        # t: current scalar time
        # dt: scalar step size
        # return shape: same as x_t
        # ==========================
        # TODO: Write your code here
        # ==========================
        raise NotImplementedError("Implement euler_step in q2_flow_matching.py.")
        # STUDENT TODO END
        return x_next

    def euler_sample(self, noise: torch.Tensor, n_steps: int) -> torch.Tensor:
        dt = -1.0 / n_steps
        x_t = noise
        for step in range(n_steps):
            t = 1.0 - step / n_steps
            x_t = self.euler_step(x_t, t, dt)
        return x_t

    def midpoint_step(self, x_t: torch.Tensor, t: float, dt: float) -> torch.Tensor:
        # STUDENT TODO START
        # One midpoint ODE step.
        # x_t shape: (batch_size, channels, height, width)
        # t: current scalar time
        # dt: scalar step size
        # return shape: same as x_t
        # ==========================
        # TODO: Write your code here
        # ==========================
        raise NotImplementedError("Implement midpoint_step in q2_flow_matching.py.")
        # STUDENT TODO END
        return x_next

    def midpoint_sample(self, noise: torch.Tensor, n_steps: int) -> torch.Tensor:
        dt = -1.0 / n_steps
        x_t = noise
        for step in range(n_steps):
            t = 1.0 - step / n_steps
            x_t = self.midpoint_step(x_t, t, dt)
        return x_t
