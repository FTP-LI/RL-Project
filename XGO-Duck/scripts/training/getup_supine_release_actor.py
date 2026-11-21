"""Apply the trained residual only to scenes latched as supine at release."""

import torch


class SupineReleaseActor(torch.nn.Module):
    """One frozen release decision per fixed first-episode evaluation."""

    def __init__(self, parent, candidate, control_counter, release_counter, threshold):
        super().__init__()
        self.parent = parent
        self.candidate = candidate
        self.control_counter = control_counter
        self.release_counter = release_counter
        self.threshold = threshold
        self.supine_at_release = None

    def forward(self, observations):
        if self.supine_at_release is None:
            if self.control_counter() != self.release_counter:
                raise ValueError('The first policy observation must be the shared release')
            gravity = observations['actor'][:, 3:6].to(torch.float64)
            magnitude = -gravity[:, 0] / torch.linalg.vector_norm(gravity, dim=-1).clamp_min(1e-6)
            self.supine_at_release = magnitude >= self.threshold
        parent = self.parent(observations)
        candidate = self.candidate(observations)
        return torch.where(self.supine_at_release[:, None], candidate, parent)
