"""Two segmentation heads on one shared trunk.

The trick that makes this work for all eleven architectures without touching any
of them: build the network with ``2 * C`` output channels instead of ``C``.
Channels ``[0:C]`` are head1, channels ``[C:2C]`` are head2. Everything before
the very last 1x1(x1) convolution -- i.e. the entire encoder, decoder and skip
structure -- is therefore shared, and the two heads are exactly the two halves of
that last layer's weight matrix -- a second output head sharing the trunk, with no
architecture-specific surgery.

Routing is per sample, so a batch may freely mix cohort_a and cohort_b cases:
the trunk runs once and each sample then keeps only its own head's channels.
"""
from typing import List, Optional, Union

import torch
from torch import nn


class DualHeadWrapper(nn.Module):
    """Wraps a network whose output has ``2 * num_classes`` channels."""

    def __init__(self, network: nn.Module, num_classes: int):
        super().__init__()
        self.network = network
        self.num_classes = num_classes
        # Used at inference, where the head is fixed for the whole volume.
        # 0 -> head1, 1 -> head2.
        self.active_head: int = 0
        # Set per batch during training (LongTensor of shape [B]).
        self._head_index: Optional[torch.Tensor] = None

    # -- routing ---------------------------------------------------------------
    def set_head_index(self, head_index: Optional[torch.Tensor]):
        """Per-sample routing for the next forward pass (training)."""
        self._head_index = head_index

    def set_active_head(self, head: int):
        """Whole-volume routing (inference / sliding window)."""
        assert head in (0, 1), f'head must be 0 or 1, got {head}'
        self.active_head = head

    # -- helpers ---------------------------------------------------------------
    def _select(self, t: torch.Tensor) -> torch.Tensor:
        c = self.num_classes
        assert t.shape[1] == 2 * c, (
            f'dual-head network produced {t.shape[1]} channels, expected {2 * c}')
        if self._head_index is None:
            offset = self.active_head * c
            return t[:, offset:offset + c]

        head = self._head_index.to(t.device)
        assert head.shape[0] == t.shape[0], (
            f'head index has {head.shape[0]} entries but the batch has {t.shape[0]} samples')
        # gather channel block [head*c : head*c + c] for every sample
        idx = head.view(-1, 1) * c + torch.arange(c, device=t.device).view(1, -1)  # [B, C]
        idx = idx.view(t.shape[0], c, *([1] * (t.ndim - 2))).expand(-1, -1, *t.shape[2:])
        return torch.gather(t, 1, idx)

    # -- nn.Module -------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        out = self.network(x)
        if isinstance(out, (list, tuple)):
            return [self._select(o) for o in out]
        return self._select(out)

    # Deep supervision is toggled on the wrapped network, not on the wrapper.
    @property
    def deep_supervision(self):
        return getattr(self.network, 'deep_supervision', None)
