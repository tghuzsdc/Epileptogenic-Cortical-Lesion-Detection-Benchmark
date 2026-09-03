"""How the ``-M*`` / ``-head2`` command line flags reach the trainer.

nnU-Net fixes the trainer's ``__init__`` signature and builds the network from a
static method, so there is no argument we could thread through. The flags are
therefore published as environment variables by the entry points before the
trainer is constructed. That also makes them survive the ``spawn`` used for DDP
and for the segmentation export pool.
"""
import os
from typing import Optional, Set

ENV_M = 'nnUNet_unified_M'
ENV_HEAD2 = 'nnUNet_unified_head2'
ENV_DUAL = 'nnUNet_unified_dual_head'


def set_training_setup(m: Optional[str], head2_file: Optional[str]) -> None:
    """Called from ``nnUNetv2_train`` once the arguments are parsed."""
    from nnunetv2.unified.subsets import M_NAMES

    if m is not None:
        assert m in M_NAMES, f'-M must be one of {M_NAMES}, got {m!r}'
        os.environ[ENV_M] = m
    else:
        os.environ.pop(ENV_M, None)

    if head2_file is not None:
        os.environ[ENV_HEAD2] = head2_file
    else:
        os.environ.pop(ENV_HEAD2, None)

    # A second output head is created when the protocol asks for it (M4) or when
    # the user explicitly supplies a head2 list.
    dual = (m == 'M4') or (head2_file is not None)
    os.environ[ENV_DUAL] = '1' if dual else '0'


def set_inference_setup(head2_file: Optional[str], dual_head: bool) -> None:
    """Called from ``nnUNetv2_predict``; ``dual_head`` comes from the checkpoint."""
    if head2_file is not None:
        os.environ[ENV_HEAD2] = head2_file
    else:
        os.environ.pop(ENV_HEAD2, None)
    os.environ[ENV_DUAL] = '1' if dual_head else '0'


def get_m() -> Optional[str]:
    value = os.environ.get(ENV_M, '').strip()
    return value or None


def get_head2_file() -> Optional[str]:
    value = os.environ.get(ENV_HEAD2, '').strip()
    return value or None


def get_head2_cases() -> Optional[Set[str]]:
    from nnunetv2.unified.subsets import read_head2_list
    path = get_head2_file()
    return read_head2_list(path) if path else None


def is_dual_head() -> bool:
    return os.environ.get(ENV_DUAL, '0') == '1'


def set_dual_head(enabled: bool) -> None:
    os.environ[ENV_DUAL] = '1' if enabled else '0'
