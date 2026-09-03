"""Top-k connected-component constraint applied *during* inference.

This is not nnU-Net post-processing: it runs inside
``convert_predicted_logits_to_segmentation_with_correct_shape``, i.e. while the
softmax probabilities of the case are still in memory, before anything is
written to disk.  Post-processing (``nnUNetv2_apply_postprocessing``) is left
untouched and can still be run on top if desired.

For every foreground label we

1. label the predicted mask into 26-connected components,
2. score each component from the probability map of that label,
3. keep the ``k`` highest scoring components and drop the rest.

Three scoring rules, selected by the ``-A`` / ``-B`` / ``-C`` inference flags:

``A``  the 95th percentile of the probabilities inside the component
``B``  the mean of the 50 highest probabilities inside the component
``C``  the mean of the probabilities above 0.3 inside the component

The flags travel to the export worker processes through environment variables
because nnU-Net exports segmentations from a ``spawn``-based process pool.
"""
import os
from typing import Optional

import numpy as np

ENV_MODE = 'nnUNet_cluster_constraint'
ENV_TOPK = 'nnUNet_cluster_topk'

MODES = ('A', 'B', 'C')
DEFAULT_TOPK = 3
#: number of highest-probability voxels averaged by rule B
B_NUM_VOXELS = 50
#: probability threshold used by rule C
C_THRESHOLD = 0.3


def get_mode() -> Optional[str]:
    mode = os.environ.get(ENV_MODE, '').strip().upper()
    return mode if mode in MODES else None


def get_topk() -> int:
    try:
        return int(os.environ.get(ENV_TOPK, DEFAULT_TOPK))
    except ValueError:
        return DEFAULT_TOPK


def set_mode(mode: Optional[str], topk: int = DEFAULT_TOPK) -> None:
    """Publish the setting so that spawned export workers inherit it."""
    if mode is None:
        os.environ.pop(ENV_MODE, None)
        os.environ.pop(ENV_TOPK, None)
        return
    mode = mode.upper()
    assert mode in MODES, f'cluster constraint mode must be one of {MODES}, got {mode!r}'
    os.environ[ENV_MODE] = mode
    os.environ[ENV_TOPK] = str(int(topk))


def _label_26(mask: np.ndarray):
    """26-connected components. Uses cc3d when available, scipy otherwise."""
    try:
        import cc3d
        labels, n = cc3d.connected_components(mask.astype(np.uint8), connectivity=26,
                                              return_N=True)
        return labels, int(n)
    except ImportError:
        from scipy.ndimage import label
        structure = np.ones((3, 3, 3), dtype=np.uint8)  # 26-connectivity
        labels, n = label(mask, structure=structure)
        return labels, int(n)


def _score(values: np.ndarray, mode: str) -> float:
    """Confidence of one component from the probabilities of its voxels."""
    if values.size == 0:
        return -np.inf
    if mode == 'A':
        return float(np.percentile(values, 95))
    if mode == 'B':
        k = min(B_NUM_VOXELS, values.size)
        # partial sort is enough and much cheaper than a full sort
        top = np.partition(values, values.size - k)[values.size - k:]
        return float(top.mean())
    if mode == 'C':
        above = values[values > C_THRESHOLD]
        # a component with nothing above the threshold scores by its maximum, so
        # that it still ranks below any component that does clear the threshold
        return float(above.mean()) if above.size else float(values.max()) - 1.0
    raise ValueError(f'unknown mode {mode!r}')


def apply_topk_cluster_constraint(segmentation: np.ndarray,
                                  probabilities: np.ndarray,
                                  foreground_labels,
                                  mode: str,
                                  topk: int = DEFAULT_TOPK,
                                  verbose: bool = False) -> np.ndarray:
    """Keep only the ``topk`` best-scoring 26-connected components per label.

    ``segmentation`` is the integer label map, ``probabilities`` the matching
    ``[num_classes, *spatial]`` softmax output.  Returns a new label map.
    """
    if segmentation.ndim != 3:
        # 2D configurations still export a 3D volume; anything else is unexpected
        if verbose:
            print(f'[cluster-constraint] skipped: segmentation has {segmentation.ndim} dimensions')
        return segmentation

    out = segmentation.copy()
    for label_value in foreground_labels:
        label_value = int(label_value)
        mask = segmentation == label_value
        if not mask.any():
            continue

        components, n = _label_26(mask)
        if n <= topk:
            if verbose:
                print(f'[cluster-constraint] label {label_value}: {n} component(s), '
                      f'nothing to remove')
            continue

        if label_value < probabilities.shape[0]:
            prob = np.asarray(probabilities[label_value], dtype=np.float32)
        else:  # pragma: no cover - defensive
            prob = mask.astype(np.float32)

        # Group the probabilities by component id. Only the foreground voxels take
        # part -- sorting the whole volume would be ~100x more work for nothing,
        # since the components live inside `mask` by construction.
        flat_components = components.reshape(-1)
        foreground = flat_components > 0
        comp_fg = flat_components[foreground]
        prob_fg = prob.reshape(-1)[foreground]

        order = np.argsort(comp_fg, kind='stable')
        comp_sorted = comp_fg[order]
        prob_sorted = prob_fg[order]
        ids = np.arange(1, n + 1)
        starts = np.searchsorted(comp_sorted, ids, side='left')
        ends = np.searchsorted(comp_sorted, ids, side='right')

        scores = np.empty(n, dtype=np.float64)
        for i in range(n):
            scores[i] = _score(prob_sorted[starts[i]:ends[i]], mode)

        keep = set((np.argsort(-scores)[:topk] + 1).tolist())
        drop = np.isin(components, list(keep), invert=True) & mask
        out[drop] = 0
        if verbose:
            kept = sorted(scores[np.argsort(-scores)[:topk]].tolist(), reverse=True)
            print(f'[cluster-constraint] label {label_value}: kept {topk}/{n} components '
                  f'(mode {mode}, scores {["%.4f" % s for s in kept]})')
    return out


def maybe_apply(segmentation: np.ndarray, probabilities, label_manager, verbose: bool = False):
    """Entry point used by the export path. No-op when no flag was given."""
    mode = get_mode()
    if mode is None or probabilities is None:
        return segmentation
    import torch
    if isinstance(segmentation, torch.Tensor):
        segmentation = segmentation.cpu().numpy()
    if isinstance(probabilities, torch.Tensor):
        probabilities = probabilities.cpu().numpy()
    return apply_topk_cluster_constraint(
        segmentation, probabilities, label_manager.foreground_labels, mode, get_topk(), verbose)
