"""Experiment planner that pins the patch and batch size.

The benchmark compares eleven architectures, so they all have to see the same
amount of context and the same number of samples per step. This planner runs
nnU-Net's normal planning (target spacing, normalisation, resampling, and the
U-Net topology) and then overrides ``patch_size`` and ``batch_size`` with the
fixed values from ``nnunetv2.unified.config``, recomputing the network topology
for the new patch size so nnU-Net's own U-Net stays consistent with the plans.

Use it with::

    nnUNetv2_plan_and_preprocess -d 1 -pl nnUNetPlannerUnified -c 2d 3d_fullres
"""
from typing import List, Tuple, Union

import numpy as np

from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner
from nnunetv2.experiment_planning.experiment_planners.network_topology import get_pool_and_conv_props
from nnunetv2.unified.config import BATCH_SIZE_2D, BATCH_SIZE_3D, PATCH_SIZE_2D, PATCH_SIZE_3D


class nnUNetPlannerUnified(ExperimentPlanner):
    """Same plans as the default planner, with a fixed patch and batch size."""

    def __init__(self, dataset_name_or_id, gpu_memory_target_in_GB: float = 8,
                 preprocessor_name: str = 'DefaultPreprocessor',
                 plans_name: str = 'nnUNetPlans',
                 overwrite_target_spacing=None,
                 suppress_transpose: bool = False):
        # keep the default plans identifier so that every trainer, and nnU-Net's own
        # tooling, finds the plans without extra flags
        super().__init__(dataset_name_or_id, gpu_memory_target_in_GB, preprocessor_name,
                         plans_name, overwrite_target_spacing, suppress_transpose)

    def get_plans_for_configuration(self,
                                    spacing: Union[np.ndarray, Tuple[float, ...], List[float]],
                                    median_shape: Union[np.ndarray, Tuple[int, ...]],
                                    data_identifier: str,
                                    approximate_n_voxels_dataset: float,
                                    _cache: dict) -> dict:
        plan = super().get_plans_for_configuration(spacing, median_shape, data_identifier,
                                                   approximate_n_voxels_dataset, _cache)

        dim = len(spacing)
        if dim == 3:
            patch_size = list(PATCH_SIZE_3D)
            batch_size = BATCH_SIZE_3D
        elif dim == 2:
            patch_size = list(PATCH_SIZE_2D)
            batch_size = BATCH_SIZE_2D
        else:  # pragma: no cover
            raise RuntimeError(f'unexpected number of spatial dimensions: {dim}')

        # 3d_lowres would need its own (smaller) patch size and is not part of the
        # benchmark; leave it exactly as nnU-Net planned it.
        if '3d_lowres' in data_identifier:
            return plan

        # Recompute the U-Net topology for the pinned patch size. get_pool_and_conv_props
        # may round the patch size up to a multiple of 2**num_pool; assert that it did not,
        # because a silently different patch size would break the comparison.
        num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, adjusted_patch_size, _ = \
            get_pool_and_conv_props(spacing, patch_size, self.UNet_featuremap_min_edge_length, 999999)
        assert list(adjusted_patch_size) == patch_size, (
            f'the pinned patch size {patch_size} is not compatible with the target spacing {spacing}; '
            f'nnU-Net wants {list(adjusted_patch_size)} instead. Pick a patch size whose axes are '
            f'divisible by the required powers of two.')

        num_stages = len(pool_op_kernel_sizes)
        max_num_features = self.UNet_max_features_2d if dim == 2 else self.UNet_max_features_3d
        arch = plan['architecture']['arch_kwargs']
        arch.update({
            'n_stages': num_stages,
            'features_per_stage': tuple(min(max_num_features, self.UNet_base_num_features * 2 ** i)
                                        for i in range(num_stages)),
            'kernel_sizes': conv_kernel_sizes,
            'strides': pool_op_kernel_sizes,
            'n_conv_per_stage': self.UNet_blocks_per_stage_encoder[:num_stages],
            'n_conv_per_stage_decoder': self.UNet_blocks_per_stage_decoder[:num_stages - 1],
        })

        plan['patch_size'] = patch_size
        plan['batch_size'] = batch_size
        print(f'[unified planner] {data_identifier}: patch_size pinned to {patch_size}, '
              f'batch_size pinned to {batch_size} ({num_stages} stages)')
        return plan
