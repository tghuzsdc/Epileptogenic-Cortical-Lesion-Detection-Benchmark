"""The eleven benchmark trainers.

One base class does all the work; the eleven concrete classes at the bottom of
the file only exist so that ``-tr <name>`` resolves to them and so that each run
gets its own results folder.  Everything that differs between them (network,
optimiser, schedule, deep supervision) is looked up in
``nnunetv2.unified.config.ARCHITECTURES``.

What this trainer adds on top of ``nnUNetTrainer``:

``-M0 .. -M4``  restrict training to one of the five defined training sets and
                build the 5-fold split over exactly those cases, stratified by
                cohort so both cohorts appear in every fold.
``-head2 FILE`` add a second output head.  The listed cases are trained through
                head2, everything else through head1, and the trunk is shared.
"""
import os
from typing import Optional, Set

import torch
from batchgenerators.utilities.file_and_folder_operations import isfile, join, load_json, save_json
from torch._dynamo import OptimizedModule

from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.unified import runtime, subsets
from nnunetv2.unified.builders import build_network, deep_supervision_scales, set_deep_supervision
from nnunetv2.unified.config import (
    AMP_DTYPE, AMP_OVERRIDE, NUM_EPOCHS, NUM_ITERATIONS_PER_EPOCH, get_spec)
from nnunetv2.unified.dual_head import DualHeadWrapper
from nnunetv2.unified.schedulers import ConstantLR, LinearWarmupCosineAnnealingLR, StepIterationLR
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager


class UnifiedTrainer(nnUNetTrainer):
    """Shared implementation for all eleven benchmark entries."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        spec = get_spec(self.__class__.__name__)
        self.spec = spec

        if spec.builder != 'nnunet' and len(self.configuration_manager.patch_size) != 3:
            raise RuntimeError(
                f'{spec.cli_name} is a 3D architecture; it cannot run the {configuration!r} '
                f'configuration. The benchmark runs the ten published architectures on '
                f'3d_fullres only, and nnU-Net itself on both 2d and 3d_fullres.')

        self.amp_dtype = AMP_OVERRIDE.get(spec.cli_name, AMP_DTYPE)
        self._apply_amp_dtype()

        self.initial_lr = spec.initial_lr
        self.weight_decay = spec.weight_decay
        self.enable_deep_supervision = spec.deep_supervision
        self.num_epochs = NUM_EPOCHS
        self.num_iterations_per_epoch = NUM_ITERATIONS_PER_EPOCH

        # --- training set and head routing, published by the entry point --------
        self.training_set: Optional[str] = runtime.get_m()
        self.head2_file: Optional[str] = runtime.get_head2_file()
        self.head2_cases: Optional[Set[str]] = runtime.get_head2_cases()
        self.dual_head: bool = runtime.is_dual_head()
        #: set by perform_actual_validation to run one head at a time
        self._validation_head: Optional[int] = None

        extra = ''.join(f' {k}={v}' for k, v in spec.optimizer_kwargs)
        if spec.optimizer == 'SGD':
            extra = f' momentum={spec.momentum} nesterov={spec.nesterov}' + extra
        self.print_to_log_file(
            f'[unified] architecture={spec.cli_name}  network={spec.builder}\n'
            f'[unified] optimizer={spec.optimizer} lr={spec.initial_lr} wd={spec.weight_decay}{extra} '
            f'scheduler={spec.scheduler} deep_supervision={spec.deep_supervision}\n'
            f'[unified] training set={self.training_set or "(dataset as-is)"} '
            f'dual_head={self.dual_head} head2_list={self.head2_file or "-"}\n'
            f'[unified] network source: {spec.source}\n'
            f'[unified] recipe: {spec.reference}',
            add_timestamp=False)

        if self.amp_dtype != 'fp16':
            self.print_to_log_file(
                f'[unified] training in {self.amp_dtype} instead of nnU-Net\'s default fp16'
                + (' (no gradient scaler: bf16 covers fp32\'s exponent range)'
                   if self.amp_dtype == 'bf16' else ' (autocast disabled)'))

        if spec.builder == 'vnet':
            total_iterations = self.num_epochs * self.num_iterations_per_epoch
            self.print_to_log_file(
                f'[unified] WARNING: V-Net decays the learning rate by 10x every 25K iterations, and this '
                f'run is {total_iterations} iterations long, so the LR reaches '
                f'{spec.initial_lr * 0.1 ** (total_iterations // 25000):.1e} by the end. That is what the '
                f'paper prescribes; pass max_decays to StepIterationLR in configure_optimizers if you '
                f'want to cap it.')

    def _apply_amp_dtype(self):
        """Point nnU-Net's own `autocast(..., enabled=True)` at the requested dtype.

        nnUNetTrainer.train_step opens the autocast context without naming a dtype, so
        it picks up the process-wide default. Setting that is enough -- no need to
        duplicate train_step/validation_step just to change the precision.
        """
        if self.device.type != 'cuda':
            return
        if self.amp_dtype == 'bf16':
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError('nnUNet_unified_amp=bf16 but this GPU has no bfloat16 support')
            dtype = torch.bfloat16
        elif self.amp_dtype == 'fp32':
            dtype = torch.float32
        elif self.amp_dtype == 'fp16':
            dtype = torch.float16
        else:
            raise ValueError(f'nnUNet_unified_amp must be fp16, bf16 or fp32, got {self.amp_dtype!r}')

        try:
            torch.set_autocast_dtype('cuda', dtype)          # torch >= 2.4
        except (AttributeError, TypeError):
            torch.set_autocast_gpu_dtype(dtype)              # older torch

        # A gradient scaler exists to keep fp16 gradients off the bottom of the range;
        # bf16 and fp32 do not need one, and nnU-Net's train_step already handles None.
        if self.amp_dtype != 'fp16':
            self.grad_scaler = None

    # -----------------------------------------------------------------------------
    # network
    # -----------------------------------------------------------------------------
    @classmethod
    def build_network_architecture(cls,
                                   plans_manager: PlansManager,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> torch.nn.Module:
        # Called both from initialize() and, at inference time, from the class.
        # Whether a second head exists is therefore read from the environment,
        # which nnUNetv2_predict fills in from the checkpoint.
        dual = runtime.is_dual_head()
        out_channels = num_output_channels * 2 if dual else num_output_channels

        network, _ = build_network(cls.__name__, plans_manager, configuration_manager,
                                   num_input_channels, out_channels, enable_deep_supervision)
        if dual:
            network = DualHeadWrapper(network, num_output_channels)
        return network

    def _get_deep_supervision_scales(self):
        return deep_supervision_scales(self.__class__.__name__, self.configuration_manager,
                                       self.enable_deep_supervision)

    def set_deep_supervision_enabled(self, enabled: bool):
        if not self.spec.deep_supervision:
            return  # architectures published without deep supervision keep it off
        mod = self.network.module if self.is_ddp else self.network
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod
        set_deep_supervision(mod, enabled)

    def _do_i_compile(self):
        # Eleven very different architectures (dynamic shapes in CoTr, einops in
        # nnFormer, a per-batch routing tensor in the dual-head wrapper) do not all
        # survive torch.compile. Opt in explicitly with nnUNet_compile=1.
        if os.environ.get('nnUNet_compile', '').lower() in ('true', '1', 't'):
            return super()._do_i_compile()
        return False

    def _dual_head_module(self) -> Optional[DualHeadWrapper]:
        if not self.dual_head:
            return None
        mod = self.network.module if self.is_ddp else self.network
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod
        return mod if isinstance(mod, DualHeadWrapper) else None

    # -----------------------------------------------------------------------------
    # optimisation
    # -----------------------------------------------------------------------------
    def configure_optimizers(self):
        spec = self.spec
        params = self.network.parameters()
        # extras the official implementation passes, e.g. MedNeXt's eps=1e-4 or
        # TransBTS's amsgrad=True
        extra = dict(spec.optimizer_kwargs)

        if spec.optimizer == 'SGD':
            optimizer = torch.optim.SGD(params, self.initial_lr, weight_decay=self.weight_decay,
                                        momentum=spec.momentum, nesterov=spec.nesterov, **extra)
        elif spec.optimizer == 'Adam':
            optimizer = torch.optim.Adam(params, self.initial_lr, weight_decay=self.weight_decay, **extra)
        elif spec.optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(params, self.initial_lr, weight_decay=self.weight_decay, **extra)
        else:
            raise ValueError(f'unknown optimizer {spec.optimizer!r}')

        if spec.scheduler == 'poly':
            scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        elif spec.scheduler == 'warmup_cosine':
            scheduler = LinearWarmupCosineAnnealingLR(optimizer, self.initial_lr, self.num_epochs,
                                                      warmup_epochs=50)
        elif spec.scheduler == 'step':
            scheduler = StepIterationLR(optimizer, self.initial_lr,
                                        iterations_per_epoch=self.num_iterations_per_epoch,
                                        step_iterations=25000, gamma=0.1)
        elif spec.scheduler == 'constant':
            scheduler = ConstantLR(optimizer, self.initial_lr)
        else:
            raise ValueError(f'unknown scheduler {spec.scheduler!r}')

        return optimizer, scheduler

    # -----------------------------------------------------------------------------
    # data split: M0-M4 plus cohort-stratified folds
    # -----------------------------------------------------------------------------
    def do_split(self):
        if self.dataset_class is None:
            from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

        all_cases = sorted(self.dataset_class.get_identifiers(self.preprocessed_dataset_folder))

        if self.training_set is None:
            selected = all_cases
            split_file_name = 'splits_final.json'
        else:
            selected = subsets.filter_cases(all_cases, self.training_set)
            if not selected:
                raise RuntimeError(
                    f'-{self.training_set} selected none of the {len(all_cases)} cases in '
                    f'{self.preprocessed_dataset_folder}. Are you pointing -d at the right dataset?')
            missing = len(all_cases) - len(selected)
            if missing:
                self.print_to_log_file(
                    f'[unified] {self.training_set} uses {len(selected)} of the {len(all_cases)} cases '
                    f'in this dataset ({missing} filtered out).')
            # When the dataset already *is* the training set (the normal case, one
            # dataset per M) we can use the standard file name.
            split_file_name = ('splits_final.json' if len(selected) == len(all_cases)
                               else f'splits_final_{self.training_set}.json')

        if self.fold == 'all':
            tr_keys = val_keys = selected
        else:
            splits_file = join(self.preprocessed_dataset_folder_base, split_file_name)
            if not isfile(splits_file):
                self.print_to_log_file(
                    f'[unified] creating a cohort-stratified 5-fold split over {len(selected)} cases '
                    f'-> {splits_file}')
                splits = subsets.generate_stratified_crossval_split(selected, n_splits=5, seed=12345)
                save_json(splits, splits_file)
            else:
                self.print_to_log_file(f'[unified] using existing split file {splits_file}')
                splits = load_json(splits_file)

            if self.fold >= len(splits):
                raise RuntimeError(f'fold {self.fold} requested but the split file only has {len(splits)}')
            selected_set = set(selected)
            tr_keys = [k for k in splits[self.fold]['train'] if k in selected_set]
            val_keys = [k for k in splits[self.fold]['val'] if k in selected_set]
            self.print_to_log_file(
                f'[unified] fold {self.fold}: {len(tr_keys)} training, {len(val_keys)} validation cases')

        if self._validation_head is not None:
            val_keys = [k for k in val_keys if self._head_for(k) == self._validation_head]
            self.print_to_log_file(
                f'[unified] restricting this validation pass to head{self._validation_head + 1}: '
                f'{len(val_keys)} cases')
        return tr_keys, val_keys

    # -----------------------------------------------------------------------------
    # head routing
    # -----------------------------------------------------------------------------
    def _head_for(self, case: str) -> int:
        if not self.dual_head:
            return 0
        if self.head2_cases is not None:
            return 1 if subsets.is_head2(case, self.head2_cases) else 0
        # No explicit list: fall back to the M4 protocol (cohort_a -> head1, 400+ -> head2)
        return subsets.head_of(case, self.training_set or 'M4', None)

    def _set_head_index_from_batch(self, batch: dict):
        wrapper = self._dual_head_module()
        if wrapper is None:
            return
        keys = batch.get('keys')
        if keys is None:
            wrapper.set_head_index(None)
            return
        wrapper.set_head_index(torch.as_tensor([self._head_for(k) for k in keys], dtype=torch.long))

    def train_step(self, batch: dict) -> dict:
        self._set_head_index_from_batch(batch)
        return super().train_step(batch)

    def validation_step(self, batch: dict) -> dict:
        self._set_head_index_from_batch(batch)
        return super().validation_step(batch)

    # -----------------------------------------------------------------------------
    # final validation: one pass per head so each case goes through its own
    # -----------------------------------------------------------------------------
    def perform_actual_validation(self, save_probabilities: bool = False):
        wrapper = self._dual_head_module()
        if wrapper is None:
            return super().perform_actual_validation(save_probabilities)

        # sliding-window inference has no per-sample routing, so fix the head and
        # run the stock implementation once per head over its own cases
        for head in (0, 1):
            self._validation_head = head
            wrapper.set_head_index(None)
            wrapper.set_active_head(head)
            _, val_keys = self.do_split()
            if not val_keys:
                self.print_to_log_file(f'[unified] no validation case uses head{head + 1}, skipping')
                continue
            self.print_to_log_file(f'[unified] final validation pass for head{head + 1}')
            super().perform_actual_validation(save_probabilities)
        self._validation_head = None

    # -----------------------------------------------------------------------------
    # checkpointing: remember how this model was trained so inference can rebuild it
    # -----------------------------------------------------------------------------
    def load_checkpoint(self, checkpoint) -> None:
        # Resuming with --c re-reads -M / -head2 from the command line. If they differ
        # from what the checkpoint was trained with, the run would silently continue
        # with a different training set or head routing, so refuse instead.
        meta = None
        if isinstance(checkpoint, str) and isfile(checkpoint):
            meta = torch.load(checkpoint, map_location='cpu', weights_only=False).get('unified')
        elif isinstance(checkpoint, dict):
            meta = checkpoint.get('unified')

        if meta is not None:
            # Re-running validation must use the precision the weights were trained at:
            # a network whose activations overflow fp16 does so in the sliding window too.
            ckpt_amp = meta.get('amp_dtype')
            if ckpt_amp and ckpt_amp != self.amp_dtype:
                self.print_to_log_file(
                    f'[unified] checkpoint was trained in {ckpt_amp}, switching from '
                    f'{self.amp_dtype} to match it')
                self.amp_dtype = ckpt_amp
                self._apply_amp_dtype()

            mismatches = []
            if meta.get('training_set') != self.training_set:
                mismatches.append(f"training set: checkpoint={meta.get('training_set')}, "
                                  f"now={self.training_set}")
            if bool(meta.get('dual_head')) != bool(self.dual_head):
                mismatches.append(f"dual_head: checkpoint={meta.get('dual_head')}, now={self.dual_head}")
            if mismatches:
                raise RuntimeError(
                    'this checkpoint was trained with different unified settings than the ones '
                    'given on the command line (' + '; '.join(mismatches) + '). Re-run with the '
                    'same -M/-head2 flags, or train from scratch into a different folder.')
        super().load_checkpoint(checkpoint)

    def save_checkpoint(self, filename: str) -> None:
        # Same as nnUNetTrainer.save_checkpoint plus a 'unified' block. Written in one
        # go rather than by re-opening the file the parent just wrote, because these
        # checkpoints reach ~2 GB for the larger transformers.
        if self.local_rank != 0:
            return
        if self.disable_checkpointing:
            self.print_to_log_file('No checkpoint written, checkpointing is disabled')
            return

        mod = self.network.module if self.is_ddp else self.network
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod

        torch.save({
            'network_weights': mod.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'grad_scaler_state': self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
            'logging': self.logger.get_checkpoint(),
            '_best_ema': self._best_ema,
            'current_epoch': self.current_epoch + 1,
            'init_args': self.my_init_kwargs,
            'trainer_name': self.__class__.__name__,
            'inference_allowed_mirroring_axes': self.inference_allowed_mirroring_axes,
            'unified': {
                'architecture': self.spec.cli_name,
                'amp_dtype': self.amp_dtype,
                'training_set': self.training_set,
                'dual_head': self.dual_head,
                'head2_file': self.head2_file,
                'deep_supervision': self.spec.deep_supervision,
            },
        }, filename)


# =================================================================================
# The eleven concrete trainers. `-tr <cli name>` maps onto these class names via
# nnunetv2.unified.config.resolve_trainer_name, and the class name is what ends up
# in nnUNet_results/<dataset>/<class>__<plans>__<configuration>.
# =================================================================================
class nnUNetTrainer_swinunetrv1(UnifiedTrainer):
    pass


class nnUNetTrainer_swinunetrv2(UnifiedTrainer):
    pass


class nnUNetTrainer_vnet(UnifiedTrainer):
    pass


class nnUNetTrainer_3dunet(UnifiedTrainer):
    pass


class nnUNetTrainer_transbts(UnifiedTrainer):
    pass


class nnUNetTrainer_stunet(UnifiedTrainer):
    pass


class nnUNetTrainer_mednext3d(UnifiedTrainer):
    pass


class nnUNetTrainer_nnformer(UnifiedTrainer):
    pass


class nnUNetTrainer_unetr(UnifiedTrainer):
    pass


class nnUNetTrainer_cotr(UnifiedTrainer):
    pass


class nnUNetTrainer_nnunet(UnifiedTrainer):
    """Stock nnU-Net, routed through the same interface so -M / -head2 also work."""
    pass
