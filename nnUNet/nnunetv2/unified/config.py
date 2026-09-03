"""Central configuration for the unified 11-architecture benchmark.

Everything that distinguishes one architecture run from another lives here:
the CLI name, the class name of its trainer, and the optimisation recipe taken
from the architecture's paper / official implementation.

Changing a number in this file is enough to change the behaviour of the
corresponding trainer -- the trainers themselves contain no hyper-parameters.
"""
import os
from typing import Dict, NamedTuple, Optional, Tuple


def _env_ints(name: str, default):
    """Read a comma-separated integer tuple from the environment."""
    raw = os.environ.get(name, '').strip()
    if not raw:
        return default
    return tuple(int(v) for v in raw.replace('x', ',').split(','))


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, '').strip()
    return int(raw) if raw else default


# ----------------------------------------------------------------------------------
# Unified patch / batch size. Every one of the 11 methods trains on exactly these,
# so that differences in the results come from the architecture and not from the
# amount of context or the effective batch statistics.
#
# 128^3 is the largest size that satisfies every architecture's divisibility
# constraint at once (UNETR needs /16, SwinUNETR needs /32, nnFormer needs /32,
# TransBTS is built for 128, CoTr needs /16) and is what the MedNeXt, STU-Net and
# CoTr papers use themselves.
#
# All four can be overridden from the environment, which is how the integration
# test runs a miniature version of the pipeline:
#   nnUNet_unified_patch3d=64,64,64  nnUNet_unified_batch3d=2
#   nnUNet_unified_patch2d=128,128   nnUNet_unified_batch2d=4
#   nnUNet_unified_epochs=2
# These must be set *before planning* (they end up in the plans) and are only meant
# for testing -- the benchmark runs on the defaults below.
# ----------------------------------------------------------------------------------
PATCH_SIZE_3D: Tuple[int, ...] = _env_ints('nnUNet_unified_patch3d', (128, 128, 128))
BATCH_SIZE_3D: int = _env_int('nnUNet_unified_batch3d', 2)
PATCH_SIZE_2D: Tuple[int, ...] = _env_ints('nnUNet_unified_patch2d', (512, 512))
BATCH_SIZE_2D: int = _env_int('nnUNet_unified_batch2d', 12)

NUM_EPOCHS: int = _env_int('nnUNet_unified_epochs', 1000)
NUM_ITERATIONS_PER_EPOCH: int = _env_int('nnUNet_unified_iters', 250)

#: Mixed-precision dtype. nnU-Net's autocast uses fp16, whose 65504 ceiling three of
#: these architectures overrun: mednext3d, stunet and unetr drove their logits to
#: exactly 65504 -> Inf -> loss NaN, and because the gradient scaler then skips every
#: step, training kept running without ever updating a weight again. bf16 has fp32's
#: exponent range at the same speed on A100 and needs no gradient scaling, so it is
#: the default here. With bf16 all twelve runs train cleanly and the three that used
#: to die are among the best performers.
#:   'bf16' (default) | 'fp16' (nnU-Net's own default) | 'fp32'
#: Override per run with nnUNet_unified_amp=fp16, or per architecture in AMP_OVERRIDE.
AMP_DTYPE: str = os.environ.get('nnUNet_unified_amp', 'bf16').strip().lower() or 'bf16'

#: Architectures that need something other than AMP_DTYPE. Empty by default: changing
#: the numerics of only some entries makes the benchmark less comparable, so this is
#: a deliberate, documented choice rather than a silent per-model workaround.
AMP_OVERRIDE = {}


class ArchSpec(NamedTuple):
    """One benchmark entry: how to build it and how to optimise it."""
    cli_name: str            # what the user types after -tr
    trainer_class: str       # the actual python class name (and results folder name)
    builder: str             # key understood by nnunetv2.unified.builders.build_network
    initial_lr: float
    optimizer: str           # 'SGD' | 'Adam' | 'AdamW'
    weight_decay: float
    momentum: Optional[float]        # SGD only
    nesterov: bool                   # SGD only
    scheduler: str           # 'poly' | 'warmup_cosine' | 'step' | 'constant'
    deep_supervision: bool
    source: str              # provenance of the network code
    reference: str           # where the optimisation recipe comes from
    #: extra arguments the official implementation passes to the optimizer
    optimizer_kwargs: Tuple[Tuple[str, object], ...] = ()


#: ``poly`` is nnU-Net's ``PolyLRScheduler`` with exponent 0.9, i.e.
#: ``lr = lr0 * (1 - epoch / num_epochs) ** 0.9`` -- identical to the "poly"
#: policy described by CoTr, STU-Net, nnFormer, MedNeXt and TransBTS.
#: ``warmup_cosine`` is the SwinUNETR recipe: linear warmup then cosine annealing.
#: ``step`` is V-Net's "decrease by one order of magnitude every 25K iterations".

ARCHITECTURES: Tuple[ArchSpec, ...] = (
    ArchSpec(
        cli_name='swinunetrv1', trainer_class='nnUNetTrainer_swinunetrv1', builder='swinunetrv1',
        initial_lr=8e-4, optimizer='AdamW', weight_decay=1e-5, momentum=None, nesterov=False,
        scheduler='warmup_cosine', deep_supervision=False,
        source='MONAI monai.networks.nets.SwinUNETR (the official Swin UNETR release lives in '
               'Project-MONAI/research-contributions and imports this very class)',
        reference='Hatamizadeh et al., Swin UNETR (BraTS 2021): AdamW, linear warmup + cosine '
                  'annealing; lr as specified by the user (8e-4); no deep supervision.'),
    ArchSpec(
        cli_name='swinunetrv2', trainer_class='nnUNetTrainer_swinunetrv2', builder='swinunetrv2',
        initial_lr=4e-4, optimizer='AdamW', weight_decay=1e-5, momentum=None, nesterov=False,
        scheduler='warmup_cosine', deep_supervision=False,
        source='MONAI monai.networks.nets.SwinUNETR(use_v2=True) -- use_v2 is the official '
               'SwinUNETR-V2 stagewise-convolution variant contributed by the paper authors',
        reference='He et al., SwinUNETR-V2 (MICCAI 2023) sec. Implementation Details: "our training '
                  'recipe is the same as that by SwinUNETR. We changed the initial learning rate to 4e-4".'),
    ArchSpec(
        cli_name='vnet', trainer_class='nnUNetTrainer_vnet', builder='vnet',
        initial_lr=1e-4, optimizer='SGD', weight_decay=0.0, momentum=0.99, nesterov=False,
        scheduler='step', deep_supervision=False,
        source='MONAI monai.networks.nets.VNet (the original release is Caffe-only)',
        reference='Milletari et al., V-Net: "We used a momentum of 0.99 and a initial learning rate '
                  'of 0.0001 which decreases by one order of magnitude every 25K iterations."'),
    ArchSpec(
        cli_name='3dunet', trainer_class='nnUNetTrainer_3dunet', builder='3dunet',
        initial_lr=1e-2, optimizer='SGD', weight_decay=3e-5, momentum=0.99, nesterov=True,
        scheduler='poly', deep_supervision=False,
        source='MONAI monai.networks.nets.BasicUNet configured to the Cicek et al. topology '
               '(two 3x3x3 convs + BatchNorm + ReLU per level, doubling 32->512 features)',
        reference='Cicek et al., 3D U-Net does not report a learning rate or optimiser setting; '
                  'per user decision we use the nnU-Net default (SGD, momentum 0.99, nesterov, poly).'),
    ArchSpec(
        cli_name='transbts', trainer_class='nnUNetTrainer_transbts', builder='transbts',
        initial_lr=4e-4, optimizer='Adam', weight_decay=1e-5, momentum=None, nesterov=False,
        scheduler='poly', deep_supervision=False,
        source='official Wenxuan-1119/TransBTS models/TransBTS/TransBTS_downsample8x_skipconnection.py',
        reference='Wang et al., TransBTS: "We adopt the Adam optimizer ... initial learning rate is set '
                  'to 0.0004 with a poly learning rate decay ... weight decay rate of 1e-5." The '
                  'official train.py defaults to --amsgrad True, and its adjust_learning_rate is '
                  'exactly poly with power 0.9.',
        optimizer_kwargs=(('amsgrad', True),)),
    ArchSpec(
        cli_name='stunet', trainer_class='nnUNetTrainer_stunet', builder='stunet',
        initial_lr=1e-2, optimizer='SGD', weight_decay=3e-5, momentum=0.99, nesterov=True,
        scheduler='poly', deep_supervision=True,
        source='official uni-medical/STU-Net nnunet/network_architecture/STUNet.py (STU-Net-B preset)',
        reference='Huang et al., STU-Net: "SGD optimizer with Nestrov momentum of 0.99 ... learning rate '
                  'starts at 0.01 ... decayed following the poly learning rate policy (1-epoch/1000)^0.9." '
                  'Weight decay is kept at the nnU-Net default 3e-5 used by the official STUNetTrainer.'),
    ArchSpec(
        cli_name='mednext3d', trainer_class='nnUNetTrainer_mednext3d', builder='mednext3d',
        initial_lr=1e-3, optimizer='AdamW', weight_decay=3e-5, momentum=None, nesterov=False,
        scheduler='poly', deep_supervision=True,
        source='official MIC-DKFZ/MedNeXt nnunet_mednext/network_architecture/mednextv1 (MedNeXt-B, k=3)',
        reference='Roy et al., MedNeXt: "uses the nnUNet as a backbone ... AdamW as optimizer ... '
                  'The learning rate for all MedNeXt models is 0.001." The official '
                  'nnUNetTrainerV2_Optim_and_LR passes eps=1e-4 with the comment "1e-8 might cause '
                  'nans in fp16" -- we train under autocast, so this matters.',
        optimizer_kwargs=(('eps', 1e-4),)),
    ArchSpec(
        cli_name='nnformer', trainer_class='nnUNetTrainer_nnformer', builder='nnformer',
        initial_lr=1e-2, optimizer='SGD', weight_decay=3e-5, momentum=0.99, nesterov=True,
        scheduler='poly', deep_supervision=True,
        source='official 282857341/nnFormer nnformer/network_architecture/nnFormer_tumor.py',
        reference='Zhou et al., nnFormer: "The initial learning rate is set to 0.01 and we employ a poly '
                  'decay strategy ... optimizer is SGD where we set the momentum to 0.99. The weight '
                  'decay is set to 3e-5." The paper does not mention Nesterov, but the official '
                  'nnFormerTrainerV2_nnformer_tumor uses SGD(momentum=0.99, nesterov=True).'),
    ArchSpec(
        cli_name='unetr', trainer_class='nnUNetTrainer_unetr', builder='unetr',
        initial_lr=1e-4, optimizer='AdamW', weight_decay=1e-5, momentum=None, nesterov=False,
        scheduler='warmup_cosine', deep_supervision=False,
        source='MONAI monai.networks.nets.UNETR (the official UNETR release is the MONAI '
               'research-contributions BTCV pipeline, which uses this class)',
        reference='Hatamizadeh et al., UNETR: "trained ... using the AdamW optimizer with initial '
                  'learning rate of 0.0001". The official BTCV main.py defaults to '
                  '--optim_name adamw --reg_weight 1e-5 --lrschedule warmup_cosine --warmup_epochs 50, '
                  'which is what we use.'),
    ArchSpec(
        cli_name='cotr', trainer_class='nnUNetTrainer_cotr', builder='cotr',
        initial_lr=1e-2, optimizer='SGD', weight_decay=3e-5, momentum=0.99, nesterov=True,
        scheduler='poly', deep_supervision=True,
        source='official YtongXie/CoTr CoTr_package/CoTr/network_architecture (ResTranUnet + '
               'pure-PyTorch 3D deformable attention -- no CUDA extension needed)',
        reference='Xie et al., CoTr: "stochastic gradient descent algorithm with a momentum of 0.99 and '
                  'an initial learning rate of 0.01"; nnU-Net poly schedule and weight decay 3e-5. '
                  'The official nnUNetTrainerV2_ResTrans uses SGD(momentum=0.99, nesterov=True).'),
    # The nnU-Net baseline itself, exposed through the same interface so that -M / -head2
    # work identically for it. Architecture and optimiser are stock nnU-Net.
    ArchSpec(
        cli_name='nnunet', trainer_class='nnUNetTrainer_nnunet', builder='nnunet',
        initial_lr=1e-2, optimizer='SGD', weight_decay=3e-5, momentum=0.99, nesterov=True,
        scheduler='poly', deep_supervision=True,
        source='nnU-Net v2 PlainConvUNet built from the plans (unmodified)',
        reference='Isensee et al., nnU-Net defaults.'),
)

ARCH_BY_CLI: Dict[str, ArchSpec] = {a.cli_name: a for a in ARCHITECTURES}
ARCH_BY_TRAINER: Dict[str, ArchSpec] = {a.trainer_class: a for a in ARCHITECTURES}


def resolve_trainer_name(name: str) -> str:
    """Map a short ``-tr`` name onto the real trainer class name.

    ``-tr swinunetrv1`` -> ``nnUNetTrainer_swinunetrv1``.  Names that are already
    class names (or belong to stock nnU-Net) are returned unchanged, so this is
    safe to call unconditionally on whatever the user typed.
    """
    if name in ARCH_BY_CLI:
        return ARCH_BY_CLI[name].trainer_class
    return name


def get_spec(trainer_class_name: str) -> ArchSpec:
    if trainer_class_name not in ARCH_BY_TRAINER:
        raise KeyError(f'{trainer_class_name} is not a unified benchmark trainer. '
                       f'Known: {sorted(ARCH_BY_TRAINER)}')
    return ARCH_BY_TRAINER[trainer_class_name]
