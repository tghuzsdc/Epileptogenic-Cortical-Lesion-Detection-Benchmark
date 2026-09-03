"""Build one of the eleven benchmark networks from the nnU-Net plans.

Every builder returns ``(network, deep_supervision_scales)``:

* ``network`` takes ``[B, in_ch, *patch_size]`` and returns either a single
  logits tensor (deep supervision off) or a list of logits tensors ordered from
  the highest to the lowest resolution (deep supervision on) -- the convention
  nnU-Net v2 expects.
* ``deep_supervision_scales`` is the list of per-axis scale factors the network
  actually produces, e.g. ``[[1,1,1], [.5,.5,.5], ...]``.  nnU-Net normally
  derives these from the plans, which is only correct for its own U-Net; CoTr for
  instance emits anisotropic scales.  Returning the true scales here keeps the
  downsampled targets aligned with the outputs.  ``None`` means "no deep supervision".
"""
from typing import List, Optional, Tuple

import numpy as np
from torch import nn

from nnunetv2.unified.config import get_spec


def _require_monai():
    try:
        import monai  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            'MONAI is required for swinunetrv1 / swinunetrv2 / vnet / 3dunet / unetr. '
            'Install it with:  pip install "monai>=1.3"') from e
    return monai


def _uniform_scales(num_outputs: int, dim: int = 3) -> List[List[float]]:
    return [[1 / (2 ** i)] * dim for i in range(num_outputs)]


# ---------------------------------------------------------------------------------
# MONAI-backed architectures
# ---------------------------------------------------------------------------------
def _build_swinunetr(in_ch, out_ch, patch_size, use_v2):
    _require_monai()
    from monai.networks.nets import SwinUNETR
    kwargs = dict(in_channels=in_ch, out_channels=out_ch, feature_size=48,
                  depths=(2, 2, 2, 2), num_heads=(3, 6, 12, 24),
                  drop_rate=0.0, attn_drop_rate=0.0, dropout_path_rate=0.0,
                  use_checkpoint=False, spatial_dims=len(patch_size), use_v2=use_v2)
    try:
        # MONAI >= 1.5 dropped the (now inferred) img_size argument
        return SwinUNETR(**kwargs)
    except TypeError:
        return SwinUNETR(img_size=tuple(patch_size), **kwargs)


def _build_vnet(in_ch, out_ch, patch_size):
    _require_monai()
    from monai.networks.nets import VNet
    common = dict(spatial_dims=len(patch_size), in_channels=in_ch, out_channels=out_ch,
                  act=('elu', {'inplace': True}), dropout_dim=3, bias=False)
    # V-Net's own dropout (p=0.5 in the deeper stages) is kept as published. MONAI >= 1.2
    # split dropout_prob into a down and an up value, and the up one must be a 2-tuple;
    # the deprecated single dropout_prob argument is actually broken in MONAI 1.4.
    try:
        return VNet(dropout_prob_down=0.5, dropout_prob_up=(0.5, 0.5), **common)
    except TypeError:
        return VNet(dropout_prob=0.5, **common)


def _build_3dunet(in_ch, out_ch, patch_size):
    _require_monai()
    from monai.networks.nets import BasicUNet
    # Cicek et al.: two 3x3x3 convolutions + BatchNorm + ReLU per level, feature
    # count doubling from 32.  BasicUNet's TwoConv block is exactly that; we keep
    # nnU-Net's usual 5-level depth so the receptive field matches the other models
    # at a 128^3 patch.  The trailing 32 is BasicUNet's final upsampling width.
    return BasicUNet(spatial_dims=len(patch_size), in_channels=in_ch, out_channels=out_ch,
                     features=(32, 64, 128, 256, 512, 32),
                     act=('ReLU', {'inplace': True}), norm=('batch', {}), dropout=0.0)


def _build_unetr(in_ch, out_ch, patch_size):
    _require_monai()
    from monai.networks.nets import UNETR
    assert all(p % 16 == 0 for p in patch_size), \
        f'UNETR needs every patch axis divisible by 16, got {patch_size}'
    return UNETR(in_channels=in_ch, out_channels=out_ch, img_size=tuple(patch_size),
                 feature_size=16, hidden_size=768, mlp_dim=3072, num_heads=12,
                 norm_name='instance', res_block=True, dropout_rate=0.0,
                 spatial_dims=len(patch_size))


# ---------------------------------------------------------------------------------
# Vendored official architectures
# ---------------------------------------------------------------------------------
def _build_transbts(in_ch, out_ch, patch_size):
    from nnunetv2.unified.nets.transbts import BTS
    assert len(set(patch_size)) == 1, \
        f'TransBTS is defined for a cubic input; patch size {patch_size} is not cubic'
    assert patch_size[0] % 16 == 0, f'TransBTS needs the patch size divisible by 16, got {patch_size}'
    return BTS(img_dim=int(patch_size[0]), patch_dim=8, num_channels=in_ch, num_classes=out_ch,
               embedding_dim=512, num_heads=8, num_layers=4, hidden_dim=4096,
               dropout_rate=0.1, attn_dropout_rate=0.1,
               conv_patch_representation=True, positional_encoding_type='learned')


def _build_stunet(in_ch, out_ch, strides, kernel_sizes):
    from nnunetv2.unified.nets.stunet import STUNet
    # STU-Net-B: the base preset of the paper (depth 1 per stage, 32..512 features).
    num_pool = len(strides) - 1
    dims = [min(32 * 2 ** i, 512) for i in range(num_pool + 1)]
    return STUNet(input_channels=in_ch, num_classes=out_ch,
                  depth=[1] * (num_pool + 1), dims=dims,
                  pool_op_kernel_sizes=strides[1:], conv_kernel_sizes=kernel_sizes)


def _build_mednext3d(in_ch, out_ch, deep_supervision):
    from nnunetv2.unified.nets.mednext import create_mednext_v1
    # MedNeXt-B with kernel size 3, the configuration the paper reports for 3D.
    return create_mednext_v1(num_input_channels=in_ch, num_classes=out_ch,
                             model_id='B', kernel_size=3, deep_supervision=deep_supervision)


def _build_nnformer(in_ch, out_ch, patch_size, deep_supervision):
    from nnunetv2.unified.nets.nnformer import nnFormer
    embed_patch = [4, 4, 4]
    for axis, (p, e) in enumerate(zip(patch_size, embed_patch)):
        assert p % (e * 8) == 0, (
            f'nnFormer needs axis {axis} of the patch size divisible by {e * 8}, got {patch_size}')
    return nnFormer(crop_size=list(patch_size), embedding_dim=192, input_channels=in_ch,
                    num_classes=out_ch, conv_op=nn.Conv3d, depths=[2, 2, 2, 2],
                    num_heads=[6, 12, 24, 48], patch_size=embed_patch,
                    window_size=[4, 4, 8, 4], deep_supervision=deep_supervision)


def _build_cotr(in_ch, out_ch, patch_size, deep_supervision):
    from nnunetv2.unified.nets.cotr import ResTranUnet
    return ResTranUnet(norm_cfg='IN', activation_cfg='LeakyReLU', img_size=list(patch_size),
                       num_classes=out_ch, weight_std=False,
                       deep_supervision=deep_supervision, in_channels=in_ch)


def _build_nnunet(plans_manager, configuration_manager, in_ch, out_ch, deep_supervision):
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
    return get_network_from_plans(
        configuration_manager.network_arch_class_name,
        configuration_manager.network_arch_init_kwargs,
        configuration_manager.network_arch_init_kwargs_req_import,
        in_ch, out_ch, allow_init=True, deep_supervision=deep_supervision)


# ---------------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------------
def deep_supervision_scales(trainer_class_name: str,
                            configuration_manager,
                            enable_deep_supervision: bool) -> Optional[List[List[float]]]:
    """The scales a given architecture's deep supervision outputs actually live at.

    nnU-Net derives these from the plans, which only describes its own U-Net.
    MedNeXt always emits five isotropic levels, nnFormer three, and CoTr's CNN
    backbone starts with a (1,2,2) stride so its scales are anisotropic. Getting
    this wrong means the downsampled targets do not match the outputs.
    """
    spec = get_spec(trainer_class_name)
    if not enable_deep_supervision or not spec.deep_supervision:
        return None

    builder = spec.builder
    if builder in ('stunet', 'nnunet'):
        strides = np.vstack([list(i) for i in configuration_manager.pool_op_kernel_sizes])
        return list(list(i) for i in 1 / np.cumprod(strides, axis=0))[:-1]
    if builder == 'mednext3d':
        return _uniform_scales(5)
    if builder == 'nnformer':
        return _uniform_scales(3)
    if builder == 'cotr':
        return [[1, 1, 1], [1, .5, .5], [.5, .25, .25], [.25, .125, .125]]
    raise ValueError(f'no deep supervision scales defined for builder {builder!r}')


def build_network(trainer_class_name: str,
                  plans_manager,
                  configuration_manager,
                  num_input_channels: int,
                  num_output_channels: int,
                  enable_deep_supervision: bool) -> Tuple[nn.Module, Optional[List[List[float]]]]:
    spec = get_spec(trainer_class_name)
    builder = spec.builder
    patch_size = tuple(int(i) for i in configuration_manager.patch_size)
    ds = bool(enable_deep_supervision) and spec.deep_supervision

    # MedNeXt only creates out_1..out_4 when it is constructed with deep supervision,
    # and nnFormer's number of final_patch_expanding layers depends on it too. Inference
    # builds the network with enable_deep_supervision=False and then loads the training
    # weights, so the *structure* must always follow the architecture's own definition;
    # only the runtime flag is toggled. set_deep_supervision below does that.
    structural_ds = spec.deep_supervision

    def _scales(enabled):
        return deep_supervision_scales(trainer_class_name, configuration_manager, enabled)

    def _finish(net):
        if spec.deep_supervision:
            set_deep_supervision(net, ds)
        return net, _scales(ds)

    if builder == 'swinunetrv1':
        return _build_swinunetr(num_input_channels, num_output_channels, patch_size, use_v2=False), None
    if builder == 'swinunetrv2':
        return _build_swinunetr(num_input_channels, num_output_channels, patch_size, use_v2=True), None
    if builder == 'vnet':
        return _build_vnet(num_input_channels, num_output_channels, patch_size), None
    if builder == '3dunet':
        return _build_3dunet(num_input_channels, num_output_channels, patch_size), None
    if builder == 'unetr':
        return _build_unetr(num_input_channels, num_output_channels, patch_size), None
    if builder == 'transbts':
        return _build_transbts(num_input_channels, num_output_channels, patch_size), None

    if builder == 'stunet':
        strides = [list(i) for i in configuration_manager.pool_op_kernel_sizes]
        # ConfigurationManager exposes the strides but not the kernel sizes, so read
        # those straight out of the architecture kwargs in the plans
        kernels = [list(i) for i in configuration_manager.network_arch_init_kwargs['kernel_sizes']]
        # one output per decoder stage, highest resolution first
        return _finish(_build_stunet(num_input_channels, num_output_channels, strides, kernels))

    if builder == 'mednext3d':
        return _finish(_build_mednext3d(num_input_channels, num_output_channels, structural_ds))

    if builder == 'nnformer':
        return _finish(_build_nnformer(num_input_channels, num_output_channels, patch_size, structural_ds))

    if builder == 'cotr':
        # [result, ds0, ds1, ds2] -- the CNN backbone downsamples (1,2,2) before
        # the three isotropic stages, so the scales are anisotropic.
        return _finish(_build_cotr(num_input_channels, num_output_channels, patch_size, structural_ds))

    if builder == 'nnunet':
        return _finish(_build_nnunet(plans_manager, configuration_manager,
                                     num_input_channels, num_output_channels, structural_ds))

    raise ValueError(f'unknown builder {builder!r}')


def set_deep_supervision(network: nn.Module, enabled: bool) -> bool:
    """Toggle deep supervision on whatever kind of network we built.

    Each vendored implementation uses its own attribute name; nnU-Net's own
    U-Net keeps the flag on its decoder. Architectures that never had deep
    supervision are left alone.
    """
    from nnunetv2.unified.dual_head import DualHeadWrapper
    if isinstance(network, DualHeadWrapper):
        network = network.network

    touched = False
    for attr in ('do_ds', '_deep_supervision'):
        if hasattr(network, attr):
            setattr(network, attr, enabled)
            touched = True
    if hasattr(network, 'decoder') and hasattr(network.decoder, 'deep_supervision'):
        network.decoder.deep_supervision = enabled
        touched = True
    return touched
