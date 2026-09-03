# Environment for the unified nnU-Net benchmark.
#
#   conda activate nnunet
#   source <PROJECT>/scripts/env.sh
#
# Everything here is on top of `conda activate nnunet`. Nothing needs to be
# installed by this script; see the README for the one-off pip step.
#
# PROJECT is the repository root -- the folder holding nnUNet/, scripts/ and data/.
# It is derived from this script's own location, so the checkout can live anywhere.
# Override it by exporting NNUNET_BENCH_PROJECT before sourcing, which is what you
# want if data/ sits on a different filesystem from the code.

if [ -n "${NNUNET_BENCH_PROJECT:-}" ]; then
    PROJECT="$NNUNET_BENCH_PROJECT"
elif [ -n "${BASH_SOURCE[0]:-}" ]; then
    PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
elif [ -n "${ZSH_VERSION:-}" ]; then
    PROJECT="$(cd "$(dirname "${(%):-%x}")/.." && pwd)"
else
    echo "cannot locate the project root: export NNUNET_BENCH_PROJECT=<repo> first" >&2
    return 1 2>/dev/null || exit 1
fi

if [ ! -d "$PROJECT/nnUNet/nnunetv2" ]; then
    echo "*** $PROJECT does not look like the benchmark repo (no nnUNet/nnunetv2)" >&2
    return 1 2>/dev/null || exit 1
fi

# Data root. Keep it inside the repo by default; point NNUNET_BENCH_DATA somewhere
# with room if the checkout lives on a small disk (the preprocessed arrays are large).
DATA="${NNUNET_BENCH_DATA:-$PROJECT/data}"

# Drop any other nnU-Net checkout from PYTHONPATH before prepending this one, so
# `import nnunetv2` cannot fall back to a different tree. This matters whenever a
# login script already exports PYTHONPATH for an earlier nnU-Net install.
_cleaned=""
_IFS_SAVE=$IFS
IFS=':'
for _p in ${PYTHONPATH:-}; do
    case "$_p" in
        ''|*/nnUNet|*/nnUNet/) ;;                      # skip: another nnU-Net source tree
        *) _cleaned="${_cleaned:+$_cleaned:}$_p" ;;
    esac
done
IFS=$_IFS_SAVE
export PYTHONPATH="$PROJECT/nnUNet${_cleaned:+:$_cleaned}"
unset _cleaned _p _IFS_SAVE

# PYTHONPATH wins over site-packages, so the console scripts installed by pip
# (nnUNetv2_train, nnUNetv2_predict, ...) execute this source tree.
export nnUNet_raw="$DATA/nnUNet_raw"
export nnUNet_preprocessed="$DATA/nnUNet_preprocessed"
export nnUNet_results="$DATA/nnUNet_results"

# One GPU on this node; keep BLAS from oversubscribing the CPU.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# torch.compile is off by default for the benchmark trainers (eleven very different
# architectures, several of which do not compile cleanly, plus a per-batch routing
# tensor in the dual head). Set to 1 to opt back in.
export nnUNet_compile=0

# Data augmentation worker count. Tune to the node.
export nnUNet_n_proc_DA="${nnUNet_n_proc_DA:-12}"

# Mixed precision. bf16 is the default and the one the results were produced with:
# MedNeXt, STU-Net and UNETR overflow fp16 (see the Training section of the README).
export nnUNet_unified_amp="${nnUNet_unified_amp:-bf16}"

echo "PROJECT             = $PROJECT"
echo "PYTHONPATH          = $PYTHONPATH"
echo "nnUNet_raw          = $nnUNet_raw"
echo "nnUNet_preprocessed = $nnUNet_preprocessed"
echo "nnUNet_results      = $nnUNet_results"
echo "amp dtype           = $nnUNet_unified_amp"
python - <<'PY'
import os, sys
try:
    import nnunetv2
    where = os.path.dirname(nnunetv2.__file__)
    ok = where.startswith(os.environ['PYTHONPATH'].split(':')[0])
    print(f"nnunetv2 loaded from = {where}   {'OK' if ok else '*** NOT THIS PROJECT ***'}")
    if not ok:
        sys.exit(1)
    from nnunetv2.unified.config import ARCHITECTURES
    print(f"unified layer       = {len(ARCHITECTURES)} architectures registered "
          f"({', '.join(a.cli_name for a in ARCHITECTURES)})")
except Exception as e:
    print(f"*** environment is not usable: {type(e).__name__}: {e}")
    sys.exit(1)
PY
