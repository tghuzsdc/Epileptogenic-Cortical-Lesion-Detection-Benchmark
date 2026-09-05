# Epileptogenic Cortical Lesion Detection Benchmark

A single nnU-Net v2 command line for twelve shared-pipeline models: ten published architectures (Swin UNETR-V1, Swin UNETR-V2, V-Net, 3D U-Net, TransBTS, STU-Net, MedNeXt, nnFormer, UNETR, CoTr) plus nnU-Net itself in `2d` and `3d_fullres`. All twelve share one preprocessing run, fixed patch and batch sizes for each input dimensionality. Each method keeps the optimizer, schedule and learning rate of its own paper or official implementation.

Adds five training compositions (`-M0`..`-M4`), an optional second output head (`-head2`), and three top-3 export cap variants at inference (`-A`/`-B`/`-C`).

## Installation

```bash
conda activate nnunet
pip install --no-deps --no-cache-dir 'monai==1.4.0'
source /path/to/nnunet-unified-benchmark/scripts/env.sh
```

`--no-deps` is required, otherwise the resolver upgrades torch to an incompatible version. Minimum versions: `monai>=1.4.0`, `batchgeneratorsv2>=0.3.5`,
`acvl-utils>=0.2.6`, `dynamic-network-architectures>=0.4.4`, `batchgenerators>=0.25.1`.

Source `env.sh` in every shell. It derives the repository root from its own location, prepends `nnUNet/` to `PYTHONPATH`, exports `nnUNet_raw`, `nnUNet_preprocessed` and `nnUNet_results`, and verifies that `import nnunetv2` resolves to this tree. Overridable:

| Variable | Default | Meaning |
|---|---|---|
| `NNUNET_BENCH_PROJECT` | script location | repository root; export before sourcing |
| `NNUNET_BENCH_DATA` | `<REPO>/data` | data root |
| `nnUNet_n_proc_DA` | `12` | data-augmentation workers |
| `nnUNet_compile` | `0` | `torch.compile`; off because several networks do not compile cleanly |
| `nnUNet_unified_amp` | `bf16` | mixed precision (bfloat16/float16) |
| `nnUNet_unified_patch3d` / `nnUNet_unified_batch3d` | `128,128,128` / `2` | 3D patch and batch size |
| `nnUNet_unified_patch2d` / `nnUNet_unified_batch2d` | `512,512` / `12` | 2D patch and batch size |
| `nnUNet_unified_epochs` / `nnUNet_unified_iters` | `1000` / `250` | epochs and iterations per epoch |

## Data

```
${NNUNET_BENCH_DATA}
    [nnUNet_raw]
        [Dataset001_M0] [Dataset002_M1] [Dataset003_M2] [Dataset004_M3] [Dataset005_M4]
            - dataset.json
            [imagesTr]      # sub-testNNN_0000.nii.gz (T1-weighted), _0001.nii.gz (FLAIR)
            [labelsTr]      # sub-testNNN.nii.gz, uint8 0/1
            [imagesTs]
    [nnUNet_preprocessed]
    [nnUNet_results]
    [lists]
        - head2_train_M4.txt
        - head2_test_all.txt
    [labelsTs_all]          # external test reference masks, used for evaluation
    [predictions]
    [evaluation]
```

## Preprocessing

```bash
nnUNetv2_plan_and_preprocess -d [dataset ...] -pl nnUNetPlannerUnified --no_pp \
    --verify_dataset_integrity -np 8
nnUNetv2_preprocess -d [dataset ...] -plans_name nnUNetPlans -c 3d_fullres 2d -np 8 8

# Example: M1
nnUNetv2_plan_and_preprocess -d 2 -pl nnUNetPlannerUnified --no_pp \
    --verify_dataset_integrity -np 8
nnUNetv2_preprocess -d 2 -plans_name nnUNetPlans -c 3d_fullres 2d -np 8 8
```

## Training

```bash
nnUNetv2_train [dataset] [config] [fold] -tr [architecture] -M[k] [-head2 txt]

# Example: M1 fold 0
nnUNetv2_train 002 3d_fullres 0 -tr nnformer -M1

# Example: M4 fold 0 (two output heads)
nnUNetv2_train 005 3d_fullres 0 -tr nnformer -M4 \
    -head2 ${NNUNET_BENCH_DATA}/lists/head2_train_M4.txt
```

`-tr` accepts `swinunetrv1` `swinunetrv2` `vnet` `3dunet` `transbts` `stunet`
`mednext3d` `nnformer` `unetr` `cotr` `nnunet`.

All results were produced with bfloat16 (`bf16`), the default; MedNeXt, STU-Net and UNETR overflow in float16 (`fp16`).

## Validation

Re-run validation with `checkpoint_best.pth`. Results are written to
`[model]/fold_X/validation_best/`:

```bash
nnUNetv2_train [dataset] [config] [fold] -tr [architecture] -M[k] [-head2 txt] \
    --val --val_best

# Example: M1 fold 0
nnUNetv2_train 002 3d_fullres 0 -tr nnformer -M1 --val --val_best

# Example: M4 fold 0
nnUNetv2_train 005 3d_fullres 0 -tr nnformer -M4 \
    -head2 ${NNUNET_BENCH_DATA}/lists/head2_train_M4.txt --val --val_best
```

## Postprocessing

Run the two commands in order. They produce `postprocessing.pkl`:

```bash
nnUNetv2_accumulate_crossval_results [dataset] -c [config] -tr [architecture] \
    -f 0 1 2 3 4 -val_folder validation_best
nnUNetv2_determine_postprocessing -i [crossval_results_dir] -ref [labelsTr] -np 8

# Example: M1
nnUNetv2_accumulate_crossval_results 002 -c 3d_fullres -tr nnformer -f 0 1 2 3 4 \
    -val_folder validation_best
nnUNetv2_determine_postprocessing \
    -i ${nnUNet_results}/Dataset002_M1/nnUNetTrainer_nnformer__nnUNetPlans__3d_fullres/crossval_results_validation_best_folds_0_1_2_3_4 \
    -ref ${nnUNet_raw}/Dataset002_M1/labelsTr -np 8
```

## Inference

```bash
nnUNetv2_predict -i [imagesTs] -o [output_dir] -d [dataset] -c [config] \
    -f 0 1 2 3 4 -tr [architecture] -chk checkpoint_best.pth \
    [-A|-B|-C] [-topk n] [-head2 txt]
nnUNetv2_apply_postprocessing -i [output_dir] -o [output_dir]_pp \
    -pp_pkl_file [postprocessing.pkl] -np 8

# Example: M1
nnUNetv2_predict \
    -i ${nnUNet_raw}/Dataset002_M1/imagesTs \
    -o ${NNUNET_BENCH_DATA}/predictions/M1/nnformer_3d_fullres_A \
    -d 002 -c 3d_fullres -f 0 1 2 3 4 -tr nnformer -chk checkpoint_best.pth -A

# Example: M4 (every test case is routed through head2)
nnUNetv2_predict \
    -i ${nnUNet_raw}/Dataset005_M4/imagesTs \
    -o ${NNUNET_BENCH_DATA}/predictions/M4/nnformer_3d_fullres_A \
    -d 005 -c 3d_fullres -f 0 1 2 3 4 -tr nnformer -chk checkpoint_best.pth -A \
    -head2 ${NNUNET_BENCH_DATA}/lists/head2_test_all.txt
```

`-A`, `-B` and `-C` are mutually exclusive. All three keep only the three most confident 26-connected clusters (`-topk` changes the count); they differ in how cluster confidence is calculated: `-A` the 95th percentile of its voxel probabilities, `-B` the mean of its 50 highest voxel probabilities, `-C` the mean of its voxel probabilities above 0.3.

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

## Acknowledgements

Our code is based on the [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) framework.
Swin UNETR-V1, Swin UNETR-V2, UNETR, V-Net and 3D U-Net are built with
[MONAI](https://github.com/Project-MONAI/MONAI); the implementations of
[TransBTS](https://github.com/Wenxuan-1119/TransBTS),
[STU-Net](https://github.com/uni-medical/STU-Net),
[MedNeXt](https://github.com/MIC-DKFZ/MedNeXt),
[nnFormer](https://github.com/282857341/nnFormer) and
[CoTr](https://github.com/YtongXie/CoTr) are adapted from their official repositories.
Changes to upstream files are listed in [`CHANGES_vs_nnUNet.md`](CHANGES_vs_nnUNet.md).
