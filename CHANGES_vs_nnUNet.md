# Changes relative to upstream nnU-Net

Baseline: [MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet) `master`, version `2.8.1` as declared in `pyproject.toml`.

There are only two kinds of change: **seven upstream files were patched** (+208 / −10 lines in total) and **three paths were added**. Every patch inside an upstream file carries the comment `# >>> unified benchmark`, so they can be listed with:

```bash
grep -rn '>>> unified benchmark' nnUNet/nnunetv2
```

Everything else is byte-identical to upstream and can be checked with `diff -rq` against an official checkout.

## 1. Patched upstream files

| File | +/− | What changed |
|---|---|---|
| `nnunetv2/run/run_training.py` | +45 / −3 | Adds the mutually exclusive `-M0..-M4` group and `-head2 <txt>`; lets `-tr` take an architecture short name; allows `--val` to run when only `checkpoint_best.pth` exists and writes its output to `validation_best/` |
| `nnunetv2/inference/predict_from_raw_data.py` | +119 / −0 | Adds the mutually exclusive `-A/-B/-C`, `-topk` and `-head2`; reads `dual_head` and `amp_dtype` back from the checkpoint's `unified` metadata; routes each case to an output head |
| `nnunetv2/inference/export_prediction.py` | +14 / −1 | Forces the probability path when a constraint is armed (without probabilities there is nothing to rank clusters by) and applies `cluster_constraint.maybe_apply` before export |
| `nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py` | +4 / −1 | Makes the validation output folder name overridable (`validation_folder_name`, default `validation`), so re-running validation with the best checkpoint does not overwrite the existing `validation/` |
| `nnunetv2/utilities/find_objects.py` | +6 / −0 | Resolves a short name to a trainer class (`-tr cotr` → `nnUNetTrainer_cotr`) |
| `nnunetv2/evaluation/accumulate_cv_results.py` | +5 / −3 | Adds a `val_folder` parameter instead of hard-coding `fold_X/validation` |
| `nnunetv2/evaluation/find_best_configuration.py` | +15 / −2 | The same, exposed on the command line as `-val_folder` |

## 2. Added paths

| Path | Lines | Contents |
|---|---|---|
| `nnunetv2/unified/` | 1132 + 3357 vendored | The shared layer, see below |
| `nnunetv2/training/nnUNetTrainer/unified_trainers.py` | 440 | The `UnifiedTrainer` base class and eleven concrete trainers |
| `nnunetv2/experiment_planning/experiment_planners/unified_planner.py` | 87 | `nnUNetPlannerUnified`, which pins the patch and batch size to one value for every method |

Inside `nnunetv2/unified/`:

| File | Lines | Contents |
|---|---|---|
| `config.py` | 212 | Architecture registry: optimiser, schedule, learning rate, deep supervision, and the source and reference for each |
| `subsets.py` | 209 | M0-M4 definitions, cohort membership, test ranges, stratified 5-fold split |
| `builders.py` | 264 | Construction of the eleven networks and their deep supervision scales |
| `cluster_constraint.py` | 173 | The top-k 26-connected-component constraint behind `-A/-B/-C` |
| `schedulers.py` | 128 | Linear warmup + cosine (matching the official closed form), step, constant |
| `dual_head.py` | 70 | The second output head |
| `runtime.py` | 68 | Passing CLI flags through to the trainer (spawn- and DDP-safe) |
| `nets/` | 3357 | The official source of TransBTS, STU-Net, MedNeXt, nnFormer and CoTr. Only the imports were rewritten; every functional change is marked `# [unified]` |

## 3. Why patches and not a pure plugin

nnU-Net's plugin mechanism (`-tr` finding a custom trainer class) is enough to add a new architecture, but it cannot add **command line arguments**. Flags such as `-M0..-M4`, `-head2` and `-A/-B/-C` have to be declared in the argparse setup of `run_training.py` and `predict_from_raw_data.py` before `nnUNetv2_train` and `nnUNetv2_predict` will accept them. The edits were deliberately kept small: those two entry points absorb 167 of the 218 changed lines, and the other five files changed by 5–17 lines each.
