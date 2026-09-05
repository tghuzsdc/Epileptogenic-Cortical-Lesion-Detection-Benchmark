#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import multiprocessing as mp
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
from scipy import ndimage

# presentation
PERCENT_METRICS = {
    "Specificity",
    "Empty",
    "Precision-boxDSC",
    "Sensitivity-boxDSC",
    "AUPRC",
    "Detection rate-boxDSC",
    "Detection rate-PPV",
    "Detection rate-distance",
    "Detection rate-dice",
    "Pinpointing rate",
}

METRIC_ORDER = [
    "Specificity",
    "False Positives/Subject (Healthy)",
    "Empty",
    "dice_average",
    "Precision-boxDSC",
    "Sensitivity-boxDSC",
    "F1-score",
    "Number/Subject",
    "False Positives/Subject",
    "AUPRC",
    "Detection rate-boxDSC",
    "Detection rate-PPV",
    "Detection rate-distance",
    "Detection rate-dice",
    "Pinpointing rate",
]

EPS = 1e-12


def strict_gt(x: float, thr: float) -> bool:
    return x > (thr - EPS)


def le_tol(x: float, thr: float) -> bool:
    return x <= (thr + EPS)


def fmt_metric(name: str, est: float, lo: float, hi: float) -> str:
    if np.isnan(est) or np.isnan(lo) or np.isnan(hi):
        return f"{name}：NaN[NaN, NaN]"
    if name in PERCENT_METRICS:
        return f"{name}：{est * 100:.2f}%[{lo * 100:.2f}%, {hi * 100:.2f}%]"
    return f"{name}：{est:.4f}[{lo:.4f}, {hi:.4f}]"


# geometry
@dataclass(frozen=True)
class Box:
    x0: int
    x1: int
    y0: int
    y1: int
    z0: int
    z1: int

    def volume(self) -> int:
        return (max(0, self.x1 - self.x0) * max(0, self.y1 - self.y0)
                * max(0, self.z1 - self.z0))

    def intersection(self, other: "Box") -> int:
        dx = max(0, min(self.x1, other.x1) - max(self.x0, other.x0))
        dy = max(0, min(self.y1, other.y1) - max(self.y0, other.y0))
        dz = max(0, min(self.z1, other.z1) - max(self.z0, other.z0))
        return dx * dy * dz

    def dice(self, other: "Box") -> float:
        denom = self.volume() + other.volume()
        if denom <= 0:
            return 1.0 if self.volume() == 0 and other.volume() == 0 else 0.0
        return (2.0 * self.intersection(other)) / float(denom)

    def as_tuple(self):
        return (self.x0, self.x1, self.y0, self.y1, self.z0, self.z1)


def get_structure(connectivity: int) -> np.ndarray:
    if connectivity == 6:
        return ndimage.generate_binary_structure(3, 1)
    if connectivity == 18:
        return ndimage.generate_binary_structure(3, 2)
    return ndimage.generate_binary_structure(3, 3)


def dice_bool(a: np.ndarray, b: np.ndarray) -> float:
    a_sum, b_sum = int(a.sum()), int(b.sum())
    if a_sum == 0 and b_sum == 0:
        return 1.0
    denom = a_sum + b_sum
    if denom <= 0:
        return 0.0
    return (2.0 * int(np.logical_and(a, b).sum())) / float(denom)


# file discovery
def build_id_map(folders: Sequence[Path], glob_pattern: str, id_regex: str,
                 label: str) -> Dict[str, Path]:
    reg = re.compile(id_regex)
    out: Dict[str, Path] = {}
    for folder in folders:
        if not folder.is_dir():
            raise SystemExit(f'{label}: not a folder: {folder}')
        for p in sorted(folder.rglob(glob_pattern)):
            m = reg.search(p.name)
            if not m:
                continue
            sid = m.group(1)
            if sid in out and out[sid] != p:
                raise SystemExit(
                    f'{label}: subject {sid} matches two files:\n  {out[sid]}\n  {p}\n'
                    f'Narrow --{label}_glob or --id_regex.')
            out[sid] = p
    return out


# per-subject computation
def filter_small_clusters(mask: np.ndarray, structure: np.ndarray, min_vox: int) -> np.ndarray:
    if min_vox <= 1 or not mask.any():
        return mask
    cc, n = ndimage.label(mask, structure=structure)
    if n <= 0:
        return mask
    sizes = np.bincount(cc.ravel())
    keep = sizes >= int(min_vox)
    keep[0] = False
    return keep[cc]


def overlap_table(cc_pred: np.ndarray, cc_gt: np.ndarray, n_pred: int, n_gt: int):
    pairs: Dict[Tuple[int, int], int] = {}
    if n_pred <= 0 or n_gt <= 0:
        return pairs
    both = (cc_pred > 0) & (cc_gt > 0)
    if not both.any():
        return pairs
    stride = n_gt + 1
    codes = cc_pred[both].astype(np.int64) * stride + cc_gt[both].astype(np.int64)
    uniq, counts = np.unique(codes, return_counts=True)
    for code, cnt in zip(uniq.tolist(), counts.tolist()):
        pairs[(code // stride, code % stride)] = int(cnt)
    return pairs


def greedy_match_by_box_dice(pred_boxes: List[Box], gt_boxes: List[Box],
                             thr: float) -> Tuple[int, int, int]:
    candidates = []
    for i, pb in enumerate(pred_boxes):
        for j, gb in enumerate(gt_boxes):
            d = pb.dice(gb)
            if strict_gt(d, thr):
                candidates.append((d, i, j))
    candidates.sort(key=lambda x: (-x[0], x[1], x[2]))

    pred_used = [False] * len(pred_boxes)
    gt_used = [False] * len(gt_boxes)
    tp = 0
    for _, i, j in candidates:
        if not pred_used[i] and not gt_used[j]:
            pred_used[i] = gt_used[j] = True
            tp += 1
    return tp, len(pred_boxes) - tp, len(gt_boxes) - tp


def per_case_detection_flags(scores: List[float], pred_boxes: List[Box],
                             gt_boxes: List[Box], thr: float) -> List[bool]:
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], pred_boxes[i].as_tuple()))
    used = [False] * len(gt_boxes)
    is_tp = [False] * len(scores)
    for i in order:
        best_j, best_d = -1, -1.0
        for j, gb in enumerate(gt_boxes):
            if used[j]:
                continue
            d = pred_boxes[i].dice(gb)
            if d > best_d:
                best_d, best_j = d, j
        if best_j >= 0 and strict_gt(best_d, thr):
            used[best_j] = True
            is_tp[i] = True
    return is_tp


def average_precision(scores: np.ndarray, is_tp: np.ndarray, total_gt: int) -> float:
    if total_gt <= 0:
        return float('nan')
    if scores.size == 0:
        return 0.0
    order = np.argsort(-scores, kind='stable')
    tp_cum = np.cumsum(is_tp[order].astype(np.int64))
    fp_cum = np.cumsum(1 - is_tp[order].astype(np.int64))
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    recall = tp_cum / float(total_gt)

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for k in range(len(mpre) - 2, -1, -1):
        mpre[k] = max(mpre[k], mpre[k + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _compute_subject_record(sid: str, pred_path: str, label_path: str, threshold: float,
                            connectivity: int, min_pred_cluster_vox: int,
                            healthy_fp_tol_vox: int, boxdsc_thr: float, ppv_thr: float,
                            dist_thr_mm: float, dice_thr: float, score_mode: str
                            ) -> Tuple[str, str, Optional[Dict[str, Any]], Optional[str]]:
    try:
        structure = get_structure(connectivity)
        pred_img = nib.load(pred_path)
        lab_img = nib.load(label_path)
        pred_raw = np.asarray(pred_img.dataobj, dtype=np.float32)
        label = np.asarray(lab_img.dataobj, dtype=np.float32)

        if pred_raw.shape != label.shape:
            return sid, 'error', None, (f'[{sid}] shape mismatch: pred {pred_raw.shape} '
                                        f'vs label {label.shape}')

        zooms = pred_img.header.get_zooms()[:3]
        zooms_xyz = (float(zooms[0]), float(zooms[1]), float(zooms[2]))

        label_bin = label > 0
        pred_bin = filter_small_clusters(pred_raw > float(threshold), structure,
                                         int(min_pred_cluster_vox))

        pred_vox = int(pred_bin.sum())
        cc_pred, n_pred = ndimage.label(pred_bin, structure=structure)
        n_pred = int(n_pred)

        if not label_bin.any():
            return sid, 'healthy', {
                'id': sid,
                'spec_success': bool(pred_vox <= int(healthy_fp_tol_vox)),
                'pred_vox': pred_vox,
                'n_pred': n_pred,
            }, None

        cc_gt, n_gt = ndimage.label(label_bin, structure=structure)
        n_gt = int(n_gt)
        dice_global = dice_bool(pred_bin, label_bin)

        pred_slices = ndimage.find_objects(cc_pred)
        gt_slices = ndimage.find_objects(cc_gt)

        gt_boxes: List[Box] = []
        gt_ids: List[int] = []
        for j in range(1, n_gt + 1):
            sl = gt_slices[j - 1]
            if sl is None:
                continue
            gt_boxes.append(Box(sl[0].start, sl[0].stop, sl[1].start, sl[1].stop,
                                sl[2].start, sl[2].stop))
            gt_ids.append(j)

        pred_boxes: List[Box] = []
        pred_ids: List[int] = []
        pred_scores: List[float] = []
        for i in range(1, n_pred + 1):
            sl = pred_slices[i - 1]
            if sl is None:
                continue
            cmask = cc_pred[sl] == i
            size = int(cmask.sum())
            if size <= 0:
                continue
            pred_boxes.append(Box(sl[0].start, sl[0].stop, sl[1].start, sl[1].stop,
                                  sl[2].start, sl[2].stop))
            pred_ids.append(i)
            pred_scores.append(float(size) if score_mode == 'cluster_size'
                               else float(pred_raw[sl][cmask].max()))

        n_pred_eff = len(pred_boxes)
        n_gt_eff = len(gt_boxes)

        tp, fp, fn = greedy_match_by_box_dice(pred_boxes, gt_boxes, boxdsc_thr)

        boxdsc_detect = ppv_detect = dist_detect = dice_detect = pinpoint = False
        if n_pred_eff > 0:
            pred_sizes = np.bincount(cc_pred.ravel(), minlength=n_pred + 1).astype(np.int64)
            gt_sizes = np.bincount(cc_gt.ravel(), minlength=n_gt + 1).astype(np.int64)
            pairs = overlap_table(cc_pred, cc_gt, n_pred, n_gt)

            inter_total = np.zeros(n_pred + 1, dtype=np.int64)
            best_dice = np.zeros(n_pred + 1, dtype=np.float64)
            for (pi, gj), cnt in pairs.items():
                inter_total[pi] += cnt
                denom = pred_sizes[pi] + gt_sizes[gj]
                if denom > 0:
                    best_dice[pi] = max(best_dice[pi], (2.0 * cnt) / float(denom))

            weights = np.where(cc_pred > 0, np.maximum(pred_raw, 0.0), 0.0)
            if not np.any(weights > 0):
                weights = (cc_pred > 0).astype(np.float32)
            centroids = ndimage.center_of_mass(weights, cc_pred, pred_ids) if pred_ids else []

            dist_map = None
            if any(inter_total[i] == 0 for i in pred_ids):
                dist_map = ndimage.distance_transform_edt(~label_bin, sampling=zooms_xyz)

            shape_max = np.array(label_bin.shape) - 1
            for k, i in enumerate(pred_ids):
                size = int(pred_sizes[i])
                if strict_gt(inter_total[i] / float(size), ppv_thr):
                    ppv_detect = True
                if strict_gt(best_dice[i], dice_thr):
                    dice_detect = True
                if gt_boxes and strict_gt(max(pred_boxes[k].dice(gb) for gb in gt_boxes),
                                          boxdsc_thr):
                    boxdsc_detect = True

                if inter_total[i] > 0:
                    dist_detect = True
                elif dist_map is not None and not dist_detect:
                    sl = pred_slices[i - 1]
                    cmask = cc_pred[sl] == i
                    if le_tol(float(dist_map[sl][cmask].min()), dist_thr_mm):
                        dist_detect = True

                c = centroids[k]
                if not any(np.isnan(v) for v in c):
                    cidx = np.clip(np.round(np.asarray(c)).astype(int), 0, shape_max)
                    if label_bin[tuple(cidx)]:
                        pinpoint = True

        is_tp = per_case_detection_flags(pred_scores, pred_boxes, gt_boxes, boxdsc_thr)
        return sid, 'patient', {
            'id': sid,
            'pred_empty': bool(pred_vox == 0),
            'dice_global': float(dice_global),
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn),
            'n_pred': int(n_pred_eff), 'n_gt': int(n_gt_eff),
            'boxdsc_detect': bool(boxdsc_detect),
            'ppv_detect': bool(ppv_detect),
            'dist_detect': bool(dist_detect),
            'dice_detect': bool(dice_detect),
            'pinpoint': bool(pinpoint),
            'det_scores': [float(s) for s in pred_scores],
            'det_is_tp': [bool(v) for v in is_tp],
            'pred_binary': bool(np.array_equal(np.unique(pred_raw), np.array([0], np.float32))
                                or set(np.unique(pred_raw).tolist()) <= {0.0, 1.0}),
        }, None

    except Exception as e:  # noqa: BLE001
        return sid, 'error', None, f'[{sid}] exception: {e!r}'


# bootstrap
_G: Dict[str, Any] = {}


def _bootstrap_init(pack: Dict[str, Any]):
    _G.clear()
    _G.update(pack)


def _bootstrap_worker(start: int, count: int, seed: int) -> Dict[str, np.ndarray]:
    H, P = _G['H'], _G['P']
    out = {k: np.full(count, np.nan, dtype=np.float64) for k in METRIC_ORDER}

    for local, t in enumerate(range(start, start + count)):
        rng = np.random.default_rng(np.random.SeedSequence([seed, t]))

        if H > 0:
            ih = rng.integers(0, H, size=H)
            out['Specificity'][local] = float(_G['healthy_spec'][ih].mean())
            out['False Positives/Subject (Healthy)'][local] = float(_G['healthy_n_pred'][ih].mean())

        if P == 0:
            continue
        ip = rng.integers(0, P, size=P)

        out['Empty'][local] = float(_G['pat_empty'][ip].mean())
        out['dice_average'][local] = float(_G['pat_dice'][ip].mean())

        tp = int(_G['pat_tp'][ip].sum())
        fp = int(_G['pat_fp'][ip].sum())
        fn = int(_G['pat_fn'][ip].sum())
        out['Precision-boxDSC'][local] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        out['Sensitivity-boxDSC'][local] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        out['F1-score'][local] = ((2 * tp) / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0

        out['Number/Subject'][local] = float(_G['pat_n_pred'][ip].mean())
        out['False Positives/Subject'][local] = float(_G['pat_fp'][ip].mean())
        out['Detection rate-boxDSC'][local] = float(_G['pat_box'][ip].mean())
        out['Detection rate-PPV'][local] = float(_G['pat_ppv'][ip].mean())
        out['Detection rate-distance'][local] = float(_G['pat_dist'][ip].mean())
        out['Detection rate-dice'][local] = float(_G['pat_dice_det'][ip].mean())
        out['Pinpointing rate'][local] = float(_G['pat_pin'][ip].mean())

        scores = _G['det_scores']
        istp = _G['det_is_tp']
        picked_scores = [scores[i] for i in ip.tolist() if scores[i].size]
        picked_istp = [istp[i] for i in ip.tolist() if istp[i].size]
        total_gt = int(_G['pat_n_gt'][ip].sum())
        if picked_scores:
            out['AUPRC'][local] = average_precision(np.concatenate(picked_scores),
                                                    np.concatenate(picked_istp), total_gt)
        else:
            out['AUPRC'][local] = 0.0 if total_gt > 0 else float('nan')
    return out


def _mp_context():
    try:
        return mp.get_context('fork')
    except ValueError:
        return mp.get_context()



def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pred_dir', nargs='+', required=True, help='folder(s) with predictions')
    ap.add_argument('--label_dir', nargs='+', required=True, help='folder(s) with ground truth')
    ap.add_argument('--suffix', default='.nii.gz')
    ap.add_argument('--id_regex', default=r'sub-test(\d+)',
                    help=r'regex whose group 1 is the subject id. Default: sub-test(\d+)')
    ap.add_argument('--pred_glob', default=None, help='default: *{suffix}')
    ap.add_argument('--label_glob', default=None, help='default: *{suffix}')
    ap.add_argument('--id_min', type=int, default=None, help='keep ids >= this (numeric ids only)')
    ap.add_argument('--id_max', type=int, default=None, help='keep ids <= this')

    ap.add_argument('--threshold', type=float, default=0.5, help='pred > threshold (default 0.5)')
    ap.add_argument('--connectivity', type=int, default=26, choices=[6, 18, 26],
                    help='connected-component connectivity (default 26)')
    ap.add_argument('--min_pred_cluster_vox', type=int, default=100,
                    help='drop predicted clusters smaller than this (default 100)')
    ap.add_argument('--healthy_fp_tol_vox', type=int, default=0,
                    help='a healthy subject counts as a true negative if the predicted '
                         'volume is <= this, after small clusters are dropped (default 0)')
    ap.add_argument('--score_mode', choices=['prob_max', 'cluster_size'], default='prob_max',
                    help='how a cluster is ranked for AUPRC. prob_max needs real '
                         'probabilities; use cluster_size for binary segmentations')

    ap.add_argument('--boxdsc_thr', type=float, default=0.22)
    ap.add_argument('--ppv_thr', type=float, default=0.5)
    ap.add_argument('--dist_thr_mm', type=float, default=20.0)
    ap.add_argument('--dice_thr', type=float, default=0.15)

    ap.add_argument('--bootstrap', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=20262026)
    ap.add_argument('--n_jobs', type=int, default=15)

    ap.add_argument('--name', default=None, help='tag written into the json/csv output')
    ap.add_argument('--out_json', default=None)
    ap.add_argument('--out_csv', default=None, help='append one row of metrics')
    ap.add_argument('--per_case_csv', default=None)
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()

    pred_dirs = [Path(p) for p in args.pred_dir]
    label_dirs = [Path(p) for p in args.label_dir]
    pred_map = build_id_map(pred_dirs, args.pred_glob or f'*{args.suffix}', args.id_regex, 'pred')
    label_map = build_id_map(label_dirs, args.label_glob or f'*{args.suffix}', args.id_regex, 'label')

    common = sorted(set(pred_map) & set(label_map))
    if args.id_min is not None or args.id_max is not None:
        lo = args.id_min if args.id_min is not None else -10 ** 9
        hi = args.id_max if args.id_max is not None else 10 ** 9
        common = [s for s in common if s.isdigit() and lo <= int(s) <= hi]
    if not common:
        raise SystemExit('no subject id matched between --pred_dir and --label_dir; '
                         'check --id_regex and the file names')

    only_pred = sorted(set(pred_map) - set(label_map))
    only_label = sorted(set(label_map) - set(pred_map))
    if not args.quiet:
        print(f'[info] matched {len(common)} cases '
              f'({len(only_pred)} prediction-only, {len(only_label)} ground-truth-only)',
              file=sys.stderr)
        if only_label[:5]:
            print(f'[info] first cases without a prediction: {only_label[:5]}', file=sys.stderr)

    n_jobs = max(1, int(args.n_jobs))
    ctx = _mp_context()

    healthy: List[Dict[str, Any]] = []
    patients: List[Dict[str, Any]] = []
    errors: List[str] = []

    with ProcessPoolExecutor(max_workers=n_jobs, mp_context=ctx) as ex:
        futures = [ex.submit(_compute_subject_record, sid, str(pred_map[sid]),
                             str(label_map[sid]), float(args.threshold), int(args.connectivity),
                             int(args.min_pred_cluster_vox), int(args.healthy_fp_tol_vox),
                             float(args.boxdsc_thr), float(args.ppv_thr),
                             float(args.dist_thr_mm), float(args.dice_thr), args.score_mode)
                   for sid in common]
        for f in as_completed(futures):
            sid, group, rec, warn = f.result()
            if warn:
                errors.append(warn)
            if rec is None:
                continue
            (healthy if group == 'healthy' else patients).append(rec)

    for w in sorted(errors):
        print(w, file=sys.stderr)

    healthy.sort(key=lambda r: r['id'])
    patients.sort(key=lambda r: r['id'])
    H, P = len(healthy), len(patients)
    if not args.quiet:
        print(f'[info] {H} healthy (empty ground truth), {P} patients, '
              f'{len(errors)} unreadable', file=sys.stderr)

    if args.score_mode == 'prob_max' and patients and all(r['pred_binary'] for r in patients):
        print('[warn] the predictions are binary segmentations (0/1 only), so every '
              'cluster scores 1.0 and AUPRC collapses to a single operating point. '
              'Either predict with --save_probabilities, or pass '
              '--score_mode cluster_size to rank clusters by size.', file=sys.stderr)

    healthy_spec = np.array([r['spec_success'] for r in healthy], dtype=bool)
    healthy_n_pred = np.array([r['n_pred'] for r in healthy], dtype=np.int64)
    pat_empty = np.array([r['pred_empty'] for r in patients], dtype=bool)
    pat_dice = np.array([r['dice_global'] for r in patients], dtype=np.float64)
    pat_tp = np.array([r['tp'] for r in patients], dtype=np.int64)
    pat_fp = np.array([r['fp'] for r in patients], dtype=np.int64)
    pat_fn = np.array([r['fn'] for r in patients], dtype=np.int64)
    pat_n_pred = np.array([r['n_pred'] for r in patients], dtype=np.int64)
    pat_n_gt = np.array([r['n_gt'] for r in patients], dtype=np.int64)
    pat_box = np.array([r['boxdsc_detect'] for r in patients], dtype=bool)
    pat_ppv = np.array([r['ppv_detect'] for r in patients], dtype=bool)
    pat_dist = np.array([r['dist_detect'] for r in patients], dtype=bool)
    pat_dice_det = np.array([r['dice_detect'] for r in patients], dtype=bool)
    pat_pin = np.array([r['pinpoint'] for r in patients], dtype=bool)
    det_scores = [np.asarray(r['det_scores'], dtype=np.float64) for r in patients]
    det_is_tp = [np.asarray(r['det_is_tp'], dtype=bool) for r in patients]

    nan = float('nan')
    tp, fp, fn = int(pat_tp.sum()), int(pat_fp.sum()), int(pat_fn.sum())
    est = {
        'Specificity': float(healthy_spec.mean()) if H else nan,
        'False Positives/Subject (Healthy)': float(healthy_n_pred.mean()) if H else nan,
        'Empty': float(pat_empty.mean()) if P else nan,
        'dice_average': float(pat_dice.mean()) if P else nan,
        'Precision-boxDSC': (tp / (tp + fp) if (tp + fp) > 0 else (0.0 if P else nan)),
        'Sensitivity-boxDSC': (tp / (tp + fn) if (tp + fn) > 0 else (0.0 if P else nan)),
        'F1-score': ((2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else (0.0 if P else nan)),
        'Number/Subject': float(pat_n_pred.mean()) if P else nan,
        'False Positives/Subject': float(pat_fp.mean()) if P else nan,
        'AUPRC': (average_precision(np.concatenate(det_scores) if any(s.size for s in det_scores)
                                    else np.zeros(0),
                                    np.concatenate(det_is_tp) if any(s.size for s in det_is_tp)
                                    else np.zeros(0, dtype=bool),
                                    int(pat_n_gt.sum())) if P else nan),
        'Detection rate-boxDSC': float(pat_box.mean()) if P else nan,
        'Detection rate-PPV': float(pat_ppv.mean()) if P else nan,
        'Detection rate-distance': float(pat_dist.mean()) if P else nan,
        'Detection rate-dice': float(pat_dice_det.mean()) if P else nan,
        'Pinpointing rate': float(pat_pin.mean()) if P else nan,
    }

    n_boot = int(args.bootstrap)
    if n_boot <= 1:
        ci = {k: (v, v) for k, v in est.items()}
    else:
        pack = {'H': H, 'P': P, 'healthy_spec': healthy_spec, 'healthy_n_pred': healthy_n_pred,
                'pat_empty': pat_empty, 'pat_dice': pat_dice, 'pat_tp': pat_tp, 'pat_fp': pat_fp,
                'pat_fn': pat_fn, 'pat_n_pred': pat_n_pred, 'pat_n_gt': pat_n_gt,
                'pat_box': pat_box, 'pat_ppv': pat_ppv, 'pat_dist': pat_dist,
                'pat_dice_det': pat_dice_det, 'pat_pin': pat_pin,
                'det_scores': det_scores, 'det_is_tp': det_is_tp}
        chunks, start = [], 0
        per = max(1, -(-n_boot // n_jobs))
        while start < n_boot:
            chunks.append((start, min(per, n_boot - start)))
            start += per
        collected = {k: [] for k in METRIC_ORDER}
        with ProcessPoolExecutor(max_workers=len(chunks), mp_context=ctx,
                                 initializer=_bootstrap_init, initargs=(pack,)) as ex:
            futures = [ex.submit(_bootstrap_worker, s, c, int(args.seed)) for s, c in chunks]
            for f in as_completed(futures):
                for k, arr in f.result().items():
                    collected[k].append(arr)
        ci = {}
        for k in METRIC_ORDER:
            vals = np.concatenate(collected[k]) if collected[k] else np.array([np.nan])
            vals = vals[~np.isnan(vals)]
            ci[k] = ((float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
                     if vals.size else (nan, nan))

    for k in METRIC_ORDER:
        print(fmt_metric(k, est[k], ci[k][0], ci[k][1]))

    payload = {
        'name': args.name, 'pred_dir': [str(p) for p in pred_dirs],
        'label_dir': [str(p) for p in label_dirs],
        'n_matched': len(common), 'n_healthy': H, 'n_patient': P, 'n_error': len(errors),
        'settings': {'threshold': args.threshold, 'connectivity': args.connectivity,
                     'min_pred_cluster_vox': args.min_pred_cluster_vox,
                     'healthy_fp_tol_vox': args.healthy_fp_tol_vox,
                     'boxdsc_thr': args.boxdsc_thr, 'ppv_thr': args.ppv_thr,
                     'dist_thr_mm': args.dist_thr_mm, 'dice_thr': args.dice_thr,
                     'score_mode': args.score_mode, 'bootstrap': args.bootstrap,
                     'seed': args.seed},
        'metrics': {k: {'estimate': est[k], 'ci_low': ci[k][0], 'ci_high': ci[k][1]}
                    for k in METRIC_ORDER},
    }
    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        with open(args.out_json, 'w') as f:
            json.dump(payload, f, indent=2)
    if args.out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
        header = ['name', 'n_healthy', 'n_patient'] + [f'{k}{s}' for k in METRIC_ORDER
                                                       for s in ('', '_lo', '_hi')]
        row = {'name': args.name or '', 'n_healthy': H, 'n_patient': P}
        for k in METRIC_ORDER:
            row[k], row[f'{k}_lo'], row[f'{k}_hi'] = est[k], ci[k][0], ci[k][1]
        exists = os.path.isfile(args.out_csv)
        with open(args.out_csv, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=header)
            if not exists:
                w.writeheader()
            w.writerow(row)
    if args.per_case_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.per_case_csv)), exist_ok=True)
        with open(args.per_case_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['id', 'group', 'n_pred', 'n_gt', 'dice_global', 'tp', 'fp', 'fn',
                        'boxdsc', 'ppv', 'dist', 'dice_det', 'pinpoint'])
            for r in healthy:
                w.writerow([r['id'], 'healthy', r['n_pred'], 0, '', '', '', '', '', '', '', '', ''])
            for r in patients:
                w.writerow([r['id'], 'patient', r['n_pred'], r['n_gt'], f"{r['dice_global']:.6f}",
                            r['tp'], r['fp'], r['fn'], int(r['boxdsc_detect']),
                            int(r['ppv_detect']), int(r['dist_detect']),
                            int(r['dice_detect']), int(r['pinpoint'])])
    return 0


if __name__ == '__main__':
    sys.exit(main())
