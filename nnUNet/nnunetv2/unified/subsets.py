"""Definition of the M0-M4 training sets, the test set, and the head routing.

A "case identifier" here is the nnU-Net one, i.e. ``sub-test002`` -- the file name
of ``imagesTr/sub-test002_0000.nii.gz`` without the channel suffix.

Cohorts
-------
``cohort_a``  subjects 001-170   (0.8 mm isotropic)
``cohort_b``  subjects 400+      (0.83 x 0.83 x 1.0 mm)
``test``      subjects 174-267, 301-319 and 320-352

Training sets
-------------
M0  63 hand-picked subjects out of cohort_a (size-matched to cohort_b)
M1  all of cohort_a                                   (170 subjects)
M2  all of cohort_b                                   (63 subjects)
M3  cohort_a + cohort_b, single output head           (233 subjects)
M4  cohort_a + cohort_b, cohort_a -> head1, cohort_b -> head2
"""
import re
from typing import Dict, List, Optional, Sequence, Set

CASE_RE = re.compile(r'^sub-test(\d+)$')

#: The 63 subjects that make up M0: a control for M2, matched on size (63 vs 63),
#: on lesion prevalence (51 patients + 12 healthy, same as 400+) and -- unlike the
#: original hand-picked list -- on the per-fold composition of the seeded 5-fold
#: split as well. Found by scripts/match_m0_folds.py, which resampled 51 patients
#: and 12 healthy subjects out of 001-170 (85/85 overall) until the split produced
#: M2's per-fold counts exactly: 13:10/3, 13:11/2, 13:12/1, 12:9/3, 12:9/3.
M0_IDS: List[int] = [
    1, 4, 6, 10, 15, 16, 18, 20, 27, 28, 32, 34, 37, 42, 43, 44, 57, 58, 61, 63,
    66, 71, 72, 73, 74, 78, 80, 83, 88, 89, 90, 91, 92, 95, 97, 98, 99, 100, 103,
    107, 108, 109, 110, 115, 120, 121, 122, 124, 125, 126, 127, 128, 132, 134,
    136, 138, 141, 142, 144, 148, 157, 159, 168,
]

#: The original hand-picked M0, kept for provenance. It matched M2 in total
#: (51 patients + 12 healthy) but not fold by fold: its per-fold counts were
#: 13:11/2, 13:10/3, 13:9/4, 12:11/1, 12:10/2 against M2's 13:10/3, 13:11/2,
#: 13:12/1, 12:9/3, 12:9/3. 28 of its 63 subjects survive in the list above.
M0_IDS_ORIGINAL: List[int] = [
    2, 4, 6, 9, 10, 14, 17, 18, 20, 22, 24, 27, 32, 33, 34, 38, 40, 43, 55, 58,
    60, 63, 68, 71, 74, 76, 77, 80, 85, 89, 92, 96, 98, 100, 101, 105, 106, 108,
    109, 112, 115, 116, 117, 119, 120, 121, 123, 126, 128, 129, 130, 131, 132,
    133, 135, 136, 138, 139, 140, 145, 146, 154, 160,
]

COHORT_A_RANGE = (1, 170)     # inclusive
COHORT_B_MIN = 400            # everything numbered >= 400
#: Held-out test subjects. These go into imagesTs and never take part in training.
#: One entry per cohort the test set is assembled from.
TEST_RANGES = ((174, 267), (301, 319), (320, 352))

M_NAMES = ('M0', 'M1', 'M2', 'M3', 'M4')

#: M4 is M3 plus a second output head. Same images, same preprocessing.
M_EQUIVALENT_DATA = {'M0': 'M0', 'M1': 'M1', 'M2': 'M2', 'M3': 'M3', 'M4': 'M3'}

#: Which nnU-Net dataset each M setting is trained on.
M_TO_DATASET_ID = {'M0': 1, 'M1': 2, 'M2': 3, 'M3': 4, 'M4': 5}
M_TO_DATASET_NAME = {
    'M0': 'Dataset001_M0', 'M1': 'Dataset002_M1', 'M2': 'Dataset003_M2',
    'M3': 'Dataset004_M3', 'M4': 'Dataset005_M4',
}


def case_id(case: str) -> int:
    """``sub-test002`` -> ``2``. Raises for anything that is not a subject folder."""
    m = CASE_RE.match(case)
    if m is None:
        raise ValueError(f'cannot parse a subject number out of case identifier {case!r}')
    return int(m.group(1))


def case_name(num: int) -> str:
    return f'sub-test{num:03d}'


def in_cohort_a(num: int) -> bool:
    return COHORT_A_RANGE[0] <= num <= COHORT_A_RANGE[1]


def in_cohort_b(num: int) -> bool:
    return num >= COHORT_B_MIN


def in_test(num: int) -> bool:
    return any(lo <= num <= hi for lo, hi in TEST_RANGES)


def cohort_of(case: str) -> str:
    """'a', 'b', 'test' or 'other' -- used for stratified splitting and head routing."""
    num = case_id(case)
    if in_cohort_a(num):
        return 'a'
    if in_cohort_b(num):
        return 'b'
    if in_test(num):
        return 'test'
    return 'other'


def belongs_to(case: str, m: str) -> bool:
    """Is ``case`` part of training set ``m``?"""
    num = case_id(case)
    if m == 'M0':
        return num in set(M0_IDS)
    if m == 'M1':
        return in_cohort_a(num)
    if m == 'M2':
        return in_cohort_b(num)
    if m in ('M3', 'M4'):
        return in_cohort_a(num) or in_cohort_b(num)
    raise ValueError(f'unknown training set {m!r}; expected one of {M_NAMES}')


def filter_cases(cases: Sequence[str], m: str) -> List[str]:
    return sorted([c for c in cases if belongs_to(c, m)], key=case_id)


def head_of(case: str, m: str, head2_cases: Optional[Set[str]] = None) -> int:
    """Return 0 for head1 and 1 for head2.

    Without ``-head2`` everything goes through head1. With ``-head2`` the listed
    cases go through head2. For M4 the default (when no list is given) is the
    definition from the protocol: cohort_a -> head1, cohort_b -> head2.
    """
    if head2_cases is not None:
        return 1 if case in head2_cases else 0
    if m == 'M4':
        return 1 if cohort_of(case) == 'b' else 0
    return 0


def read_head2_list(path: str) -> Set[str]:
    """Read the ``-head2`` txt: one file name per line.

    Accepts bare case identifiers (``sub-test401``), file names with the nnU-Net
    channel suffix (``sub-test401_0000.nii.gz``) and plain file names
    (``sub-test401.nii.gz``) -- all are reduced to the case identifier.
    The literal value ``all`` routes every case to head2, which is what M4
    inference needs: every test case goes through head2.
    """
    if path.strip().lower() == 'all':
        return {'__ALL__'}

    out: Set[str] = set()
    with open(path, 'r', encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            name = line.split('/')[-1]
            for suffix in ('.nii.gz', '.nii', '.nrrd', '.mha', '.npy', '.npz', '.b2nd'):
                if name.endswith(suffix):
                    name = name[: -len(suffix)]
                    break
            # strip an nnU-Net channel suffix such as _0000
            name = re.sub(r'_\d{4}$', '', name)
            out.add(name)
    if not out:
        raise ValueError(f'the -head2 file {path} did not contain any case names')
    return out


def is_head2(case: str, head2_cases: Optional[Set[str]]) -> bool:
    if not head2_cases:
        return False
    if '__ALL__' in head2_cases:
        return True
    return case in head2_cases


def generate_stratified_crossval_split(cases: Sequence[str], n_splits: int = 5, seed: int = 12345):
    """5-fold CV that keeps the cohort ratio constant across folds.

    For a single-cohort dataset (M0/M1/M2) this is just nnU-Net's seeded random
    split. For M3/M4 it matters: every fold must contain both cohorts, otherwise
    an M4 fold could end up with no training data for one of the two heads.
    """
    import numpy as np

    try:
        cases = sorted(cases, key=case_id)
        by_cohort: Dict[str, List[str]] = {}
        for c in cases:
            by_cohort.setdefault(cohort_of(c), []).append(c)
    except ValueError:
        # not this study's sub-testNNN naming -> nothing to stratify by, use
        # nnU-Net's own seeded split so the trainer still works on other datasets
        from nnunetv2.utilities.crossval_split import generate_crossval_split
        return generate_crossval_split(sorted(cases), seed=seed, n_splits=n_splits)

    # fold assignment per cohort, then merged
    folds: List[List[str]] = [[] for _ in range(n_splits)]
    for cohort in sorted(by_cohort):
        members = by_cohort[cohort]
        rng = np.random.RandomState(seed)
        order = rng.permutation(len(members))
        for rank, idx in enumerate(order):
            folds[rank % n_splits].append(members[idx])

    splits = []
    for i in range(n_splits):
        val = sorted(folds[i], key=case_id)
        train = sorted([c for j in range(n_splits) if j != i for c in folds[j]], key=case_id)
        splits.append({'train': train, 'val': val})
    return splits
