"""K direction bins on the circle: the student's fine action space (P4, docs/policy_v2_design_ko.md).

Unity's ``Unit.Move`` uses ``input.normalized``: only the heading of a move matters (the zero
vector stands).  A K-bin student emits one logit per bin -- bin ``k`` is the heading at angle
``2 pi k / K`` -- plus, with ``--stand-action``, a STAND logit after them (index K).  Bins
``0, K/8, 2K/8, ...`` are the 8 compass directions of ``my_policy.DIRECTION_VECTORS``, so K = 8
is exactly the 8-way action space, and every helper here reduces to the 8-way one at K = 8
(tests/test_direction_bins.py checks it bit for bit).

K must be a multiple of 8 (the compass directions are bins, and the team-B mirror
``t -> 90 deg - t`` maps bins to bins), and at most 64 (labels are stored as uint64 bit sets).
An output row of width ``n`` is ``K = n`` bins, or ``K = n - 1`` bins + STAND when ``n`` is odd.

Labels are bit sets over the K bins: uint8 at K = 8 (the format every shard before P4 used),
uint64 otherwise.  ``widen_mask`` turns an 8-direction teacher set into the bins within a half
width of its directions; ``arc_mask`` labels a continuous move (an act-teacher's heading) with
its nearest bin and ``half_bins`` neighbours on each side; ``convert_bits`` re-expresses a stored
label table at another K (an old uint8 shard in a K-bin ring).
"""

from __future__ import annotations

import math

import numpy as np

N_DIRECTIONS = 8  # the compass directions (my_policy.DIRECTION_VECTORS is bin_vectors(8))

ACTION_BINS = (8, 16, 32, 64)
MAX_BINS = 64
SET_HALF_WIDTH_DEG = 22.5  # an allowed compass direction widens to the bins within this of it
_ANGLE_EPS = 1e-6  # degrees: a bin exactly on the half width counts as inside


def check_bins(k: int) -> int:
    k = int(k)
    if k < N_DIRECTIONS or k % N_DIRECTIONS or k > MAX_BINS:
        raise ValueError(f"action bins must be a multiple of 8 in [8, {MAX_BINS}], got {k}")
    return k


def bins_of_width(n_outputs: int) -> tuple[int, bool]:
    """(K, has STAND) of a head with ``n_outputs`` logits per unit."""
    stand = n_outputs % 2 == 1
    return check_bins(n_outputs - int(stand)), stand


# K = 8 written exactly as my_policy always computed it (index i at i * 45 deg), so the 8-way
# vectors keep their float32 bits.
_VECTORS: dict[int, np.ndarray] = {N_DIRECTIONS: np.array(
    [[math.cos(i * math.pi / 4), math.sin(i * math.pi / 4)] for i in range(N_DIRECTIONS)], dtype=np.float32)}


def bin_vectors(k: int) -> np.ndarray:
    """[K, 2] float32 unit vectors, bin k at angle 2 pi k / K (K = 8: DIRECTION_VECTORS itself)."""
    if k not in _VECTORS:
        check_bins(k)
        _VECTORS[k] = np.array([[math.cos(2 * math.pi * i / k), math.sin(2 * math.pi * i / k)] for i in range(k)],
                               dtype=np.float32)
    return _VECTORS[k]


def action_vectors(k: int, stand: bool) -> np.ndarray:
    """What each output index executes: the K bin headings, then the zero move for STAND."""
    vectors = bin_vectors(k)
    return np.vstack([vectors, np.zeros((1, 2), np.float32)]) if stand else vectors


def vector_to_bin(vectors: np.ndarray, k: int) -> np.ndarray:
    """Nearest bin of [..., 2] moves by max dot product (``direction_vector_to_idx`` at K = 8,
    zero vector -> bin 0 like it)."""
    return (np.asarray(vectors, np.float32) @ bin_vectors(k).T).argmax(axis=-1)


def mirror_bin(idx, k: int):
    """Team-B reflection of bin indices (angle t -> 90 deg - t): k -> (K/4 - k) mod K."""
    if isinstance(idx, (int, np.integer)):
        return int((k // 4 - int(idx)) % k)
    return (k // 4 - np.asarray(idx, np.int64)) % k


def _angular_gap_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    gap = np.abs(a - b) % 360.0
    return np.minimum(gap, 360.0 - gap)


def widen_mask(mask: np.ndarray, k: int, half_width_deg: float = SET_HALF_WIDTH_DEG, src_bins: int | None = None) -> np.ndarray:
    """[..., S] bool (S = ``src_bins`` or the last axis; 8 for a teacher's compass set) -> [..., K]
    bool: every bin whose centre is within ``half_width_deg`` (inclusive) of an allowed source bin.
    At K = S and a half width below one bin spacing, the mask is unchanged."""
    mask = np.asarray(mask, bool)
    s = mask.shape[-1] if src_bins is None else src_bins
    if s == k and half_width_deg < 360.0 / k:
        return mask.copy()
    near = _angular_gap_deg(np.arange(k)[:, None] * 360.0 / k, np.arange(s)[None, :] * 360.0 / s) <= half_width_deg + _ANGLE_EPS
    return (mask[..., None, :] & near).any(axis=-1)  # [..., K, S] -> [..., K]


def arc_mask(move: np.ndarray, k: int, half_bins: int = 0) -> np.ndarray:
    """[K] bool: the bin nearest ``move``'s heading and ``half_bins`` bins either side; all False
    for a zero move (the caller decides what standing means)."""
    out = np.zeros(k, bool)
    move = np.asarray(move, np.float32)
    if not np.any(move):
        return out
    centre = int(vector_to_bin(move, k))
    out[(centre + np.arange(-half_bins, half_bins + 1)) % k] = True
    return out


def label_dtype(k: int):
    return np.uint8 if k == N_DIRECTIONS else np.uint64


def pack_bits(mask: np.ndarray) -> np.ndarray:
    """[..., K] bool -> [...] bit sets (uint8 at K = 8, else uint64), bit i = bin i."""
    mask = np.asarray(mask, bool)
    k = mask.shape[-1]
    if k == N_DIRECTIONS:
        return np.packbits(mask, axis=-1, bitorder="little")[..., 0]
    weights = np.left_shift(np.uint64(1), np.arange(k, dtype=np.uint64))
    return (mask.astype(np.uint64) * weights).sum(axis=-1, dtype=np.uint64)


def unpack_bits(bits: np.ndarray, k: int) -> np.ndarray:
    """[...] bit sets -> [..., K] bool (uint8 or uint64 input)."""
    bits = np.asarray(bits)
    if bits.dtype == np.uint8 and k == N_DIRECTIONS:
        return np.unpackbits(bits[..., None], axis=-1, bitorder="little").astype(bool)
    shifts = np.arange(k, dtype=np.uint64)
    return ((bits.astype(np.uint64)[..., None] >> shifts) & np.uint64(1)).astype(bool)


def mirror_bits(bits: np.ndarray, k: int) -> np.ndarray:
    """Reflect a bit-set table's bins (no unit reordering)."""
    mask = unpack_bits(bits, k)
    return pack_bits(mask[..., mirror_bin(np.arange(k), k)])  # bit j of the result = bit mirror(j)


def bits_bins(bits: np.ndarray, meta_bins: int | None = None) -> int:
    """K of a stored label table: uint8 = the 8-way format; uint64 = ``meta_bins`` (the match
    summary's ``action_bins``), else 64."""
    if np.asarray(bits).dtype == np.uint8:
        return N_DIRECTIONS
    return check_bins(meta_bins) if meta_bins else MAX_BINS


def convert_bits(bits: np.ndarray, src_bins: int, dst_bins: int) -> np.ndarray:
    """A label table at ``src_bins`` re-expressed at ``dst_bins``: each target bin is allowed when
    an allowed source bin lies within half the coarser spacing of it (8 -> K: the +-22.5 deg
    widening of the set teachers).  Same K: returned as is."""
    if src_bins == dst_bins:
        return bits
    half = 180.0 / min(src_bins, dst_bins)
    return pack_bits(widen_mask(unpack_bits(bits, src_bins), dst_bins, half, src_bins=src_bins))


def resample_rows(rows, src_bins: int, dst_bins: int):
    """[S, ...] per-bin parameter rows (torch or numpy) -> [K, ...]: bin k takes the linear
    interpolation, in angle, of the two source rows around it.  The logit of a linear output
    layer interpolates the same way, so the argmax over K bins is always a source bin (strictly,
    unless its two neighbours tie): a K-bin head warm-started this way moves exactly like the
    source head."""
    pos = np.arange(dst_bins) * src_bins / dst_bins
    lo = np.floor(pos).astype(np.int64)
    t = pos - lo
    hi = (lo + 1) % src_bins
    out = []
    for i in range(dst_bins):
        a, b, w = int(lo[i]), int(hi[i]), float(t[i])
        out.append(rows[a] if w == 0.0 else rows[a] * (1.0 - w) + rows[b] * w)
    if isinstance(rows, np.ndarray):
        return np.stack(out)
    import torch

    return torch.stack(out)


# ------------------------------------------------------------------------------ continuous angle (P11)
#
# An angle head (``MyModel(angle_head=True)``, P11) adds to every bin k an offset in
# [-pi/K, pi/K]: the heading it executes is ``2 pi k / K + offset_k``.  Angles here are radians
# in the same frame as the bins (0 = +x, counter-clockwise).  The team-B mirror ``t -> pi/2 - t``
# takes bin k to ``mirror_bin(k)`` and negates its offset.


def wrap_angle(angle):
    """Radians wrapped to [-pi, pi) (numpy or torch)."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def move_angle(move: np.ndarray) -> float:
    """Heading of one continuous move in radians, NaN for a move that stands (see act_teacher.ZERO_MOVE)."""
    move = np.asarray(move, np.float64)
    if float(np.hypot(move[0], move[1])) < 1e-5:
        return math.nan
    return math.atan2(move[1], move[0])


def mirror_angle(angle):
    """Team-B reflection of headings (radians, NaN stays NaN): t -> pi/2 - t, wrapped."""
    return wrap_angle(math.pi / 2 - np.asarray(angle, np.float64))


def angle_bin(angle, k: int):
    """The bin whose centre is nearest each heading (radians; NaN -> 0, mask it yourself)."""
    angle = np.nan_to_num(np.asarray(angle, np.float64))
    return np.rint(angle * k / (2 * math.pi)).astype(np.int64) % k


def angle_residual(angle, k: int):
    """Heading minus its nearest bin's centre, in [-pi/K, pi/K] (radians)."""
    angle = np.asarray(angle, np.float64)
    return wrap_angle(angle - angle_bin(angle, k) * 2 * math.pi / k)


def offset_vectors(offsets: np.ndarray) -> np.ndarray:
    """[..., K] per-bin offsets (radians) -> [..., K, 2] float32 unit headings 2 pi k / K + offset_k."""
    offsets = np.asarray(offsets, np.float64)
    k = offsets.shape[-1]
    angles = 2 * math.pi * np.arange(k) / k + offsets
    return np.stack([np.cos(angles), np.sin(angles)], axis=-1).astype(np.float32)
