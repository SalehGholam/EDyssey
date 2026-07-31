# -*- coding: utf-8 -*-
"""Smart-scan (sparse/pattern-file) acquisition support: matching a folder's
detection/acquisition data files to their pattern files and to each other by
tilt angle, and flagging mismatches for review.

See "other_scripts/smart scanning guide/" for the exploratory reference
scripts this generalizes, and the "Smart Scanned" checkbox present in every
tab's Scan Size box for where this is used.

A tomography smart-scan run typically writes, per tilt angle, two data files
(a "detection" file - a normal dense raster, doesn't need a pattern file -
and an "acquisition" file - sparse, needs a pattern file to reshape) plus one
per-angle pattern .txt file, all alongside one shared comment.txt (see
metadata.get_metadata/get_metadata_block_count) and one shared pattern.txt
(an unrelated file, always skipped here). File naming isn't fully
standardized between datasets/instruments - `parse_angle` is deliberately a
best-effort heuristic rather than tied to one exact scheme, which is why
every call site of `match_tilt_files` is expected to let the user review and
correct its output (see ui_tabs/smart_scan_dialog.py's SmartScanCheckDialog)
rather than trust it blindly.

Data files: .tpx3 is read natively (eventem's own `set_pattern_file`) - see
`loaders.load_tpx3`. .mib/.hspy/.zspy are all loaded through HyperSpy
identically (`hs.load`, just dispatching to a different reader plugin per
extension) - to HyperSpy, and to this module, a smart-scanned acquisition in
any one of those three is the same shape of problem: a flat, un-reshaped
stream of acquired frames that needs `loaders._reconstruct_smart_scan` (a
full dense-grid rebuild) or, for single/few-frame masked extraction,
`worker_extract_frame.py`'s direct pattern-index frame selection.
"""
import os
import re
from glob import glob

_NUMERIC_TOKEN_RE = re.compile(r'-?\d+(?:[.,]\d+)?')

# Data-file extensions smart-scan matching/reconstruction currently
# supports - .tpx3 natively (eventem), .mib/.hspy/.zspy via HyperSpy (see
# module docstring above).
DATA_EXTENSIONS = ('.tpx3', '.mib', '.hspy', '.zspy')

# Row 'status' values match_tilt_files can report (a row may have several).
STATUS_OK = 'ok'
STATUS_MISSING_DETECTION = 'missing_detection'
STATUS_MISSING_ACQUISITION = 'missing_acquisition'
STATUS_MISSING_PATTERN = 'missing_pattern'
STATUS_EXTRA_FILES = 'extra_files'
STATUS_ANGLE_UNPARSED = 'angle_unparsed'
STATUS_UNMATCHED_PATTERN = 'unmatched_pattern'

STATUS_LABELS = {
    STATUS_OK: 'OK',
    STATUS_MISSING_DETECTION: 'Missing detection file',
    STATUS_MISSING_ACQUISITION: 'Missing acquisition file',
    STATUS_MISSING_PATTERN: 'Missing pattern file',
    STATUS_EXTRA_FILES: 'Extra file(s) at this angle',
    STATUS_ANGLE_UNPARSED: "Couldn't parse a tilt angle",
    STATUS_UNMATCHED_PATTERN: 'Pattern file matches no data file',
}


def parse_angle(filename):
    """Best-effort tilt angle extraction from a filename.

    Numeric tokens (signed, comma- or dot-decimal) are extracted from the
    filename stem; the last one *with a decimal point* is preferred (an
    index/counter field elsewhere in the name, e.g. "raw_0000_-50.00_000000",
    is almost always a bare integer, while a logged angle almost always has a
    fractional part) - if none has a decimal point, the last numeric token of
    any kind is used instead (covers whole-number angles, e.g. "..._45.mib").

    Returns:
        float, or None if the filename has no numeric token at all.
    """
    stem = os.path.splitext(os.path.basename(filename))[0]
    tokens = _NUMERIC_TOKEN_RE.findall(stem)
    if not tokens:
        return None
    with_decimal = [t for t in tokens if '.' in t or ',' in t]
    token = with_decimal[-1] if with_decimal else tokens[-1]
    try:
        return float(token.replace(',', '.'))
    except ValueError:
        return None


def _group_by_angle(fns):
    """{angle: [fn, ...]} (sorted, ascending angle) plus a list of files
    parse_angle couldn't assign an angle to at all."""
    groups = {}
    unparsed = []
    for fn in fns:
        angle = parse_angle(fn)
        if angle is None:
            unparsed.append(fn)
        else:
            groups.setdefault(angle, []).append(fn)
    for fns_at_angle in groups.values():
        fns_at_angle.sort()
    return groups, unparsed


def match_tilt_files(data_dir, data_ext, pattern_dir=None, angle_tolerance=0.05):
    """Match a folder's smart-scan data files (detection + acquisition, by
    role) to their pattern files, by tilt angle.

    Args:
        data_dir: Folder containing the data files.
        data_ext: Extension to match - one of DATA_EXTENSIONS.
        pattern_dir: Folder containing the per-angle pattern .txt files -
            defaults to `data_dir` (pattern files usually sit alongside the
            data).
        angle_tolerance: Max absolute angle difference (degrees) for a
            pattern file to be considered a match for a data-file angle
            group - accounts for the data file's and pattern file's angles
            occasionally being logged at slightly different precision.

    Returns:
        List of row dicts, one per detected tilt angle (plus one row per
        unmatched/unparsed file), each with keys:
            'angle': float or None,
            'detection_file', 'acquisition_file', 'pattern_file': str or None,
            'extra_files': list of str (data files beyond the expected
                detection+acquisition pair, at the same angle),
            'status': list of STATUS_* strings (['ok'] if clean),
            'excluded': bool - True whenever `status` isn't just ['ok'];
                SmartScanCheckDialog toggles this per row, and callers should
                skip any row with `excluded` True.
        Sorted by angle (angle=None rows last).
    """
    pattern_dir = pattern_dir or data_dir
    data_fns = sorted(glob(os.path.join(data_dir, f'*{data_ext}')))
    pattern_fns = sorted(glob(os.path.join(pattern_dir, '*.txt')))
    # comment.txt and pattern.txt (singular - a different, unrelated file
    # that sits alongside the per-angle pattern_*.txt ones) are never
    # themselves per-angle pattern files.
    pattern_fns = [fn for fn in pattern_fns
                   if os.path.basename(fn).lower() not in ('comment.txt', 'pattern.txt')]

    data_groups, data_unparsed = _group_by_angle(data_fns)
    pattern_groups, pattern_unparsed = _group_by_angle(pattern_fns)

    rows = []
    used_patterns = set()
    for angle in sorted(data_groups):
        files = data_groups[angle]  # already sorted - first=detection, second=acquisition
        detection_file = files[0] if len(files) > 0 else None
        acquisition_file = files[1] if len(files) > 1 else None
        extra_files = files[2:]

        # nearest pattern file within tolerance
        pattern_file = None
        best_diff = None
        for p_angle, p_fns in pattern_groups.items():
            diff = abs(p_angle - angle)
            if diff <= angle_tolerance and (best_diff is None or diff < best_diff):
                best_diff = diff
                pattern_file = p_fns[0]
        if pattern_file:
            used_patterns.add(pattern_file)

        status = []
        if detection_file is None:
            status.append(STATUS_MISSING_DETECTION)
        if acquisition_file is None:
            status.append(STATUS_MISSING_ACQUISITION)
        if pattern_file is None:
            status.append(STATUS_MISSING_PATTERN)
        if extra_files:
            status.append(STATUS_EXTRA_FILES)
        if not status:
            status = [STATUS_OK]
        rows.append({
            'angle': angle, 'detection_file': detection_file,
            'acquisition_file': acquisition_file, 'pattern_file': pattern_file,
            'extra_files': extra_files, 'status': status,
            'excluded': status != [STATUS_OK],
        })

    for fn in data_unparsed:
        rows.append({
            'angle': None, 'detection_file': fn, 'acquisition_file': None,
            'pattern_file': None, 'extra_files': [], 'status': [STATUS_ANGLE_UNPARSED],
            'excluded': True,
        })

    unused_pattern_fns = [fn for fns_at_angle in pattern_groups.values()
                          for fn in fns_at_angle if fn not in used_patterns]
    for fn in unused_pattern_fns + pattern_unparsed:
        rows.append({
            'angle': parse_angle(fn), 'detection_file': None, 'acquisition_file': None,
            'pattern_file': fn, 'extra_files': [], 'status': [STATUS_UNMATCHED_PATTERN],
            'excluded': True,
        })

    rows.sort(key=lambda r: (r['angle'] is None, r['angle']))
    return rows


def resolve_smart_scan_files(rows, role='acquisition'):
    """Extract the ordered (angle, data file, pattern file) list for one
    file role, from a match table returned by `match_tilt_files` (after any
    manual review/correction - see SmartScanCheckDialog).

    Args:
        rows: Match-table rows, as returned by `match_tilt_files` (or the
            same shape reloaded from a saved 'smart_scan' metadata.json
            block - see tab_create_navSignal.py's `_save_results_impl`).
        role: 'acquisition' (the smart-scanned file - needs `pattern_file`)
            or 'detection' (the plain dense file - `pattern_file` is
            informational only, not required for reading it).

    Returns:
        List of {'angle', 'file', 'pattern_file'} dicts, sorted by angle -
        rows marked `excluded`, or missing a file for `role`, are skipped.
    """
    key = f'{role}_file'
    out = []
    for row in rows:
        if row.get('excluded'):
            continue
        fn = row.get(key)
        if fn is None:
            continue
        out.append({'angle': row['angle'], 'file': fn, 'pattern_file': row.get('pattern_file')})
    out.sort(key=lambda r: r['angle'])
    return out
