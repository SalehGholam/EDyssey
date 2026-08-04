# -*- coding: utf-8 -*-
"""Smart-scan (sparse/pattern-file) acquisition support: matching a folder's
detection/acquisition data files to their pattern files and to each other by
tilt angle, and flagging mismatches for review.

See "other_scripts/smart scanning guide/" for the exploratory reference
scripts this generalizes, and the "Smart Scanned" checkbox present in every
tab's Scan Size box for where this is used.

A tomography smart-scan run always writes, per tilt angle, an "acquisition"
data file - sparse, needs a pattern file to reshape - plus one per-angle
pattern .txt file, alongside one shared comment.txt (see
metadata.get_metadata/get_metadata_block_count) and one shared pattern.txt
(an unrelated file, always skipped here). It *may* also write a "detection"
image - a normal dense reference raster, doesn't need a pattern file - but
where that lives differs by format:

- .tpx3: detection is a second .tpx3 file at the same angle (eventem's own
  reader handles both roles identically) - two same-extension files per
  angle is the norm, so a missing partner is treated as a real problem (see
  `detection_required`).
- .mib/.hspy/.zspy: there is normally only *one* data file per angle (the
  acquisition) - detection, when present at all, is a separate .tif/.tiff
  image, not a second file of the same format. Missing detection is the
  *expected* case here, not a problem - plenty of real acquisitions never
  wrote one. (Occasionally a dataset still does write two same-extension
  files per angle like .tpx3 - `match_tilt_files` falls back to that
  automatically whenever it finds one.)

File naming isn't fully standardized between datasets/instruments -
`parse_angle` is deliberately a best-effort heuristic rather than tied to
one exact scheme, which is why every call site of `match_tilt_files` is
expected to let the user review and correct its output (see
ui_tabs/smart_scan_dialog.py's SmartScanCheckDialog) rather than trust it
blindly.

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

# Extensions searched for a separate detection image when a data-file angle
# group has only one file - see module docstring.
DETECTION_EXTENSIONS = ('.tif', '.tiff')

# Row 'status' values match_tilt_files can report (a row may have several).
STATUS_OK = 'ok'
STATUS_MISSING_DETECTION = 'missing_detection'
STATUS_MISSING_ACQUISITION = 'missing_acquisition'
STATUS_MISSING_PATTERN = 'missing_pattern'
STATUS_EXTRA_FILES = 'extra_files'
STATUS_ANGLE_UNPARSED = 'angle_unparsed'
STATUS_UNMATCHED_PATTERN = 'unmatched_pattern'
STATUS_UNMATCHED_DETECTION = 'unmatched_detection'

STATUS_LABELS = {
    STATUS_OK: 'OK',
    STATUS_MISSING_DETECTION: 'Missing detection file',
    STATUS_MISSING_ACQUISITION: 'Missing acquisition file',
    STATUS_MISSING_PATTERN: 'Missing pattern file',
    STATUS_EXTRA_FILES: 'Extra file(s) at this angle',
    STATUS_ANGLE_UNPARSED: "Couldn't parse a tilt angle",
    STATUS_UNMATCHED_PATTERN: 'Pattern file matches no data file',
    STATUS_UNMATCHED_DETECTION: 'Detection image matches no data file',
}


def detection_required(data_ext):
    """Whether a missing detection file should count as a match problem
    (excluding the row) for `data_ext`.

    True only for .tpx3: detection and acquisition are the same
    two-same-extension-file group there, so a missing partner is genuinely
    ambiguous (which role does the lone file play?). For .mib/.hspy/.zspy,
    a missing detection file is the *normal* case (see module docstring)
    and never excludes a row.
    """
    return data_ext == '.tpx3'


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


def _nearest_at_angle(groups, angle, angle_tolerance, pick=None):
    """File from whichever group in `groups` (an {angle: [fn, ...]} dict) is
    nearest `angle`, within `angle_tolerance` - or None.

    `pick`, if given, chooses which file to use out of a matched angle's
    (already-sorted) file list - defaults to the first. See
    `_pick_non_acquisition_haadf` for the one caller that needs this.
    """
    pick = pick or (lambda fns: fns[0])
    best_fn, best_diff = None, None
    for g_angle, fns in groups.items():
        diff = abs(g_angle - angle)
        if diff <= angle_tolerance and (best_diff is None or diff < best_diff):
            best_diff = diff
            best_fn = pick(fns)
    return best_fn


def _pick_non_acquisition_haadf(fns):
    """Prefer a HAADF/reference image that ISN'T the acquisition-phase one
    when more than one exists at the same angle - by this app's naming
    convention, e.g. "scan_image_0000_-45.00.tiff" (detection phase) and
    "scan_image_0000_-45.00_2.tiff" (acquisition phase, suffixed "_2"
    before the extension) - "detection" here specifically wants the
    former. Falls back to the first (sorted) file if that convention
    doesn't clearly identify one (e.g. only one file exists, or neither/
    both are "_2"-suffixed)."""
    non_acq = [fn for fn in fns if not os.path.splitext(fn)[0].endswith('_2')]
    return non_acq[0] if non_acq else fns[0]


def _dirs_differ(a, b):
    """True if `a` and `b` refer to different directories, comparing
    case- and separator-normalized paths."""
    return os.path.normcase(os.path.normpath(a)) != os.path.normcase(os.path.normpath(b))


def match_tilt_files(data_dir, data_ext, pattern_dir=None, angle_tolerance=0.05,
                     detection_dir=None):
    """Match a folder's smart-scan data files (detection + acquisition, by
    role) to their pattern files, by tilt angle.

    Args:
        data_dir: Folder containing the acquisition data files (and, unless
            `detection_dir` points elsewhere, the detection ones too - see
            below).
        data_ext: Extension to match - one of DATA_EXTENSIONS.
        pattern_dir: Folder containing the per-angle pattern .txt files -
            defaults to `data_dir` (pattern files usually sit alongside the
            data, but a cleaner acquisition setup may keep them in their own
            folder instead).
        angle_tolerance: Max absolute angle difference (degrees) for a
            pattern/detection file to be considered a match for a data-file
            angle group - accounts for angles occasionally being logged at
            slightly different precision between file types.
        detection_dir: Folder to search for detection files - defaults to
            `data_dir` (the common case: detection and acquisition files
            live side by side). Giving a *different* folder here means
            "detection lives somewhere else, don't expect to find it mixed
            in with the acquisition files in data_dir" - for .mib/.hspy/
            .zspy this only changes where DETECTION_EXTENSIONS (.tif/.tiff)
            are searched for; for .tpx3 it additionally switches off the
            same-folder "two files per angle, first=detection" grouping in
            favour of matching two independent, single-file-per-angle
            folders by nearest angle instead (see module docstring).

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
    # A separate detection_dir is a deliberate "detection lives elsewhere,
    # don't look for it co-located with the acquisition files" signal from
    # the caller - see the combined_mode explanation below.
    separate_detection_dir = detection_dir is not None and _dirs_differ(detection_dir, data_dir)
    detection_dir = detection_dir or data_dir
    data_fns = sorted(glob(os.path.join(data_dir, f'*{data_ext}')))
    pattern_fns = sorted(glob(os.path.join(pattern_dir, '*.txt')))
    # comment.txt and pattern.txt (singular - a different, unrelated file
    # that sits alongside the per-angle pattern_*.txt ones) are never
    # themselves per-angle pattern files.
    pattern_fns = [fn for fn in pattern_fns
                   if os.path.basename(fn).lower() not in ('comment.txt', 'pattern.txt')]

    data_groups, data_unparsed = _group_by_angle(data_fns)
    pattern_groups, pattern_unparsed = _group_by_angle(pattern_fns)

    tpx3_style = (data_ext == '.tpx3')
    # combined_mode: detection may be co-located with acquisition in
    # data_dir, as a second same-extension file sharing its angle (the
    # historical default assumption, still the norm for .tpx3) - true
    # unless the caller explicitly pointed detection_dir somewhere else,
    # in which case data_dir is trusted to hold acquisition files only and
    # detection is looked up exclusively from detection_dir instead.
    combined_mode = not separate_detection_dir
    is_haadf_detection = not tpx3_style  # .tif/.tiff, vs. a same-ext .tpx3 in its own folder
    detection_groups = {}
    if not combined_mode or is_haadf_detection:
        detection_exts = DETECTION_EXTENSIONS if is_haadf_detection else (data_ext,)
        detection_fns = []
        for ext in detection_exts:
            detection_fns += glob(os.path.join(detection_dir, f'*{ext}'))
        detection_groups, _ = _group_by_angle(sorted(detection_fns))
    detection_pick = _pick_non_acquisition_haadf if is_haadf_detection else None

    rows = []
    used_patterns = set()
    used_detections = set()
    for angle in sorted(data_groups):
        files = data_groups[angle]  # already sorted
        if combined_mode and (tpx3_style or len(files) >= 2):
            # Two-same-extension-file convention: first=detection,
            # second=acquisition - the norm for .tpx3, and occasionally
            # also true for .mib/.hspy/.zspy (see module docstring).
            detection_file = files[0] if len(files) > 0 else None
            acquisition_file = files[1] if len(files) > 1 else None
            extra_files = files[2:]
        else:
            # Either the normal case for .mib/.hspy/.zspy (the one data
            # file at this angle IS the acquisition - detection, if any, is
            # a separate HAADF image) or an explicit separate-folders setup
            # (detection_dir given and different from data_dir) - either
            # way, data_dir is acquisition-only here and detection comes
            # from detection_groups (a different folder and/or extension).
            detection_file = _nearest_at_angle(detection_groups, angle, angle_tolerance,
                                               pick=detection_pick)
            acquisition_file = files[0] if files else None
            extra_files = files[1:]
            if detection_file:
                used_detections.add(detection_file)

        pattern_file = _nearest_at_angle(pattern_groups, angle, angle_tolerance)
        if pattern_file:
            used_patterns.add(pattern_file)

        status = []
        if detection_file is None and detection_required(data_ext):
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
            'angle': None, 'detection_file': None, 'acquisition_file': fn,
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

    if detection_groups:
        # Informational only (not excluded, doesn't block anything) - just
        # surfaced so a stray/misdated detection image doesn't go unnoticed.
        unused_detection_fns = [fn for fns_at_angle in detection_groups.values()
                                for fn in fns_at_angle if fn not in used_detections]
        for fn in unused_detection_fns:
            rows.append({
                'angle': parse_angle(fn), 'detection_file': fn, 'acquisition_file': None,
                'pattern_file': None, 'extra_files': [], 'status': [STATUS_UNMATCHED_DETECTION],
                'excluded': False,
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
