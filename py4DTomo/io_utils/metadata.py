# -*- coding: utf-8 -*-
"""tpx3 acquisition metadata (comment.txt) parsing, and the navigator tab's
own saved-analysis metadata.json - both are read by CV2/SAM2 to auto-fill
scan parameters from what the navigator tab (or the raw acquisition)
already recorded.
"""
import os
import json

def _resolve_metadata_fn(path_main):
    """`path_main` is either a comment.txt path itself, or a directory/file
    next to which a comment.txt lives."""
    if os.path.isfile(path_main):
        return path_main if path_main.endswith('.txt') else os.path.join(
            os.path.dirname(path_main), 'comment.txt')
    return os.path.join(path_main, 'comment.txt')

def get_metadata(path_main, count=0):
    """Parse the (0-indexed) `count`-th measurement block from a tpx3
    acquisition's comment.txt. `path_main` may be the comment.txt file
    itself, or the directory/4D-signal-file next to which it sits. Blocks
    are separated by a blank line, or by two consecutive blank lines when
    the same block repeats back-to-back."""
    with open(_resolve_metadata_fn(path_main), 'r') as f:
        ls = f.readlines()
    metadata = {}
    blank = [i for i, x in enumerate(ls) if x == '\n']
    end = blank[count]
    while (count != 0) and (end == blank[count-1]+1):  # 2 empty lines
        end = blank[count+1]
    for item in ls[:end]:
        try:
            key, val = item.split(':')
            if ' microseconds' in val:
                val = val.rstrip(' microseconds\n')
            metadata[key] = float(val)
        except Exception:
            pass
    return metadata

def get_metadata_block_count(path_main):
    """Number of measurement blocks logged in a tpx3 comment.txt (see
    `get_metadata`), i.e. how many separate acquisitions it records - blocks
    are separated by one or more consecutive blank lines."""
    with open(_resolve_metadata_fn(path_main), 'r') as f:
        ls = f.readlines()
    n_blocks = 0
    in_block = False
    for line in ls:
        if line.strip():
            if not in_block:
                n_blocks += 1
                in_block = True
        else:
            in_block = False
    return n_blocks

def default_analysis_save_dir(fn_nav):
    """Best-effort default "Save Dir." for a tab that loads an existing
    navigation signal file (CV2/SAM2's "Nav. Signal" browse button): if
    `fn_nav` sits inside a "<project>_5DED Analysis/navigator signal/
    <timestamp>/" folder - the structure the navigator tab's own save
    produces - reuse that same "<project>_5DED Analysis" folder, so this
    tab's own results land in the same per-project tree instead of a
    disconnected new one next to the file. Otherwise falls back to a
    generic "5DED Analysis" folder next to the file, as before."""
    d = os.path.dirname(fn_nav)            # .../<timestamp>
    parent = os.path.dirname(d)            # .../navigator signal
    grandparent = os.path.dirname(parent)  # .../<project>_5DED Analysis
    if os.path.basename(parent) == 'navigator signal' and grandparent:
        return grandparent
    return os.path.join(d, '5DED Analysis')

def save_analysis_info(path_save, fn_nav_source, analysis_type=None):
    """Record the source path of the navigation signal used for a save (see
    Tab_SAM2/Tab_Tracking_CV2's _save_results_impl) in a small
    analysis_info.json at the top of `path_save`, instead of copying the
    (potentially large) signal itself into every saved-analysis folder.
    `load_analysis_info` reads this back so "Load Saved Analysis" can reload
    the original file. No-op if `fn_nav_source` is falsy (nothing was ever
    loaded this session).

    `analysis_type` ('sam2' or 'cv2') tags which tab produced this save, so
    the other tab's "Load Saved Analysis" can detect a mismatch (its folder
    layout - roi No N.json columns, mask/rois file names - is tab-specific
    and doesn't load correctly in the other tab) and warn instead of failing
    confusingly partway through."""
    if not fn_nav_source:
        return
    fn_info = os.path.join(path_save, 'analysis_info.json')
    with open(fn_info, 'w') as f:
        json.dump({'nav_signal_source': fn_nav_source, 'analysis_type': analysis_type},
                   f, indent=4)

def load_analysis_info(path_save):
    """Load the analysis_info.json written by `save_analysis_info` next to a
    saved-analysis folder, if present.

    Returns:
        The parsed dict, or None if no analysis_info.json exists in
        `path_save`, or it can't be read/parsed.
    """
    fn_info = os.path.join(path_save, 'analysis_info.json')
    if not os.path.isfile(fn_info):
        return None
    try:
        with open(fn_info, 'r') as f:
            return json.load(f)
    except Exception:
        return None

def load_analysis_metadata(fn_nav):
    """Load the sibling metadata.json (written by the navigator tab's own
    save - see Tab_Create_NavSignal._save_results_impl) next to a
    navigation signal file, if present.

    Returns:
        The parsed dict, or None if no metadata.json exists next to
        `fn_nav`, or it can't be read/parsed.
    """
    fn_meta = os.path.join(os.path.dirname(fn_nav), 'metadata.json')
    if not os.path.isfile(fn_meta):
        return None
    try:
        with open(fn_meta, 'r') as f:
            return json.load(f)
    except Exception:
        return None
