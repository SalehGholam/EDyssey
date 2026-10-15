# -*- coding: utf-8 -*-
"""Every script `--worker <name>` can dispatch to must be bundled by EDyssey.spec as a loose file,
or the installed app fails with "can't find '__main__' module" (worker_eventem_call.py was missing)."""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import worker_dispatch  # noqa: E402


def test_every_dispatchable_worker_is_bundled_by_the_spec():
    with open(os.path.join(ROOT, 'EDyssey.spec'), encoding='utf-8') as f:
        bundled = set(re.findall(r"\('workers/(worker_\w+\.py)',\s*'EDyssey/workers'\)", f.read()))
    missing = sorted(s for s in worker_dispatch.WORKER_SCRIPTS.values() if s not in bundled)
    assert not missing, f'not bundled by EDyssey.spec: {missing}'


def test_dispatchable_workers_exist_on_disk():
    for script in worker_dispatch.WORKER_SCRIPTS.values():
        assert os.path.isfile(os.path.join(ROOT, 'workers', script)), script
