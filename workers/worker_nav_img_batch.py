# -*- coding: utf-8 -*-
"""Batch entry point for Navigator's "Calculate All" - one subprocess, one
internal process pool, instead of one QProcess per file (see
worker_nav_img.py for the original single-file worker, still used
directly by ROI on 4D/SAM2's one-off "Compute Virtual Image", which has
no benefit from pooling a single task).

Invoked as `--worker nav_img_batch <tasks.json path> <temp_dir>
<n_workers>` via worker_launch.worker_command()/worker_dispatch.py, the
same indirection every other worker script here uses - see
worker_pool_utils.py for the shared pool-driving/IPC-protocol machinery
this just plugs calculate_nav_img_core into.
"""
import sys
import worker_pool_utils as wpu
from worker_nav_img import calculate_nav_img_core


def run_batch(tasks_path, temp_dir, n_workers):
    # Block until the GUI confirms this process has been assigned to its
    # Windows Job Object, before spawning any pool workers of our own - see
    # wait_for_job_assignment's docstring for why this ordering matters.
    wpu.wait_for_job_assignment()
    tasks = wpu.load_tasks(tasks_path)
    wpu.run_pool_batch(calculate_nav_img_core, tasks, temp_dir, int(n_workers))


if __name__ == '__main__':
    run_batch(*sys.argv[1:])
