# -*- coding: utf-8 -*-
"""Batch entry point for ROI Tracker/SAM2's "Extract!" (3DED extraction) -
one subprocess, one internal process pool, per tracked/segmented object,
instead of one QProcess per (ROI, frame) task (see worker_extract_frame.py
for the original single-task worker, still available as a standalone
entry point though no longer called by those tabs).

tab_tracking_cv2.py/tab_sam2.py launch one of these per enabled object,
strictly one at a time (the next object's batch only starts once the
current one fully finishes) - see their _launch_next_object_batch.

Invoked as `--worker extract_frame_batch <tasks.json path> <temp_dir>
<n_workers>` via worker_launch.worker_command()/worker_dispatch.py, the
same indirection every other worker script here uses - see
worker_pool_utils.py for the shared pool-driving/IPC-protocol machinery
this just plugs extract_3ded_mask_core into.
"""
import sys
import worker_pool_utils as wpu
from worker_extract_frame import extract_3ded_mask_core


def run_batch(tasks_path, temp_dir, n_workers):
    # Block until the GUI confirms this process has been assigned to its
    # Windows Job Object, before spawning any pool workers of our own - see
    # wait_for_job_assignment's docstring for why this ordering matters.
    wpu.wait_for_job_assignment()
    tasks = wpu.load_tasks(tasks_path)
    wpu.run_pool_batch(extract_3ded_mask_core, tasks, temp_dir, int(n_workers))


if __name__ == '__main__':
    run_batch(*sys.argv[1:])
