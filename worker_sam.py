# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 22:09:02 2025

@author: sgholam
"""
import os
import sys
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import json
import numpy as np
# torch/sam2 are deliberately NOT bundled in a frozen build (huge, and
# CUDA-version-specific - see INSTALL.md) - guard the import so a missing
# install produces a tagged JSON error on stdout (caught by tab_sam2.py's
# handle_finished_sam()/handle_finished_image_sam()) instead of an
# invisible traceback (this is a windowed app with no console attached).
try:
    import torch
    from sam2.build_sam import build_sam2, build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
except ImportError as _exc:
    print(json.dumps({'error': 'missing_dependency',
                       'message': f'torch/sam2 not installed: {_exc}'}))
    sys.exit(0)
from time import strftime
from glob import glob
import pickle
# from sam2.sam2_video_predictor import SAM2VideoPredictor
#%% funcs
def check_torch_device():
    # check device (cuda or cpu)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    # print(f"using device: {device}")
    
    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        # print(
        #     "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        #     "give numerically different outputs and sometimes degraded performance on MPS. "
        #     "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        # )
        return 0
    return device

def delete_model(args):
    for item in args:
        try:
            del item
        except Exception:
            pass
        
    torch.cuda.empty_cache()
    import gc
    _ = gc.collect()
#%% main
if __name__ == "__main__":
    typ, path_seg_input, idx = sys.argv[1:]
    idx = int(idx)
    
# =============================================================================
#     path_seg_input = r'C:\My Files\OneDrive - Universiteit Antwerpen\GitHub\EDyssey\other_scripts\test_data\EDyssey Analysis\JPG Images\0'
#     idx = 0
#     path_checkpoints = r'C:\My Files\OneDrive - Universiteit Antwerpen\GitHub\EDyssey\EDyssey\tracking_utils\SAM2_checkpoints'
# =============================================================================
    
    # asset_fetch.resolve_sam2_checkpoint_dir() is the single source of
    # truth for where the checkpoint actually is - it may be staged
    # read-only next to the app (offline installer) or downloaded into a
    # writable per-user location (online installer, since the install
    # directory itself isn't guaranteed to be writable - see
    # EDyssey/io_utils/app_dirs.py). Re-deriving this path independently
    # here used to silently disagree with asset_fetch's own resolution.
    from EDyssey.tracking_utils.asset_fetch import (
        resolve_sam2_checkpoint_dir, SAM2_CHECKPOINT_FILENAME)
    sam2_checkpoint = os.path.join(resolve_sam2_checkpoint_dir(), SAM2_CHECKPOINT_FILENAME)
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    if not os.path.isfile(sam2_checkpoint):
        # Normally unreachable - tab_sam2.py downloads this checkpoint (via
        # asset_fetch.ensure_sam2_checkpoint()) before ever launching this
        # worker. Kept as a defensive backstop, using the same tagged-JSON-
        # on-stdout convention as the missing-torch/sam2 case above rather
        # than an uncaught exception, since this is a windowed app with no
        # console to show a traceback in.
        print(json.dumps({'error': 'missing_dependency',
                           'message': 'SAM2 checkpoint not found: ' + sam2_checkpoint}))
        sys.exit(0)
    

    device = check_torch_device()
    if typ == 'image':
        # Single-frame segmentation (no cross-frame propagation): the GUI
        # writes the frame + a {obj_id: {points, labels}} map to
        # seg_input.pkl - every batched object shares this ONE image, so
        # set_image() (the expensive step - image encoding) runs once and
        # predict() is called per object, instead of one full subprocess
        # (with its own image encoding) per object. Callers batch objects
        # that are all applicable to the same frame (see Tab_SAM2's
        # initiate_image_segmentation) - Tab_ROI_on_4D's segment_image()
        # always sends just one object (id 0), since it doesn't track
        # multiple named objects the way the SAM2 tab does.
        with open(os.path.join(path_seg_input, 'seg_input.pkl'), 'rb') as f:
            seg_input = pickle.load(f)
        image = seg_input['image']
        if image.ndim == 2:
            # SAM2ImagePredictor expects an RGB image; the 'video' branch
            # gets this for free since SAM2 loads JPEG frames as RGB, but a
            # single grayscale nav-image frame needs converting explicitly.
            image = np.stack([image] * 3, axis=-1)

        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
        predictor = SAM2ImagePredictor(sam2_model)
        predictor.set_image(image)

        save_kwargs = {}
        for obj_id, obj_data in seg_input['objects'].items():
            points = obj_data['points']
            labels = obj_data['labels']
            # Points are an optional prompt - pass None rather than an
            # empty array when unused, to avoid tripping SAM2's own
            # prompt-shape assertions.
            point_coords = np.array(points, dtype=np.float32) if len(points) else None
            point_labels = np.array(labels, dtype=np.int32) if len(labels) else None
            # All of one object's points/labels are passed together in a
            # single predict() call (they jointly resolve one mask), not one
            # call per point - multimask_output=False returns SAM2's single
            # best-guess mask instead of the 3 ambiguity-resolving candidates.
            masks, scores, logits = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=False,
            )
            save_kwargs[f'obj_{obj_id}'] = masks[0].astype(bool)
            save_kwargs[f'obj_{obj_id}_score'] = np.array(float(scores[0]))

        fn_save = os.path.join(path_seg_input, 'mask_single')
        np.savez_compressed(fn_save, **save_kwargs)

        result = {'path': fn_save + '.npz', 'idx': idx,
                  'obj_ids': [int(o) for o in seg_input['objects'].keys()]}
        print(json.dumps(result))

    elif typ == 'video':
        with open(os.path.join(path_seg_input, 'seg_input.pkl'), 'rb') as f:
            seg_input = pickle.load(f)
        predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
        inference_state = predictor.init_state(seg_input['path_jpg'])
        # Seed every batched object's points against this one encoded video
        # chunk - objects sharing a start/end frame range (and therefore
        # this same JPG export) are batched into one run instead of a
        # separate SAM2 run each (see Tab_SAM2's
        # _group_objects_by_frame_range/initiate_video_segmentation).
        # propagate_in_video() below naturally tracks every seeded obj_id
        # together in one pass.
        for obj_id, obj_data in seg_input['objects'].items():
            frame_idx_arr = obj_data['frame_idx']
            points_arr = obj_data['points']
            labels_arr = obj_data['labels']
            for fr_no in np.unique(frame_idx_arr):
                cond = np.where(frame_idx_arr == fr_no)
                predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=int(fr_no),
                    obj_id=int(obj_id),
                    points=points_arr[cond],
                    labels=labels_arr[cond])

        # propagate in video
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            for i, out_obj_id in enumerate(out_obj_ids):
                mask = (out_mask_logits[i] > 0.0).cpu().numpy()
                video_segments.setdefault(out_obj_id, {})[out_frame_idx] = mask

        # convert each object's {frame: mask} dict to one stacked array
        save_kwargs = {}
        for obj_id, frames_dict in video_segments.items():
            frames = sorted(frames_dict.items())
            w, h = frames[0][1].shape[1:]
            masks = np.zeros((len(frames), w, h), dtype=np.uint8)
            masks[:] = np.array([m for _, m in frames])[:, 0, :, :]
            save_kwargs[f'obj_{obj_id}'] = masks

        fn_save = os.path.join(path_seg_input, 'masks')
        np.savez_compressed(fn_save, **save_kwargs)

        result = {'path': fn_save+'.npz',
                  'idx': idx}
        # delete_model([predictor, build_sam2, build_sam2_video_predictor, SAM2ImagePredictor,
        #              out_mask_logits, video_segments, torch])
        print(json.dumps(result))

    elif typ == 'auto':
        # SAM2's automatic mask generator - finds every candidate object on
        # one frame with no point/box prompts at all (a grid of points
        # sampled over the whole image instead), for Tab_SAM2's Auto
        # Detector widget (ui_tabs/sam2_auto_detector_widget.py). Unlike
        # 'image'/'video', this isn't seeded per-object - candidates are
        # found first, then the widget lets the user pick which ones become
        # tracked objects.
        with open(os.path.join(path_seg_input, 'seg_input.pkl'), 'rb') as f:
            seg_input = pickle.load(f)
        image = seg_input['image']
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        params = seg_input.get('params', {})

        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
        generator = SAM2AutomaticMaskGenerator(sam2_model, **params)
        candidates = generator.generate(image)

        if candidates:
            masks = np.stack([c['segmentation'].astype(bool) for c in candidates])
        else:
            masks = np.zeros((0,) + image.shape[:2], dtype=bool)
        fn_masks = os.path.join(path_seg_input, 'auto_masks')
        np.savez_compressed(fn_masks, masks=masks)

        # Metadata (everything but the mask itself, already in the npz
        # above) as a plain JSON list, index-aligned with `masks`.
        meta = [{'area': int(c['area']), 'bbox': [float(v) for v in c['bbox']],
                 'predicted_iou': float(c['predicted_iou']),
                 'stability_score': float(c['stability_score'])} for c in candidates]
        fn_meta = os.path.join(path_seg_input, 'auto_meta.json')
        with open(fn_meta, 'w') as f:
            json.dump(meta, f)

        result = {'path': fn_masks + '.npz', 'meta_path': fn_meta, 'idx': idx}
        print(json.dumps(result))