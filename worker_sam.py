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
except ImportError as _exc:
    print(json.dumps({'error': 'missing_dependency',
                       'message': f'torch/sam2 not installed: {_exc}'}))
    sys.exit(0)
from time import strftime
import pandas as pd
from glob import glob
import pickle
import matplotlib.pyplot as plt
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
    
    path_file = os.path.abspath(__file__)
    path_file = os.path.dirname(path_file)
    path_checkpoints = os.path.join(path_file, 'EDyssey', 'tracking_utils', 'SAM2_checkpoints')
    fn_checkpoint = 'sam2.1_hiera_large.pt'
    sam2_checkpoint = os.path.join(path_checkpoints, fn_checkpoint)
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
        # writes the frame + its point prompts for this tab.SAM2 tab's
        # currently-selected object/frame to seg_input.pkl, mirroring the
        # 'video' branch's input convention below.
        df = pd.read_pickle(os.path.join(path_seg_input, 'seg_input.pkl'))
        image = df['image']
        if image.ndim == 2:
            # SAM2ImagePredictor expects an RGB image; the 'video' branch
            # gets this for free since SAM2 loads JPEG frames as RGB, but a
            # single grayscale nav-image frame needs converting explicitly.
            image = np.stack([image] * 3, axis=-1)
        points = df['points']
        labels = df['labels']
        # Points and box are both optional prompts - SAM2 accepts either
        # alone or combined (a box narrows the search region, points refine
        # it further); pass None rather than an empty array when unused, to
        # avoid tripping SAM2's own prompt-shape assertions.
        point_coords = np.array(points, dtype=np.float32) if len(points) else None
        point_labels = np.array(labels, dtype=np.int32) if len(labels) else None
        box = df.get('box')
        if box is not None:
            box = np.array(box, dtype=np.float32)

        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
        predictor = SAM2ImagePredictor(sam2_model)
        predictor.set_image(image)
        # All points/labels (and the box, if any) are passed together in a
        # single predict() call (they jointly resolve one mask), not one
        # call per point - multimask_output=False returns SAM2's single
        # best-guess mask instead of the 3 ambiguity-resolving candidates.
        masks, scores, logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=False,
        )
        mask = masks[0].astype(bool)

        fn_save = os.path.join(path_seg_input, 'mask_single')
        np.savez_compressed(fn_save, mask=mask, score=float(scores[0]))

        result = {'path': fn_save + '.npz', 'idx': idx}
        print(json.dumps(result))

    elif typ == 'video':
        df = pd.read_pickle(os.path.join(path_seg_input, 'seg_input.pkl'))
        predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
        inference_state = predictor.init_state(df['path_jpg'])
        # add points
        for _, fr_no in enumerate(np.unique(df.frame_idx)): # only one object is inside df
            points = df.points[np.where(df.frame_idx == fr_no)]
            labels = df.labels[np.where(df.frame_idx == fr_no)]
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=fr_no,
                obj_id=0,
                points=points,
                labels=labels)

# several objects
# =============================================================================
#         for obj_id in df.index:
#             for i, fr_no in enumerate(df.loc[obj_id, 'frame_idx']):
#                 _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
#                     inference_state=inference_state,
#                     frame_idx=fr_no,
#                     obj_id=obj_id,
#                     points=df.loc[obj_id, 'points'][i],
#                     labels=df.loc[obj_id, 'labels'][i])
# =============================================================================
        
        # propagate in video 
        video_segments = {}
        
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            for i, out_obj_id in enumerate(out_obj_ids):
                mask = (out_mask_logits[i] > 0.0).cpu().numpy()
                fig, ax = plt.subplots()
                ax.imshow(mask[0])
                if out_obj_id not in video_segments:
                    video_segments[out_obj_id] = {}  # Temporary: frame-to-mask map
        
                video_segments[out_obj_id][out_frame_idx] = mask
        
# =============================================================================
#         # debugging
#         p_d = r'C:\My Files\OneDrive - Universiteit Antwerpen\GitHub\EDyssey\other_scripts\test_data\EDyssey Analysis'
#         f_d = os.path.join(p_d, "video_segments.pkl")
#         with open(f_d, "wb") as f:
#             pickle.dump(video_segments, f)
# =============================================================================
        
        # convert dicts to arrays
        w, h = video_segments[out_obj_ids[0]][0].shape[1:]
        masks = np.zeros((len(video_segments[out_obj_ids[0]]), w, h), dtype=np.uint8)
        
        frames = sorted(video_segments[0].items())
        masks[:] = np.array([m for _, m in frames])[:,0,:,:]
        
        fn_save = os.path.join(path_seg_input, 'masks')
        np.savez_compressed(fn_save, masks=masks)
        
        result = {'path': fn_save+'.npz',
                  'idx': idx}
        # delete_model([predictor, build_sam2, build_sam2_video_predictor, SAM2ImagePredictor,
        #              out_mask_logits, video_segments, torch])
        print(json.dumps(result))