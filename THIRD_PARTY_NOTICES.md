# Third-Party Notices

EDyssey itself is licensed under the GNU General Public License v3.0 (see
[LICENSE](LICENSE)). It also uses model weights from third parties, some
downloaded on demand rather than bundled in the installer (see
[EDyssey/tracking_utils/asset_fetch.py](EDyssey/tracking_utils/asset_fetch.py)) -
this file documents what each one is, its license, and where it comes from.

## SAM2 checkpoint (`sam2.1_hiera_large.pt`)

- **What**: Meta's Segment Anything Model 2 (SAM 2.1, Hiera-Large variant),
  used by the SAM2 Seg. tab.
- **Source**: `https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt`
  (Meta's own official download host - the same one
  [facebookresearch/sam2](https://github.com/facebookresearch/sam2)'s
  `checkpoints/download_ckpts.sh` uses).
- **License**: Apache License 2.0 (covers the whole `facebookresearch/sam2`
  repository, including the published checkpoints - no separate weights
  license exists).

## OpenCV DaSiamRPN tracker (`dasiamrpn_model.onnx`, `dasiamrpn_kernel_cls1.onnx`, `dasiamrpn_kernel_r1.onnx`)

- **What**: The DaSiamRPN visual tracker's ONNX weights, used by the
  "dasiamrpn" option in the Tracking by CV2 tab.
- **Source**: [opencv/opencv_zoo](https://github.com/opencv/opencv_zoo),
  pinned to commit `fef72f8fa7c52eaf116d3df358d24e6e959ada0e` - the last
  commit before that project removed DaSiamRPN in favor of its
  [VitTrack](https://github.com/opencv/opencv_zoo/tree/main/models/object_tracking_vittrack)
  model ("Remove DaSiamRPN since we have its superseder VitTrack now
  (opencv/opencv_zoo#213)"). The pinned commit URL stays valid even though
  the folder no longer exists on the default branch.
- **License**: Apache License 2.0 (`models/object_tracking_dasiamrpn/LICENSE`
  at that commit). The original research code
  ([foolwood/DaSiamRPN](https://github.com/foolwood/DaSiamRPN)) is
  separately MIT-licensed (Copyright (c) 2018 Qiang Wang).
- **Note**: opencv_zoo's own removal of this model in favor of VitTrack is
  worth taking as a signal - VitTrack is the actively maintained option
  upstream now. Migrating the "dasiamrpn" tracker option to VitTrack would
  be a reasonable follow-up, but is a behavioral change (different
  `cv2.Tracker*_Params`/API shape), not part of this packaging pass.

## OpenCV NanoTrack tracker (`backbone.onnx`, `neckhead.onnx`)

- **What**: The NanoTrack visual tracker's ONNX weights, used by the
  "nano" option in the Tracking by CV2 tab.
- **Source**: [HonglinChu/SiamTrackers](https://github.com/HonglinChu/SiamTrackers)
  (`NanoTrack/models/nanotrackv2/`) - this is the source OpenCV's own
  official sample (`opencv/samples/python/tracker.py`) points at for these
  files; they were never hosted in opencv_zoo.
- **License: UNCLEAR.** This repository has **no LICENSE file at all**
  (confirmed via the GitHub API and a full repo-root listing), so there is
  no explicit grant of redistribution rights - under default copyright
  rules that makes it "all rights reserved" by the author. This is a real
  gap, not a cleared dependency like the two entries above.
  **Before redistributing this to anyone else, either get explicit
  permission from the author, find an alternately-licensed source for the
  same weights, or drop the "nano" tracker option.** It's wired up in
  `asset_fetch.py` because the app's existing "nano" option needs it to
  function, not because the licensing question has been resolved.

## Not shipped / not downloaded

Two files exist on developer machines but are **not** distributed by the
installer or `asset_fetch.py` - both are dead code (nothing in the app
imports or calls them):

- `EDyssey/tracking_utils/opencv_models/goturn.caffemodel` /
  `goturn.prototxt` - no `cv2.TrackerGOTURN_create()` call exists anywhere
  in this codebase.
- `EDyssey/tracking_utils/SAM2_checkpoints/sam2.1_hiera_{tiny,small,base_plus}.pt` -
  only the `large` checkpoint is ever loaded (`worker_sam.py`).

## Runtime dependencies

Installed via `requirements.txt`/`pip` in every case; additionally bundled
directly inside the PyInstaller build itself for the parts noted below (see
[EDyssey.spec](EDyssey.spec)).

Notably GPL-3.0: **PyQt5** (or a commercial Riverbank license) and
**HyperSpy** - these are why EDyssey itself is GPL-3.0 rather than a more
permissive license (GPL forbids adding further restrictions on top of a
combined work). Both are bundled into every installer build (online and
offline).

Everything else (numpy, dask, opencv-contrib-python, matplotlib, tifffile,
h5py, tqdm, scipy, scikit-image, pandas, Pillow) is permissively licensed
(BSD/MIT/Apache-2.0 family) and bundled into every installer build too.

**`torch`, `torchvision`, the `sam2` package, and their own dependencies
(`hydra-core`, `iopath`, `omegaconf`)** - also permissively licensed
(BSD-3-Clause for torch/torchvision, Apache-2.0 for sam2, MIT/BSD for the
rest) - are bundled directly (including CUDA runtime DLLs) **only in the
offline installer** (`EDYSSEY_OFFLINE_BUILD=1` in EDyssey.spec). The online
installer and a from-source install instead expect these installed
separately (see INSTALL.md's "Enabling SAM2") since the right `torch` build
is CUDA-version-specific.

See each package's own PyPI page for its exact license.
