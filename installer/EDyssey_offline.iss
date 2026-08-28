; Offline installer variant: bundles EVERYTHING needed for the app to work
; with zero internet access, ever - not just on first use:
;   - torch + CUDA + torchvision + sam2 (+ hydra-core/iopath/omegaconf),
;     via EDyssey.spec's EDYSSEY_OFFLINE_BUILD=1 mode (see its header
;     comment) - the SAM2 tab works out of the box instead of needing the
;     manual `pip install --target ...` step INSTALL.md documents for the
;     online variant.
;   - The SAM2 checkpoint and Nano/DaSiamRPN tracker model weights
;     (~1.06GB), staged directly below. No code changes were needed for
;     this part: asset_fetch.py's ensure_*() functions already skip
;     downloading anything already present at the right size, so staging
;     the files here is all it takes.
;
; Build steps:
;   1. set EDYSSEY_OFFLINE_BUILD=1                    (cmd)
;      pyinstaller EDyssey.spec --distpath dist_offline --workpath build_offline
;   2. iscc installer\EDyssey_offline.iss
;
; Step 1 needs the model weight files present on YOUR machine first - they're
; git-ignored (not tracked in this repo, see THIRD_PARTY_NOTICES.md for
; where they come from and their licenses). If you don't have them yet,
; either run an online-build install once with internet access and use
; every SAM2/nano/dasiamrpn feature once (asset_fetch.py downloads them
; for you into the exact paths below), or fetch them directly via the same
; URLs asset_fetch.py uses:
;   EDyssey\tracking_utils\SAM2_checkpoints\sam2.1_hiera_large.pt
;   EDyssey\tracking_utils\opencv_models\backbone.onnx
;   EDyssey\tracking_utils\opencv_models\neckhead.onnx
;   EDyssey\tracking_utils\opencv_models\dasiamrpn_model.onnx
;   EDyssey\tracking_utils\opencv_models\dasiamrpn_kernel_cls1.onnx
;   EDyssey\tracking_utils\opencv_models\dasiamrpn_kernel_r1.onnx
;
; This installer lands at several GB (torch+CUDA alone is multi-GB) -
; deliberately NOT built by .github/workflows/build-installer.yml or
; published as a GitHub Release (GitHub caps individual release assets at
; 2GB). Build it locally on a machine that already has torch/CUDA and the
; model files installed, and distribute it yourself (internal file share,
; USB drive, direct link, etc.) rather than via GitHub.
#define Variant "Offline"
#define DistDir "..\dist_offline\EDyssey"
#include "EDyssey_common.iss"

[Files]
Source: "..\EDyssey\tracking_utils\SAM2_checkpoints\sam2.1_hiera_large.pt"; DestDir: "{app}\EDyssey\tracking_utils\SAM2_checkpoints"; Flags: ignoreversion
Source: "..\EDyssey\tracking_utils\opencv_models\backbone.onnx"; DestDir: "{app}\EDyssey\tracking_utils\opencv_models"; Flags: ignoreversion
Source: "..\EDyssey\tracking_utils\opencv_models\neckhead.onnx"; DestDir: "{app}\EDyssey\tracking_utils\opencv_models"; Flags: ignoreversion
Source: "..\EDyssey\tracking_utils\opencv_models\dasiamrpn_model.onnx"; DestDir: "{app}\EDyssey\tracking_utils\opencv_models"; Flags: ignoreversion
Source: "..\EDyssey\tracking_utils\opencv_models\dasiamrpn_kernel_cls1.onnx"; DestDir: "{app}\EDyssey\tracking_utils\opencv_models"; Flags: ignoreversion
Source: "..\EDyssey\tracking_utils\opencv_models\dasiamrpn_kernel_r1.onnx"; DestDir: "{app}\EDyssey\tracking_utils\opencv_models"; Flags: ignoreversion
