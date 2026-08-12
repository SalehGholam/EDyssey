; Online installer variant: small, downloads the SAM2 checkpoint and
; Nano/DaSiamRPN tracker model weights on first use of those features
; instead of bundling them (see EDyssey/tracking_utils/asset_fetch.py and
; EDyssey.spec's header comment for why) - needs internet the first time
; you use SAM2 Seg. or the nano/dasiamrpn trackers. This is the variant
; built automatically by .github/workflows/build-installer.yml.
;
; Build the PyInstaller output first (pyinstaller EDyssey.spec from the
; repo root), then compile this with: iscc installer\EDyssey_online.iss
#define Variant "Online"
#include "EDyssey_common.iss"
