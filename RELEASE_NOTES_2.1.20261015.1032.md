# EDyssey 2.1.20261015.1032

## Fixes

- **Installed app: tpx3 DP extraction with pyeventem appeared to do nothing, with no error**: pyeventem's compiled-kernel cache (numba) defaults to a folder inside the install directory, which is read-only under Program Files. On Windows, Python's temp-file helper retries forever when asked to create a file in such a folder, so the first `import pyeventem` of every session never returned, and every Summed DP, ROI and mask extraction waited behind it. The installed app now keeps that cache in `%LOCALAPPDATA%\EDyssey\numba_cache`. The first pyeventem call after installing compiles the kernels (about 5 to 10 seconds longer); later ones reuse the cache.

---
*Full changelog since `2.1.20261015.0927`.*
