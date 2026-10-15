# EDyssey 2.1.20261015.0115

## Fixes

- **tpx3 calls delegated to a subprocess failed in the installed app with `can't find '__main__' module`**: `worker_eventem_call.py` was not bundled by the installer. It is used whenever a tpx3 computation has to run in a separate process: always for New eventem, and for pyeventem or Old eventem after the other has already run in the same session. It is now bundled, and a test checks that every worker the app can dispatch to is listed in the build spec.

---
*Full changelog since `2.1.20261014.2054`.*
