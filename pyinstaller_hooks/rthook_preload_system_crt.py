"""Runtime hook, run before PyInstaller's own PyQt5 rthook (see EDyssey.spec's
runtime_hooks= - custom hooks always execute first).

PyInstaller's pyi_rth_pyqt5.py unconditionally imports PyQt5.QtCore at
startup (even for a --worker subprocess that never touches Qt), which loads
PyQt5's own bundled Qt5\\bin\\msvcp140.dll. Windows then reuses that same
loaded module for any later DLL that also depends on msvcp140.dll -
including torch's c10.dll, which crashes when initialized against Qt's
copy instead of the real system one. Preloading the system copy here first
makes Windows reuse *that* instead.
"""
import ctypes
import os

_system32 = os.path.join(os.environ.get('SystemRoot', r'C:\Windows'), 'System32')
try:
    ctypes.WinDLL(os.path.join(_system32, 'msvcp140.dll'))
except OSError:
    pass
