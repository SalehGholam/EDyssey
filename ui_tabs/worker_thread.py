# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 15:11:56 2024

@author: SGholam
"""
from PyQt5.QtCore import (QThread, pyqtSignal, QMutex, QMutexLocker, QObject,
                          QRunnable, QThreadPool)
from functools import partial
#%% general worker thread
class WorkerThread_General(QRunnable):
    # exec_signal = pyqtSignal(object)
    # exec_signal = pyqtSignal()
    stopped = pyqtSignal()
    
    def __init__(self, func, index=0, *args, **kwargs):
        super(WorkerThread_General, self).__init__() #TODO check
        self.func = func
        self.index = index
        self.args = args
        # print(self.args)
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        self.is_running = True
    
    def run(self):
        if self.is_running:
            self.result = self.func(*self.args, **self.kwargs)
            self.signals.results.emit(self.result, self.index)
        else:
            self.signals.stopped.emit()
        self.signals.finished.emit()
    
    def stop(self):
        self.is_running = False
#%%
# Step 1: Create a Worker class that inherits from QRunnable
class WorkerSignals(QObject):
    # Define custom signals (for communicating between thread and GUI)
    finished = pyqtSignal()  # Task is done
    stopped = pyqtSignal()
    # results = pyqtSignal(object, int)  # Task returns a result
    results = pyqtSignal(object, object)  # Task returns a result

# =============================================================================
# class Worker(QRunnable):
#     def __init__(self):
#         super().__init__()
#         self.signals = WorkerSignals()
# 
#     def run(self):
#         pass
# =============================================================================
