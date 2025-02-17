# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 17:43:00 2024

@author: SGholam
"""

p = 'E:\OneDrive - Universiteit Antwerpen\GitHub\5DED\sam2_hiera_large.pt'
import json
import sys
import hyperspy.api as hs
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import PyQt5.QtWidgets as qtw
from PyQt5.QtCore import Qt, QThreadPool
from PyQt5.QtGui import QDoubleValidator
from matplotlib_scalebar.scalebar import ScaleBar
import numpy as np
import os
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
# from sam2.build_sam import build_initiate_prediction
from sam2.build_sam import build_sam2_video_predictor
from torch.cuda import empty_cache
from PIL import Image
import gc
from copy import deepcopy
import datetime
import py4DTomo.io_utils as io
import torch
from typing import Literal
from .worker_thread import WorkerThread_General
from glob import glob
from matplotlib.colors import SymLogNorm
import py4DTomo.tracking_utils as tr
from hyperspy.api import signals as hsSignals
path_ffmpeg = r'C:\Users\sgholam\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg.Essentials_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-7.1-essentials_build\bin\ffmpeg.exe'
plt.rcParams['animation.ffmpeg_path'] = path_ffmpeg  # Windows example
#%% tab class
class Tab_SAM2(qtw.QWidget):
    def __init__(self):
        super().__init__()
        
        # threadpool to use in the entire tab
        self.threadpool = QThreadPool()
        # self.threadpool = QThreadPool.globalInstance()
        # logical_processors = os.cpu_count()
        
# =============================================================================
#         if logical_processors > 2:
#             self.threadpool.setMaxThreadCount(logical_processors - 2)
# =============================================================================
        self.threadpool.setMaxThreadCount(3)
        
        self.init_ui()
        self.device = self.check_torch_device()
        
    def init_ui(self):
        # Set the window title and dimensions
        self.setWindowTitle("SAM2 Segmentation")
        
        self.central_widget = qtw.QWidget(self)
        self.layout = qtw.QVBoxLayout(self)
        
        button_w = 110
        button_h_sml = 30
        button_h_lrg = 50
        height_layout_top = 200
        
        # layout top
        layout_top = qtw.QHBoxLayout()
        self.layout.addLayout(layout_top)
        #%% directory
        self.box_dir = qtw.QGroupBox('Directories', self)
        self.box_dir.setFixedSize(350, height_layout_top)
        # self.box_dir.setFixedWidth()
        layout_dir = qtw.QVBoxLayout(self)
        # self.layout.addLayout(layout_dir)
        layout_top.addWidget(self.box_dir)
        self.box_dir.setLayout(layout_dir)
        
        # nav signal dir
        layout_dir_entry = qtw.QHBoxLayout()
        layout_dir.addLayout(layout_dir_entry)
        label_dir = qtw.QLabel('Nav. Signal')
        layout_dir_entry.addWidget(label_dir)
        label_dir.setFixedWidth(55)
        
        self.lineEdit_dir_navSignal = qtw.QLineEdit()
        layout_dir_entry.addWidget(self.lineEdit_dir_navSignal)
        
        self.button_dir_navSignal = qtw.QPushButton('...')
        layout_dir_entry.addWidget(self.button_dir_navSignal)
        self.button_dir_navSignal.clicked.connect(lambda: self.show_dialog('file'))
        
        # 4d dir
        layout_dir_4dSignals = qtw.QHBoxLayout(self)
        layout_dir.addLayout(layout_dir_4dSignals)
        
        label_dir_4d = qtw.QLabel('4D Signals')
        layout_dir_4dSignals.addWidget(label_dir_4d)
        label_dir_4d.setFixedWidth(55)
        
        self.lineEdit_dir_4d = qtw.QLineEdit()
        layout_dir_4dSignals.addWidget(self.lineEdit_dir_4d)
        
        self.button_dir_4dSignals = qtw.QPushButton('...')
        layout_dir_4dSignals.addWidget(self.button_dir_4dSignals)
        self.button_dir_4dSignals.clicked.connect(lambda: self.show_dialog('folder'))
        
        # save dir
        layout_dir_save = qtw.QHBoxLayout()
        layout_dir.addLayout(layout_dir_save)
        
        label_dir_save = qtw.QLabel('Save Dir.')
        layout_dir_save.addWidget(label_dir_save)
        label_dir_save.setFixedWidth(55)
        
        self.lineEdit_dir_save = qtw.QLineEdit()
        layout_dir_save.addWidget(self.lineEdit_dir_save)
        
        self.button_dir_save = qtw.QPushButton('...')
        layout_dir_save.addWidget(self.button_dir_save)
        self.button_dir_save.clicked.connect(lambda: self.show_dialog('folder'))
        
        #%% box for scales
        layout_loadSignal = qtw.QHBoxLayout()
        layout_dir.addLayout(layout_loadSignal)
        self.box_scale = qtw.QGroupBox('Scale bars')
        layout_box_scale = qtw.QVBoxLayout()
        layout_loadSignal.addWidget(self.box_scale)
        
        self.box_scale.setFixedWidth(150)
        self.box_scale.setLayout(layout_box_scale)
        layout_loadSignal.addWidget(self.box_scale)
        self.double_validator = QDoubleValidator(0.0, 1e5, 5)
        
        # real space
        layout_scale_real = qtw.QHBoxLayout()
        layout_box_scale.addLayout(layout_scale_real)
        label_scale_real = qtw.QLabel('Real (nm)')
        label_scale_real.setFixedWidth(55)
        layout_scale_real.addWidget(label_scale_real)
        self.lineEdit_scale_real = qtw.QLineEdit(self)
        layout_scale_real.addWidget(self.lineEdit_scale_real)
        self.lineEdit_scale_real.setValidator(self.double_validator)
        # reciprocal space
        layout_scale_recip = qtw.QHBoxLayout()
        layout_box_scale.addLayout(layout_scale_recip)
        label_scale_recip = qtw.QLabel('Recip. (\u00C5<sup>-1</sup>)')
        label_scale_recip.setFixedWidth(55)
        layout_scale_recip.addWidget(label_scale_recip)
        self.lineEdit_scale_recip = qtw.QLineEdit(self)
        layout_scale_recip.addWidget(self.lineEdit_scale_recip)
        self.lineEdit_scale_recip.setValidator(self.double_validator)
        
        self.lineEdit_scale_recip.textChanged.connect(lambda: self.update_canvas(
            self.slider_imgNo.value()))
        self.lineEdit_scale_real.textChanged.connect(lambda: self.update_canvas(
            self.slider_imgNo.value()))
        
        
        self.button_loadNavigation = qtw.QPushButton('Load Signal')
        layout_loadSignal.addWidget(self.button_loadNavigation, alignment=Qt.AlignRight)
        self.button_loadNavigation.setFixedSize(110, 50)
        self.button_loadNavigation.clicked.connect(self.load_navSignal)
        #%% feature handling
        self.masks_images = {}
        
        layout_topRight = qtw.QVBoxLayout()
        layout_top.addLayout(layout_topRight)
        self.box_buttons = qtw.QGroupBox('Features Handling')
        self.box_buttons.setFixedSize(350, height_layout_top//2)
        layout_topRight.addWidget(self.box_buttons)
        
        layout_features = qtw.QHBoxLayout(self)
        self.box_buttons.setLayout(layout_features)
        self.layout.addLayout(layout_features)
        
        self.label_obj_id = qtw.QLabel('Object ID')
        layout_features.addWidget(self.label_obj_id)

        self.spinbox_obj_id = qtw.QSpinBox()
        layout_features.addWidget(self.spinbox_obj_id)
        self.spinbox_obj_id.setFixedWidth(50)
        self.spinbox_obj_id.setMinimum(1)
        self.spinbox_obj_id.setMaximum(1)
        # self.spinbox_obj_id.valueChanged.connect(lambda: self.update_max_obj_id(1))
        self.spinbox_obj_id.valueChanged.connect(lambda: self.update_canvas(self.slider_imgNo.value()))

        spacer = qtw.QSpacerItem(40, 20, qtw.QSizePolicy.Expanding, qtw.QSizePolicy.Minimum)
        layout_features.addItem(spacer)

        self.button_reset_image = qtw.QPushButton('Reset Image Points', self)
        self.button_reset_image.setFixedSize(button_w, button_h_sml)
        layout_features.addWidget(self.button_reset_image)
        self.button_reset_image.clicked.connect(self.reset_image_points)
        
        self.button_reset_allPoints = qtw.QPushButton('Reset All Points', self)
        self.button_reset_allPoints.setFixedSize(button_w, button_h_sml)
        layout_features.addWidget(self.button_reset_allPoints)
        self.button_reset_allPoints.clicked.connect(self.reset_all_points)
        #%% run sam2
        self.box_sam2 = qtw.QGroupBox('SAM2 Segmentation & Tracker')
        self.box_sam2.setFixedSize(350, height_layout_top//2)
        layout_topRight.addWidget(self.box_sam2)
        layout_sam = qtw.QHBoxLayout(self)
        self.box_sam2.setLayout(layout_sam)
        # image
        self.button_runSeg_img = qtw.QPushButton('Segment Image', self)
        self.button_runSeg_img.setFixedSize(button_w, button_h_lrg)
        layout_sam.addWidget(self.button_runSeg_img)
        # self.button_runSeg_img.clicked.connect(self.SAM2_image_predictor)
        self.button_runSeg_img.clicked.connect(self.initiate_prediction)
        
        # clip
        self.button_runSeg_clip = qtw.QPushButton('Track Segment(s)!', self)
        self.button_runSeg_clip.setFixedSize(button_w, button_h_lrg)
        layout_sam.addWidget(self.button_runSeg_clip)
        self.button_runSeg_clip.clicked.connect(self.propagate_in_video)
        self.button_runSeg_clip.setEnabled(False)

# =============================================================================
#         self.button_reset_state = qtw.QPushButton('Reset State', self)
#         self.button_reset_state.setFixedSize(button_w, button_h)
#         layout_sam.addWidget(self.button_reset_state)
#         self.button_reset_state.clicked.connect(self.reset_state)
# =============================================================================
        
        for wid in layout_sam.findChildren(qtw.QWidget):
            wid.setDisabled(True)
        #%% extract 3DED
        self.box_3ded = qtw.QGroupBox('Extract 3DED')
        layout_box_3ded = qtw.QVBoxLayout()
        # self.box_3ded.setFixedWidth(350)
        self.box_3ded.setFixedSize(350, height_layout_top)
        self.box_3ded.setLayout(layout_box_3ded)
        layout_top.addWidget(self.box_3ded)
        
        layout_roi_frame = qtw.QHBoxLayout()
        layout_box_3ded.addLayout(layout_roi_frame)
        label_roi_3ded = qtw.QLabel('Roi No')
        layout_roi_frame.addWidget(label_roi_3ded)
        self.combo_3ded = qtw.QComboBox()
        self.combo_3ded.setFixedWidth(50)
        layout_roi_frame.addWidget(self.combo_3ded)
        self.update_combo_3ded()
        
        # layout_roi_frame.addItem(spacer)
        
        label_threadNo = qtw.QLabel('Thread No')
        layout_roi_frame.addWidget(label_threadNo)
        self.spinbox_threadNo = qtw.QSpinBox(self)
        layout_roi_frame.addWidget(self.spinbox_threadNo)
        self.spinbox_threadNo.setRange(1,os.cpu_count()-1)
        self.spinbox_threadNo.setValue(3)
        self.spinbox_threadNo.valueChanged.connect(self.set_threadNo)
        
        label_finalFrame = qtw.QLabel('Final Frame')
        self.spinbox_finalFrame = qtw.QSpinBox()
        self.spinbox_finalFrame.setFixedWidth(50)
        self.spinbox_finalFrame.setMinimum(1)
        for wid in [label_finalFrame, self.spinbox_finalFrame]:
            layout_roi_frame.addWidget(wid)
        
        layout_extract_button = qtw.QHBoxLayout()
        layout_box_3ded.addLayout(layout_extract_button)
        self.button_3ded = qtw.QPushButton('Extract!')
        self.button_3ded.setFixedSize(button_w, button_h_lrg)
        layout_extract_button.addWidget(self.button_3ded)
        self.button_3ded.clicked.connect(self.extract_3ded)
        
        self.button_save_results = qtw.QPushButton('Save Results')
        # self.button_save_results.setFixedHeight(35)
        self.button_save_results.setFixedSize(button_w, button_h_lrg)
        layout_extract_button.addWidget(self.button_save_results)
        self.button_save_results.clicked.connect(self.save_results)
        
        self.disable_3ded_widgets(True)
        layout_top.addItem(spacer)
        #%% canvas
        layout_canvas = qtw.QHBoxLayout()
        self.layout.addLayout(layout_canvas)
        
        self.figure = Figure()
        # self.figure = Figure(figsize=(16,8)) # with figsize
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(NavigationToolbar(self.canvas, self))
        self.ax_nav = self.figure.add_subplot(131)
        self.ax_seg = self.figure.add_subplot(132)
        self.ax_dp = self.figure.add_subplot(133)
        self.img_zero = np.zeros((512,512), dtype='int16')
        self.img_display = {}
        self.img_display['nav'] = self.ax_nav.imshow(self.img_zero, cmap='gray')
        self.ax_nav.set_title('Navigation')
        self.img_display['seg'] = self.ax_seg.imshow(self.img_zero, cmap='gray')
        self.img_display['seg_mask'] = self.ax_seg.imshow(self.img_zero, cmap='gray')
        self.ax_seg.set_title('Segmented')
        self.img_display['dp'] = self.ax_dp.imshow(self.img_zero, cmap='inferno', 
                                                    norm=SymLogNorm(linthresh=1))
        self.ax_dp.set_title('Extracted DP')
        for ax in [self.ax_dp, self.ax_nav, self.ax_seg]:
            ax.set_axis_off()
        self.figure.tight_layout()
        layout_canvas.addWidget(self.canvas)
        
        self.canvas.mpl_connect("button_press_event", self.on_click)
        
        self.masks_plotted = []
        #%% slider
        layout_slider_imgCounter = qtw.QHBoxLayout(self)
        self.layout.addLayout(layout_slider_imgCounter)
        
        self.label_imgCounter = qtw.QLabel('Img No.')
        layout_slider_imgCounter.addWidget(self.label_imgCounter)
        
        self.slider_imgNo = qtw.QSlider(self)
        self.slider_imgNo.setOrientation(1)  # Horizontal slider
        self.slider_imgNo.setRange(0,0)
        layout_slider_imgCounter.addWidget(self.slider_imgNo)
        
        # self.update_canvas(0)
        self.slider_imgNo.valueChanged.connect(self.update_canvas)
        #%% progress bar
        layout_progress_bar = qtw.QHBoxLayout()
        self.layout.addLayout(layout_progress_bar)
        
        self.progress_bar = qtw.QProgressBar()
        layout_progress_bar.addWidget(self.progress_bar)
        self.progress_bar.setRange(0, 100)
    #%% GUI functions
    def check_torch_device(self):
        
        # check device (cuda or cpu)
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        print(f"using device: {device}")
        
        if device.type == "cuda":
            # use bfloat16 for the entire notebook
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        elif device.type == "mps":
            print(
                "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
                "give numerically different outputs and sometimes degraded performance on MPS. "
                "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
            )
        
        return device

            
    def make_SAM2_predictor(self, opt: Literal['video', 'image']):
        if not hasattr(self, 'device'):
            self.device = self.check_torch_device()
        # SAM2 checkpoints
        path_file = os.path.abspath(__file__)
        path_file = os.path.dirname(path_file)
        path_main = os.path.dirname(path_file)
        path_checkpoints = os.path.join(path_main, 'py4DTomo', 'tracking_utils', 'SAM2_checkpoints')
        fn_checkpoint = 'sam2.1_hiera_large.pt'
        sam2_checkpoint = os.path.join(path_checkpoints, fn_checkpoint)
        if not os.path.isfile(sam2_checkpoint):
            raise FileNotFoundError('SAM2 model checkpoints are not found for version 2.1!')
# =============================================================================
#             fn_checkpoint = [fn for fn in os.listdir(path_checkpoints) if 'hiera_large.pt' in fn][0]
#             sam2_checkpoint = os.path.join(path_checkpoints, fn_checkpoint)
# =============================================================================

        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        if opt == 'video':
            # predictor = build_initiate_prediction(model_cfg, sam2_checkpoint, device=self.device)
            predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=self.device)
        elif opt == 'image':
            sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=self.device)
            predictor = SAM2ImagePredictor(sam2_model)
        return predictor
            
# =============================================================================
#     def SAM2_image_predictor(self):
#         if not hasattr(self, 'predictor_img'):
#             self.predictor_img = self.make_SAM2_predictor(opt='image')
# 
#         imgNo = self.slider_imgNo.value()
#         img_rgb = io.convert_to_rgb(np.array([self.imgs_8bit[imgNo]]))[:,:,:,0]
#         self.predictor_img.set_image(img_rgb)
#         #TODO should it run over all objects?!
#         obj_id = self.spinbox_obj_id.value()
# 
#         input_points = []
#         input_labels = []
#         if 1 not in self.seg_points[obj_id][imgNo]:
#             raise KeyError('There is no positive points to perform segmentation!')
#         for i_p, pt in self.seg_points[obj_id][imgNo].items():
#             input_points += pt
#             input_labels += [i_p for i in pt]
#             
#         masks, scores, _ = self.predictor_img.predict(
#             point_coords=input_points,
#             point_labels=input_labels,
#             multimask_output=False,)
#         
#         self.img_display['seg'].set_data(self.imgs[imgNo])
#         self.show_mask(masks[0])
#         self.canvas.draw()
#         empty_cache()
#         gc.collect()
# =============================================================================

    def initiate_prediction(self):
        #TODO don't make jpg every time
        worker_make_jpg = WorkerThread_General(self.make_jpg_imgs, 0, self.imgs)
        self.threadpool.start(worker_make_jpg)
        worker_make_jpg.signals.results.connect(self.make_predictor)
    
    def make_predictor(self):
        # print(self.seg_points)
        if not hasattr(self, 'predictor_video'):
            self.predictor_video = self.make_SAM2_predictor('video')
        self.predictor_video = self.make_SAM2_predictor('video')
        if not hasattr(self, 'inference_state'): # check if predictor had run
            self.inference_state = None
        worker_add_points = WorkerThread_General(self.add_points_to_predictor, 0, 
                                                 self.predictor_video, self.inference_state,
                                                 self.path_jpg, self.seg_points)
        self.threadpool.start(worker_add_points)
        worker_add_points.signals.results.connect(self.get_segment_image_results)

    def add_points_to_predictor(self, predictor_video, inference_state, path_jpg, seg_points_dict): # TODO check this carefully
        if inference_state is None:
            inference_state = predictor_video.init_state(path_jpg) # TODO memory efficient?!
        for obj_id in seg_points_dict.keys():
            for i_img in seg_points_dict[obj_id].keys():
                input_points = []
                input_labels = []
                for i_p, pt in seg_points_dict[obj_id][i_img].items():
                    input_points += pt
                    input_labels += [i_p for i in pt]
                
                _, out_obj_ids, out_mask_logits = predictor_video.add_new_points_or_box(
                                        inference_state=inference_state,
                                        frame_idx=i_img,
                                        obj_id=obj_id,
                                        points=input_points,
                                        labels=input_labels)
        shape = tuple(out_mask_logits.cpu().shape)
        shape = (shape[0], shape[2], shape[3])
        out_mask_logits_np = np.zeros(shape, dtype=bool)
        for i, _ in enumerate(out_obj_ids):
            out_mask_logits_np[i] = (out_mask_logits[i] > 0.0).cpu().numpy()
        return inference_state, out_obj_ids, out_mask_logits_np 
    
    def get_segment_image_results(self, result, index):
        self.inference_state, out_obj_ids, out_mask_logits = result
        # print(out_obj_ids)
        # print(self.out_mask_logits)
        imgNo = self.slider_imgNo.value()
        self.masks_images[imgNo] = (out_obj_ids, out_mask_logits)
        self.button_runSeg_clip.setEnabled(True)
        self.update_canvas(imgNo)
        
    
    def propagate_in_video(self):
        def propagate_func(predictor_video, inference_state):
            video_segments = {}  # video_segments contains the per-frame segmentation results
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor_video.propagate_in_video(inference_state):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
            
            del out_frame_idx, out_obj_ids, out_mask_logits
            empty_cache()
            gc.collect()
            return video_segments
        worker_propagate = WorkerThread_General(propagate_func, 0, self.predictor_video, self.inference_state)
        worker_propagate.signals.results.connect(self.get_segment_video_results)
        self.threadpool.start(worker_propagate)
        worker_propagate.signals.results.connect(self.get_segment_video_results)
        
    def get_segment_video_results(self, result, index):
# =============================================================================
#         self.video_segments = {}  # video_segments contains the per-frame segmentation results
#         
#         for out_frame_idx, self.out_obj_ids, self.out_mask_logits in result:
#             self.video_segments[out_frame_idx] = {
#                     out_obj_id: (self.out_mask_logits[i] > 0.0).cpu().numpy()
#                     for i, out_obj_id in enumerate(self.out_obj_ids)
#             }
# =============================================================================
        self.video_segments = result
        w, h = self.imgs[0].shape
        # print('object IDs: ', self.seg_points.keys())
        self.masks_video = {}
        for i_obj in self.seg_points.keys():
            self.masks_video[i_obj] = np.zeros((len(self.imgs), w, h), dtype=bool)
            for i_img in self.video_segments.keys():
                self.masks_video[i_obj][i_img] = self.video_segments[i_img][i_obj]
        self.activate_3ded_widgets(True)
        self.update_canvas(self.slider_imgNo.value())
        self.spinbox_finalFrame.setMaximum(len(self.imgs))
        self.spinbox_finalFrame.setValue(len(self.imgs))
    
    def make_jpg_imgs(self, imgs):
        pathSave = self.lineEdit_dir_save.text()
        self.path_jpg = os.path.join(pathSave, 'JPG Images')
        if not os.path.isdir(pathSave):
            os.mkdir(pathSave)

        if not os.path.isdir(self.path_jpg):
            os.mkdir(self.path_jpg)
        
        # recreate jpg files if they are not the same number as navigation images
        fns = glob(os.path.join(self.path_jpg, '*.jpg'))
        if len(fns) != len(imgs):
            if len(fns) > 0:
                for fn in fns:
                    os.remove(os.path.join(self.path_jpg, fn))

            for i, img in enumerate(imgs):
            # for i, img in enumerate(self.imgs_8bit):
                img = io.convert_img_to_8bit(img)
                img = Image.fromarray(img)
                # img = img.convert('L')  # Convert to grayscale
                img.save(os.path.join(self.path_jpg, f'{i:04d}.jpg'))
      
    def activate_3ded_widgets(self, state):
        for wid in self.box_3ded.findChildren(qtw.QWidget):
            if not isinstance(wid, qtw.QLabel):
                wid.setEnabled(state)
    
    def reset_state(self):
        try:
            self.predictor_video.reset_state(self.inference_state) #TODO check
        except Exception as e:
            print(f"An error occurred: {e}")
        try: 
            del self.masks_video
        except: pass
    
    def clear_instances(self):
        pass
    
    def extract_3ded(self):
        def make_rois(masks):
            rois = {}
            for i_obj in masks.keys():
                rois[i_obj] = np.zeros((len(masks[i_obj]), 4), dtype='int16')
                for i_img, mask in enumerate(masks[i_obj]):
                    temp = np.where(mask==True)
                    ymin = temp[0].min()
                    ymax = temp[0].max() +1
                    xmin = temp[1].min()
                    xmax = temp[1].max() +1
                    w = xmax - xmin
                    h = ymax - ymin
                    r = [xmin, ymin, w, h]
                    r = [int(item) for item in r]
                    rois[i_obj][i_img] = r
            return rois

        def get_tomo_ds(result, index):
            obj_id, fr_id = index
            self.tomo_ds[obj_id][fr_id] = result
            
            # progressbar update
            self.tomo_counter += 1
            self.update_progress_bar(self.tomo_counter, self.tomo_counter_total)
            # plot results at the end
            if self.tomo_counter == self.tomo_counter_total:
                self.update_canvas()

        self.rois = make_rois(self.masks_video)
        
        path_4d = self.lineEdit_dir_4d.text()
        if path_4d == '': # no entry in 4D signals path
            qtw.QMessageBox.critical(self, 'No 4D Signals found!')
            return
        
        fns_4d = glob(os.path.join(path_4d, '*')) 
        if len(fns_4d) == 0:
            qtw.QMessageBox.critical(self, 'Wrong Path', 'No files was found in the path for 4D signals!')
            return
        dtype = os.path.splitext(fns_4d[0])[1] # select data type on the gui
        
        # check if no of files with no of images
        if len(self.imgs_8bit) != len(fns_4d):
            reply = qtw.QMessageBox.question(self, 'Mismatch',
                   'No of 4D signals mismatches the number of images. Do you want to continue?',)
            if reply == qtw.QMessageBox.No:
                return

        # set detector size for tpx3
        if dtype in ['.tpx3', '.hdf5']: # TODO not good
            shape_d_x, shape_d_y = 512, 512
        else:
            shape_d_x, shape_d_y = io.get_det_size(fns_4d[0])
        scanSize = self.imgs.shape[1:]
        
        self.tomo_counter_total = len(self.masks_video) * len(fns_4d)
        self.tomo_counter = 0
        
        self.tomo_ds = {}
        for i_obj in self.masks_video.keys():
            self.tomo_ds[i_obj] = np.zeros((len(fns_4d), shape_d_x, shape_d_y), dtype='uint32')
            for i_fr, fn in enumerate(fns_4d):
                worker = WorkerThread_General(tr.extract_3ded_mask_single_frame, (i_obj, i_fr),
                                              fn, self.masks_video[i_obj][i_fr], dtype, scanSize,
                                              self.rois[i_obj][i_fr])
                worker.signals.results.connect(get_tomo_ds)
                self.threadpool.start(worker)
    

    # def show_mask(self, mask, obj_id=None, random_color=False, borders=True):
    def show_mask(self, mask, obj_id=None, random_color=False, disp_ax=None):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            # color = np.array([30/255, 144/255, 255/255, 0.6])
            cmap = plt.get_cmap("tab10")
            cmap_idx = 0 if obj_id is None else obj_id
            color = np.array([*cmap(cmap_idx)[:3], 0.6])
        h, w = mask.shape[-2:]
        # mask = mask.astype(np.uint8)
        mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        if disp_ax is None: 
            self.img_display['seg_mask'].set_data(mask_image)
        else: # TODO not needed anymore
            disp_ax.set_data(mask_image)
            

    def show_dialog(self, f):
        sender = self.sender()
        if sender == self.button_dir_navSignal:
            file_filter = "supported signals (*.zspy *.hspy);;All Files (*)"
            # path = qtw.QFileDialog.getOpenFileNames(self, "Select 4D Signals Folder", '', file_filter)
            path = qtw.QFileDialog.getOpenFileName(self, "Select 4D Signals Folder", '', file_filter)
            if path:
                self.lineEdit_dir_navSignal.setText(path[0])
                path_save = os.path.join(os.path.dirname(path[0]), '5DED Analysis')
                self.lineEdit_dir_save.setText(path_save)
                
        elif sender == self.button_dir_4dSignals:
            path = qtw.QFileDialog.getExistingDirectory(self, "Select 4D Folder")
            if path:
                self.lineEdit_dir_4d.setText(path)
                
        elif sender == self.button_dir_save:
            path = qtw.QFileDialog.getExistingDirectory(self, "Select Destination Folder")
            if path:
                self.lineEdit_dir_save.setText(path)
    
    def load_navSignal(self):
        self.reset_data() # TODO

        self.fn_navSignal = self.lineEdit_dir_navSignal.text()
        s = hs.load(self.fn_navSignal)
        self.imgs = s.data
        self.imgs_8bit = io.convert_to_8bit(s).data
        self.seg_points = {}
        self.scatter_plots = []
        
        # set size and limit of the navigation data
        shape_x, shape_y = self.imgs[0].shape
        self.img_display['nav'].set_extent([0, shape_y, shape_x, 0])
        self.img_display['nav'].set_clim(vmin=self.imgs.min(), vmax=self.imgs.max())
        self.update_canvas(0)
        self.slider_imgNo.setRange(0, len(self.imgs)-1)
    
    def set_threadNo(self, value):
        self.threadpool.setMaxThreadCount(value)
    
    def reset_data(self):
        # resetting previous run
        if hasattr(self, 'masks_video'):
            del self.masks_video
        if hasattr(self, 'inference_state'):
            del self.inference_state
        self.button_runSeg_clip.setEnabled(False)
        
    def update_canvas(self, imgNo=None, obj_id=None):
        if imgNo is None:
            imgNo = self.slider_imgNo.value()
        if obj_id is None:
            obj_id = self.spinbox_obj_id.value()
        
            
        self.img_display['nav'].set_data(self.imgs[imgNo])
        self.plot_points(imgNo, obj_id)
        self.ax_nav.set(title=f'Nav. Image No: {imgNo}')
    
        # plot segmentation masks if they exist
        if hasattr(self, 'predictor_video'): #  or hasattr(self, 'masks_images')
            self.img_display['seg'].set_data(self.imgs[imgNo])
            self.img_display['seg'].set_clim(vmin=self.imgs[imgNo].min(), vmax=self.imgs[imgNo].max())
            try:
                self.show_mask(self.masks_video[obj_id][imgNo], obj_id=obj_id)
            except:
                try:
                    if imgNo in self.masks_images.keys():
                        obj_ids, obj_logits = self.masks_images[imgNo]
                        obj_id = obj_ids[obj_id - 1]
                        obj_logit = obj_logits[obj_id - 1]
                        self.show_mask(obj_logit, obj_id=obj_id)
                except:
                    self.img_display['seg_mask'].set_data(self.img_zero)
                    self.img_display['seg'].set_data(self.img_zero)
        if hasattr(self, 'tomo_ds'):
            self.plot_dp()
            
        # scale bars 
        #TODO adding and removing the artist is not efficient
        scale_real = self.lineEdit_scale_real.text()
        try:
            scale_real = float(scale_real)
            for ax in [self.ax_nav, self.ax_seg]:
                scalebar_real = ScaleBar(scale_real, 'nm', dimension='si-length', location='lower left')
                for artist in ax.artists:
                    if isinstance(artist, ScaleBar):
                        artist.remove()
                ax.add_artist(scalebar_real)
        except Exception as e:
            # print(e)
            pass
        scale_recip = self.lineEdit_scale_recip.text()
        try:
            scale_recip = float(scale_recip)
            if scale_recip != 0:
                scalebar_recip = ScaleBar(scale_recip*10, '1/nm', dimension='si-length-reciprocal', location='lower left',
                                    scale_formatter=lambda value, unit:  f'{value / 10}'r' $\AA^{-1}$', fixed_value=5)
                for artist in self.ax_dp.artists:
                        if isinstance(artist, ScaleBar):
                            artist.remove()
                self.ax_dp.add_artist(scalebar_recip)
        except Exception as e:
            # print(e)
            pass
        shape_x, shape_y = self.imgs[0].shape
        self.img_display['nav'].set_extent([0, shape_y, shape_x, 0])
        self.canvas.draw()
    
    def update_combo_3ded(self):
        if self.combo_3ded.count() > 0:
            self.combo_3ded.clear()
        object_ids = [self.combo_3ded.itemText(i) for i in range(self.spinbox_obj_id.maximum())]
        object_ids = ['all'] + object_ids
        self.combo_3ded.addItems(object_ids)
    
    def plot_points(self, imgNo=None, obj_id=None):
        if not imgNo:
            imgNo = self.slider_imgNo.value()
        if not obj_id:
            obj_id = self.spinbox_obj_id.value()
        for p in self.scatter_plots:
            try:
                p.remove()
            except ValueError:
                pass
        self.scatter_plots = []
        if (obj_id in self.seg_points):
            if(imgNo in self.seg_points[obj_id]):
                if 1 in self.seg_points[obj_id][imgNo]:
                    pos_points = self.seg_points[obj_id][imgNo][1]
                    pos_points = np.array(pos_points)
                    scatter_p = self.ax_nav.scatter(pos_points[:,0], pos_points[:, 1], 
                                                    color='green', marker='o', s=20, 
                                                    linewidth=1.25)
                    self.scatter_plots.append(scatter_p)
    
                if 0 in self.seg_points[obj_id][imgNo]:
                    neg_points = self.seg_points[obj_id][imgNo][0]
                    neg_points = np.array(neg_points)
                    scatter_n = self.ax_nav.scatter(neg_points[:,0], neg_points[:, 1], 
                                                    color='red', marker='o', s=20, 
                                                    linewidth=1.25)
                    self.scatter_plots.append(scatter_n)
        
        self.canvas.draw()
            
    def on_click(self, event):
        imgNo = self.slider_imgNo.value()
        obj_id = self.spinbox_obj_id.value()
        if event.inaxes == self.ax_nav:
            p = [event.xdata, event.ydata]
            # left click is positive and right click negative
            click = 1 if event.button == 1 else 0 # if event.button == 3 else False
            if obj_id not in self.seg_points: # check obj_id
                self.seg_points[obj_id] = {}
            if imgNo not in self.seg_points[obj_id]: # check imgNo
                self.seg_points[obj_id][imgNo] = {}
            try:
                self.seg_points[obj_id][imgNo][click].append(p)
            except:
                self.seg_points[obj_id][imgNo][click] = [p]
        self.update_max_obj_id(len(self.seg_points)+1)
        # self.plot_points()
        self.update_canvas(imgNo, obj_id)
            
    def update_max_obj_id(self, value):
        self.spinbox_obj_id.setMaximum(value)
    
    def reset_all_points(self):
        self.seg_points = {}
        imgNo = self.slider_imgNo.value()
        try:
            self.update_canvas(imgNo)
        except:
            print('No points to reset')
    
    def reset_image_points(self):
        imgNo = self.slider_imgNo.value()
        obj_id = self.spinbox_obj_id.value()
        try:
            self.seg_points[obj_id][imgNo] = {}
            self.update_canvas(imgNo)
        except:
            print('No points for this object or image no. to reset!')
    
    def disable_3ded_widgets(self, state):
        for wid in self.box_3ded.findChildren(qtw.QWidget):
            if not isinstance(wid, qtw.QLabel):
                wid.setDisabled(state)
    
    def plot_dp(self, roiNo=None, imgNo=None):
        if hasattr(self, 'tomo_ds'):
            if not imgNo:
                imgNo = self.slider_imgNo.value()
            if not roiNo:
                obj_id = self.spinbox_obj_id.value()
            if obj_id in self.tomo_ds.keys():
                img = self.tomo_ds[obj_id][imgNo]
                self.img_display['dp'].set_data(img)
                self.img_display['dp'].set_clim(vmin=img.min(), vmax=img.max())
                # self.img_display['dp'].set_clim(vmin=1, vmax=img.max())
                shape_x, shape_y = img.shape
                self.img_display['dp'].set_extent([0, shape_y, shape_x, 0])

    def update_progress_bar(self, value, total):
        value = value / total * 100
        value = int(value)
        self.progress_bar.setValue(value)

    def save_masks(self):
        obj_ids = self.spinbox_obj_id.value()
        fn_suffix = f'{datetime.date.today()}_{datetime.datetime.now().strftime("%H-%M-%S")}'
        self.path_main = os.path.dirname(self.fn_navSignal)
        for i_seg in range(obj_ids):
            i_seg += 1
            try:
                np.save(os.path.join(self.path_main, f'{fn_suffix}_sam2 masks_no {i_seg}'), self.masks_video[i_seg])
                print(f'Masks saved in:\n{self.path_main}')
            except Exception as e:
                print(f'Saving Masks for video failed by an error: {e}')
            # TODO save masks from images
            # try:
                # np.save(os.path.join(self.path_main, f'{sam2 masks_no i_seg}'), self.masks_video[i_seg])
    
    def save_results(self):
        path_save = self.lineEdit_dir_save.text()
        if not os.path.isdir(path_save):
            os.mkdir(path_save)
        
        date = datetime.date.today()
        tim = datetime.datetime.now().strftime("%H-%M-%S")
        
        path_save = os.path.join(path_save, f'{date}__{tim}')
        os.mkdir(path_save)
        
        
        # tracking results, rois, dp
        for obj_id, ds in self.tomo_ds.items():
            path_save_roi = os.path.join(path_save, f'roi No {obj_id}')
            os.mkdir(path_save_roi)
            # input points
            with open(os.path.join(path_save_roi, 'input_points.json'), 'w') as f:
                json.dump(self.seg_points[obj_id], f, indent=4)
            # TODO there is no roi here!
            with open(os.path.join(path_save_roi, f'roi coords_id {obj_id}.npy'), 'wb') as f:
                np.save(f, self.rois[obj_id])
            
            # dp frames
            s = hsSignals.Signal2D(self.tomo_ds[obj_id])
            s.save(os.path.join(path_save_roi, f'3DED_id {obj_id}.hspy'))
            fld_dp = os.path.join(path_save_roi, 'frames')
            worker_frames = WorkerThread_General(io.create_frames, 0, fld_dp, s)
            self.threadpool.start(worker_frames)
            
            # clip dp
            scale_recip = self.lineEdit_scale_recip.text()
            try:
                scale_recip = float(scale_recip)
            except:
                scale_recip = None
            fn_clip_dp = os.path.join(path_save_roi, 'tomo clip')
            worker_clip_dp = WorkerThread_General(io.create_clip_dp, 0, fn_clip_dp,
                                                  s, scale_recip)
            self.threadpool.start(worker_clip_dp)
            
            # clip tracking
            np.save(os.path.join(path_save_roi, f'segmentation masks_ obj ID {obj_id}.npy'), 
                    self.masks_video[obj_id])
            scale_real = self.lineEdit_scale_real.text()
            try:
                scale_real = float(scale_real)
            except:
                scale_real = None
            fn_clip_tracking = os.path.join(os.path.join(path_save_roi, 'tracking clip'))
            worker_tracking = WorkerThread_General(io.create_clip_tracking_with_mask, 0, 
                                                 fn_clip_tracking, self.imgs, 
                                                 self.masks_video[obj_id], obj_id, scale_real)
            self.threadpool.start(worker_tracking)
            
    def clear_model(self):
        try:
            if hasattr(self, 'inference_state'):
                for arg in [self.inference_state, self.predictor_video, 
                            self.out_mask_logits, self.masks_video,
                            self.video_segments
                            ]:
                    try:
                        del arg
                    except:
                        # print('didnt delete the data')
                        pass
        except:
            pass
        empty_cache()
        
    def closeEvent(self,event):
        # empty_cache()
        self.clear_model()
        gc.collect()
        event.accept()
        # app.exit()

if __name__ == "__main__":
    app = qtw.QApplication(sys.argv)
    
    # Create the main window and show it
    window = Tab_SAM2()
    window.show()
    
    # Run the application event loop
    sys.exit(app.exec_())
