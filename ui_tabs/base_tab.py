# -*- coding: utf-8 -*-
"""Shared base class for the 4 top-level tab widgets (Tab_ROI_on_4D,
Tab_Create_NavSignal, Tab_Tracking_CV2, Tab_SAM2). Each used to hand-copy
the same handful of lines into its own __init__ - a per-tab logger, a
threadpool, and the `_cancelling` guard-flag convention used throughout
this app to suppress the error popups a deliberate Cancel would otherwise
trigger (see each tab's own cancel_running_work()). This just centralizes
that so the four copies can't drift out of sync with each other (as
tab_create_navSignal.py's `_stopping`/`stop_worker` naming had, before
being unified to match the other three here).

Each tab still builds its own layout entirely itself, in its own
init_widget() - besides the one shared piece of boilerplate factored out
below (build_left_panel()), this module owns none of that.
"""
import os
import PyQt5.QtWidgets as qtw
from PyQt5.QtCore import Qt, QThreadPool
from PyQt5.QtGui import QFontDatabase
import matplotlib.pyplot as plt
import EDyssey.io_utils as io
from .logging_utils import get_tab_logger
from .display_settings import DisplaySettings, PLOT_DEFINITIONS


def compute_left_panel_width(base=440, min_width=420, max_width=480, fraction=0.22):
    """Left input panel width (px) for the current primary screen - a fixed
    (non-draggable) value like the `width_userInput` constant every tab used
    to hard-code, but scaled to the display instead of being the same
    literal number on a small laptop screen and a large monitor alike.

    `base` is returned unchanged if no screen can be queried (e.g. running
    headless) - the previous hard-coded behaviour. Otherwise the panel is
    sized to `fraction` of the primary screen's available width, clamped to
    [`min_width`, `max_width`] so it neither eats most of a small screen nor
    goes needlessly wide on a large one.

    The defaults (and `min_width` especially) are sized to comfortably fit
    the widest row across all 4 tabs' panels (long smart-scan/tracker rows
    with several labels+combos+buttons) without clipping - each tab's own
    QScrollArea wrapper (see tab_*.py's init_widget()) only scrolls
    vertically, so anything wider than this budget would otherwise be
    hidden behind the scrollbar rather than reachable at all.
    """
    screen = qtw.QApplication.primaryScreen()
    if screen is None:
        return base
    avail_width = screen.availableGeometry().width()
    if avail_width <= 0:
        return base
    return max(min_width, min(max_width, int(avail_width * fraction)))


def build_left_panel(splitter, width_userInput):
    """Build the scrollable left parameter panel and add it to `splitter` -
    the boilerplate every tab used to hand-copy identically into its own
    init_widget() (a QScrollArea wrapper, sized to `width_userInput` - see
    compute_left_panel_width() - so the panel scrolls vertically instead of
    squeezing every box into whatever height the window happens to have).

    `width_userInput` is padded by the vertical scrollbar's own width so the
    viewport (where everything actually gets laid out) still gets the full
    `width_userInput` this panel's widgets are sized for - otherwise the
    scrollbar itself would eat into that budget and the rightmost widgets in
    each row would get clipped/hidden behind it.

    Returns the QVBoxLayout to add each tab's own boxes into.
    """
    left_widget = qtw.QWidget()
    left_scroll = qtw.QScrollArea()
    left_scroll.setWidget(left_widget)
    left_scroll.setWidgetResizable(True)
    scrollbar_w = qtw.QApplication.style().pixelMetric(qtw.QStyle.PM_ScrollBarExtent)
    left_scroll.setFixedWidth(width_userInput + scrollbar_w)
    left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    splitter.addWidget(left_scroll)
    return qtw.QVBoxLayout(left_widget)


# The "Data Type" combo's own display label for the eventem-format hdf5
# entry (see resolve_hdf5_dtype/glob_ext_for_dtype below) - one shared
# constant so every combo (all 4 tabs) and every place that reads a
# combo's current selection back stays in sync. Deliberately NOT the same
# string as '.hdf5_eventem', the internal dtype identifier this resolves
# to (used throughout EDyssey.io_utils - loaders.py, nav_image.py, the
# worker_*.py subprocess scripts, etc.) - that identifier is never shown
# to the user and is unaffected by this label; only what the combo box
# itself displays/stores as its current text is this constant.
HDF5_EVENTEM_LABEL = '.hdf5 (eventem)'


def resolve_hdf5_dtype(fn, combo_selection=None):
    """Return the dtype string to actually load `fn` with - its own file
    extension, except for an ambiguous '.hdf5' file.

    eventem's own export layout (a raw `f['4D']` dataset, internal dtype
    identifier '.hdf5_eventem') and a conventional/third-party HDF5
    4D-STEM file loadable via HyperSpy ('.hdf5') both commonly live on
    disk as a plain '.hdf5' file - nothing in the file's name or extension
    distinguishes them (see EDyssey.io_utils.loaders' module docstring),
    so a '.hdf5' file can only be resolved by asking the user:
    `combo_selection` is the tab's own "Data Type" combo's currently
    selected text - if it explicitly reads HDF5_EVENTEM_LABEL or '.hdf5',
    that selection wins. Any other selection (e.g. 'All files'/'All
    Files', or a mismatched one) falls back to eventem, matching this
    app's original (pre-standard-hdf5-support) behaviour so existing
    eventem exports keep loading unchanged unless the user deliberately
    picks '.hdf5' (standard) instead.

    Every other extension (.tpx3/.hspy/.zspy/.mib/.blo/...) is returned
    as-is - no ambiguity, so `combo_selection` is irrelevant.
    """
    ext = os.path.splitext(fn)[-1]
    if ext != '.hdf5':
        return ext
    if combo_selection == '.hdf5':
        return '.hdf5'
    return '.hdf5_eventem'


def glob_ext_for_dtype(dtype):
    """The real on-disk extension to glob/filter for, given a "Data Type"
    combo selection - identity for every entry except HDF5_EVENTEM_LABEL,
    which (see resolve_hdf5_dtype) is not a real extension: eventem's own
    export layout physically lives in a plain '.hdf5' file, same as a
    conventional/HyperSpy-loadable one, so both combo entries must glob the
    same '*.hdf5' pattern."""
    return '.hdf5' if dtype == HDF5_EVENTEM_LABEL else dtype


def get_existing_directory(parent, caption, start_dir=''):
    """Folder picker that actually shows files while browsing, so the user
    can visually confirm a folder holds what they're looking for (e.g.
    .tpx3/.zspy 4D signals) before selecting it.

    Two earlier approaches to this both turned out unreliable:
    QFileDialog's own Directory mode with ShowDirsOnly=False either fell
    back to Qt's own cross-platform dialog widget (DontUseNativeDialog=True
    - functional, but a different look from the rest of the OS, reported
    as undesirable) or, left native, silently kept using Windows' legacy
    dirs-only folder tree anyway on at least some systems (ShowDirsOnly=False
    alone wasn't enough to avoid it - files still didn't show).

    This instead uses the plain native "Open File" dialog (QFileDialog.getOpenFileName) -
    guaranteed native chrome AND guaranteed to show files, since that's
    what an open-file dialog is for - and returns the *folder* the chosen
    file lives in, matching getExistingDirectory's own contract (a
    directory path, or '' if cancelled). The user picks any file inside
    the folder they want (their 4D signal itself, comment.txt, anything)
    rather than the folder directly.
    """
    fn, _ = qtw.QFileDialog.getOpenFileName(parent, caption, start_dir, 'All Files (*)')
    return os.path.dirname(fn) if fn else ''


class TabBase(qtw.QWidget):
    def __init__(self, tab_name, parent=None, own_threadpool=True):
        """
        Args:
            tab_name: Passed to get_tab_logger() - also the name shown in
                the Qt log console for this tab's messages.
            own_threadpool: Most tabs get their own QThreadPool (isolates
                one tab's background work from the others' queue). Pass
                False for a tab that deliberately shares
                QThreadPool.globalInstance() instead (e.g. the navigator
                tab, whose batch nav-image jobs are dispatched via
                QProcess anyway and only use the global pool for small
                one-off clip/frame-export workers) - in that case no
                `self.threadpool` attribute is set here, and the subclass
                is expected to call QThreadPool.globalInstance() itself.
        """
        super().__init__(parent)
        self._tab_name = tab_name
        self.logger = get_tab_logger(tab_name)
        if own_threadpool:
            self.threadpool = QThreadPool()
        self._cancelling = False  # set by cancel_running_work(); suppresses error popups it causes
        # Ribbon text/icon size and plot font scale, shared with every other
        # tab and the Edit tab (see display_settings.py) - applied once here
        # (subclasses build self.ribbon_page/self.ribbon/self.figure etc.
        # *after* this __init__ call returns, so the first real application
        # happens lazily, the next time DisplaySettings changes; each tab's
        # own init_widget() calls self.apply_display_settings() once at the
        # end of its own construction to pick up the current values too).
        DisplaySettings.instance().changed.connect(self.apply_display_settings)

    def _blit_canvas(self, canvas, figure, bg_attr, artists,
                     hide_for_background=(), titles_for_background=()):
        """Render `artists` onto `canvas` via blit instead of a full
        canvas.draw()/draw_idle() - shared by Tab_Tracking_CV2 (its main
        per-frame update_canvas) and Tab_SAM2 (its cheap denoise/contrast
        preview refresh - see _refresh_current_frame_display).

        Reuses a cached "clean" background (everything in the figure except
        `artists`) stored in `self.<bg_attr>`. That background is captured
        lazily - whenever `self.<bg_attr>` is None (first use, or after
        being invalidated elsewhere e.g. on canvas resize, new data, or a
        scale bar being added/removed) - by briefly hiding
        `hide_for_background` and blanking `titles_for_background`, doing
        one full draw(), then restoring them before the real content is
        blitted on top.
        """
        if getattr(self, bg_attr) is None:
            prev_visible = [a.get_visible() for a in hide_for_background]
            for a in hide_for_background:
                a.set_visible(False)
            prev_titles = [ax.get_title() for ax in titles_for_background]
            for ax in titles_for_background:
                ax.set_title('')

            canvas.draw()
            setattr(self, bg_attr, canvas.copy_from_bbox(figure.bbox))

            for a, v in zip(hide_for_background, prev_visible):
                a.set_visible(v)
            for ax, t in zip(titles_for_background, prev_titles):
                ax.set_title(t)

        canvas.restore_region(getattr(self, bg_attr))
        for artist in artists:
            artist.axes.draw_artist(artist)
        canvas.blit(figure.bbox)

    def cancel_running_work(self):
        """Stop this tab's running background work. The base implementation
        just raises the flag other code already checks against - subclasses
        that actually have work to cancel (QProcess pools, SAM2
        segmentation, etc.) override this, calling super().cancel_running_work()
        first, then killing/discarding whatever they own."""
        self._cancelling = True

    def cleanup(self):
        """Release resources held by this tab. Called by MainWindow's
        closeEvent on every tab, unconditionally - subclasses with real
        resources (running subprocesses, matplotlib figures, log console
        subscriptions) override this, and should still call
        super().cleanup() (currently a no-op, kept for future shared
        cleanup and so every override reads the same way)."""
        pass

    def show_metadata_dialog(self):
        """"View Metadata" button: show the raw comment.txt content this
        tab's Load/Browse Metadata controls already read from (see
        load_metadata()), read-only - lets the user see every field
        actually logged there (e.g. accelerating voltage, camera length),
        not just the scan size/dwell time load_metadata() itself parses
        into a spinbox. Shared here because every tab with metadata support
        resolves it the same way: self.metadata_path_override (set by that
        tab's own Browse Metadata button) if set, else whichever 4D-signal
        line edit this tab has - lineEdit_dir_signal (ROI on 4D/Navigator)
        or lineEdit_dir_4d (ROI Tracker/SAM2), the two names in use."""
        line_edit = getattr(self, 'lineEdit_dir_signal', None) or getattr(self, 'lineEdit_dir_4d', None)
        path_main = getattr(self, 'metadata_path_override', None) or (line_edit.text() if line_edit else '')
        if not path_main:
            qtw.QMessageBox.critical(self, 'No Signal Selected',
                'Select a 4D signal first (see above) - metadata is read from '
                'a comment.txt next to/inside it.')
            return
        try:
            fn, text = io.read_metadata_text(path_main)
        except OSError as e:
            qtw.QMessageBox.warning(self, 'No Metadata Found',
                f'Could not find/read a comment.txt for this signal:\n{e}')
            return
        dlg = qtw.QDialog(self)
        dlg.setWindowTitle(f'Metadata - {os.path.basename(fn)}')
        layout = qtw.QVBoxLayout(dlg)
        text_edit = qtw.QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setPlainText(text)
        text_edit.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        layout.addWidget(text_edit)
        button_close = qtw.QPushButton('Close')
        button_close.clicked.connect(dlg.close)
        layout.addWidget(button_close, alignment=Qt.AlignRight)
        dlg.resize(520, 520)
        dlg.exec_()

    def show_shortcuts_dialog(self, text):
        """Show `text` (this tab's mouse/keyboard interaction hints - Ctrl+
        drag/click modifiers, middle-click, etc.) in a read-only popup,
        opened from the ribbon's "?" (help) tool. Centralizes what each
        subplot's xlabel used to spell out underneath it - crowded, and on
        a narrow window two adjacent subplots' multi-line hints could
        visibly run into each other - into one place, on demand, instead of
        permanently on screen. Same QDialog+QTextEdit shape as
        show_metadata_dialog above."""
        dlg = qtw.QDialog(self)
        dlg.setWindowTitle('Shortcuts & Controls')
        layout = qtw.QVBoxLayout(dlg)
        text_edit = qtw.QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setPlainText(text)
        layout.addWidget(text_edit)
        button_close = qtw.QPushButton('Close')
        button_close.clicked.connect(dlg.close)
        layout.addWidget(button_close, alignment=Qt.AlignRight)
        dlg.resize(480, 360)
        dlg.exec_()

    # -- Display settings (Edit menu's Display Size dialog) ----------------
    def apply_display_settings(self):
        """Re-apply the shared DisplaySettings (ribbon text scale, ribbon
        icon size, ribbon height, plot font scale, figure/canvas size - see
        display_settings.py) to this tab - connected to
        DisplaySettings.changed in __init__ above, and also called once by
        each tab's own init_widget() after building its ribbon/figure, to
        pick up whatever the dialog's current values already are (e.g. if
        a value was changed, then a duplicate tab is opened afterward).

        Generic/shared for every tab: scales self.ribbon_page's font and
        height (the top parameter ribbon), self.ribbon's icon size (the
        vertical RibbonPanel beside the canvas, if present), and self.figure
        (every tab's one Figure/canvas - see PLOT_DEFINITIONS in
        display_settings.py) - covers all 4 tabs without needing a per-tab
        override, since they all use the same attribute name."""
        settings = DisplaySettings.instance()

        ribbon_page = getattr(self, 'ribbon_page', None)
        if ribbon_page is not None:
            # Captured BEFORE the font-size stylesheet below is (re)applied,
            # so it always reflects the tab's true, un-scaled design height -
            # querying sizeHint() *after* a font-size change would capture
            # an already-scaled value instead, compounding on every apply.
            base_height = getattr(ribbon_page, '_edyssey_base_height', None)
            if base_height is None:
                base_height = ribbon_page.sizeHint().height()
                ribbon_page._edyssey_base_height = base_height
            base_pt = getattr(self, '_ribbon_base_pt', 9)
            ribbon_page.setStyleSheet(f'font-size: {round(base_pt * settings.ribbon_text_scale)}pt;')
            new_height = round(base_height * settings.ribbon_height_scale)
            splitter = getattr(self, '_main_splitter', None)
            if splitter is not None:
                # ribbon_page sits in a resizable QSplitter here (see
                # init_widget) - a hard setFixedHeight would pin its pane and
                # make the splitter's own drag handle a no-op, so the scale
                # setting instead just seeds the splitter's *current* size,
                # leaving the user free to drag it afterward.
                ribbon_page.setMinimumHeight(0)
                ribbon_page.setMaximumHeight(16777215)  # QWIDGETSIZE_MAX
                total = sum(splitter.sizes()) or (new_height + 2000)
                splitter.setSizes([new_height, max(total - new_height, 0)])
            else:
                ribbon_page.setFixedHeight(new_height)

        ribbon_panel = getattr(self, 'ribbon', None)
        if ribbon_panel is not None and hasattr(ribbon_panel, 'set_icon_size'):
            ribbon_panel.set_icon_size(settings.ribbon_icon_size)

        for key, figure in self._display_settings_figures():
            self._rescale_figure_fonts(figure, settings.plot_font_scale)
            self._apply_single_figure_scale(figure, settings.figure_size_scales.get(key, 1.0))

    def wrap_canvas_in_scroll(self, canvas):
        """Wrap `canvas` in a QScrollArea and return it, to add to a layout
        in `canvas`'s place - lets apply_display_settings() give this one
        canvas an actual fixed pixel size (see _apply_single_figure_scale)
        without it being silently stretched straight back to fill its
        container, which is what a plain canvas.setMinimumSize() alone used
        to do (every canvas fills 100% of its container via a stretch=1
        layout factor, so shrinking/growing the canvas itself had no
        visible effect - only resizing something with slack around it,
        like this scroll area's viewport, does). setWidgetResizable stays
        True until a scale away from 100% is actually applied, so this is a
        no-op wrapper at every plot's default size - it still just fills
        its container exactly like a bare canvas would."""
        scroll = qtw.QScrollArea()
        scroll.setWidget(canvas)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(qtw.QFrame.NoFrame)
        canvas._edyssey_scroll_area = scroll
        return scroll

    def _apply_single_figure_scale(self, figure, scale):
        """Resize one figure's canvas to `scale` of its own natural size,
        relative to the size it had the FIRST time this ran (cached on the
        canvas itself as _edyssey_base_canvas_size) - a fixed pixel size at
        every scale except exactly 1.0 (where the canvas goes back to
        filling its scroll area, same as a bare unscaled canvas), so each
        plot resizes independently of every other plot and of the window.
        No-op if this canvas was never wrapped via wrap_canvas_in_scroll()
        (e.g. a tab still under construction, or a canvas this dialog
        doesn't cover)."""
        canvas = getattr(figure, 'canvas', None)
        if canvas is None:
            return
        scroll = getattr(canvas, '_edyssey_scroll_area', None)
        if scroll is None:
            return
        if abs(scale - 1.0) < 1e-6:
            scroll.setWidgetResizable(True)
            canvas.setMinimumSize(0, 0)
            canvas.setMaximumSize(16777215, 16777215)  # QWIDGETSIZE_MAX: clears any earlier setFixedSize
            return
        base_size = getattr(canvas, '_edyssey_base_canvas_size', None)
        if base_size is None:
            base_size = (canvas.width(), canvas.height())
            if base_size[0] < 100 or base_size[1] < 100:
                # Not laid out/shown yet (e.g. its tab has never been the
                # active one) - a real canvas.width()/height() at this point
                # would just be Qt's tiny pre-layout default, not a usable
                # "natural size" to scale from. matplotlib's own default
                # figure size (6.4x4.8in @ 100dpi) is a reasonable stand-in.
                base_size = (640, 480)
            canvas._edyssey_base_canvas_size = base_size
        scroll.setWidgetResizable(False)
        canvas.setFixedSize(round(base_size[0] * scale), round(base_size[1] * scale))

    def _display_settings_figures(self):
        """(key, Figure) for every matplotlib Figure this tab owns, per
        PLOT_DEFINITIONS in display_settings.py - covers all 4 tabs (1 plot
        each for ROI on 4D/Navigator/SAM2, 2 for ROI Tracker's split
        canvas) without needing a per-tab override."""
        figures = []
        for key, tab_name, attr, _label in PLOT_DEFINITIONS:
            if tab_name != self._tab_name:
                continue
            fig = getattr(self, attr, None)
            if fig is not None:
                figures.append((key, fig))
        return figures

    def _rescale_figure_fonts(self, figure, scale):
        """Scale every text artist (titles, axis labels, tick labels,
        figure-level sup-title/sup-xlabel) in `figure` by `scale`, relative
        to the size each one had the FIRST time this ran (cached on the
        figure itself as _edyssey_base_fontsizes) - so repeated scale
        changes are always relative to the tab's original design size, not
        the previous scale (avoids compounding drift)."""
        base_sizes = getattr(figure, '_edyssey_base_fontsizes', None)
        if base_sizes is None:
            base_sizes = {}
            figure._edyssey_base_fontsizes = base_sizes
        for ax in figure.axes:
            for text_artist in [ax.title, ax.xaxis.label, ax.yaxis.label] + list(ax.texts):
                key = id(text_artist)
                if key not in base_sizes:
                    base_sizes[key] = text_artist.get_fontsize()
                text_artist.set_fontsize(base_sizes[key] * scale)
            tick_key = ('ticks', id(ax))
            if tick_key not in base_sizes:
                base_sizes[tick_key] = plt.rcParams['font.size']
            ax.tick_params(axis='both', labelsize=base_sizes[tick_key] * scale)
        for text_artist in figure.texts:  # figure-level suptitle/supxlabel
            key = id(text_artist)
            if key not in base_sizes:
                base_sizes[key] = text_artist.get_fontsize()
            text_artist.set_fontsize(base_sizes[key] * scale)
        canvas = getattr(figure, 'canvas', None)
        if canvas is not None:
            canvas.draw_idle()

    # -- Word-ribbon top parameter panel helpers --------------------------
    # Shared by every tab's init_widget() (first built for Tab_ROI_on_4D,
    # then rolled out to the other three) - a horizontal band across the
    # top of the tab, made of compact captioned columns (each replacing
    # what used to be one QGroupBox in the old left parameter panel)
    # separated by vertical lines, instead of a tall left column. Columns
    # are packed left at their natural content width (NOT stretched to
    # fill the ribbon - see the trailing `layout_ribbon.addStretch(1)` each
    # tab adds after its last column). A tab with more groups than
    # comfortably fits as separate columns stacks the less complex/lower-
    # priority ones vertically inside one column instead (see
    # Tab_ROI_on_4D's combined Edge Detection/SAM2 Segmentation/Summed DP
    # Threshold column for the pattern: call _ribbon_group_end() once per
    # stacked sub-section, all against the same layout_group, with
    # `stretch=False` only on the first sub-section and `separator=False`
    # on every sub-section after the first).
    def _ribbon_group_start(self, layout_ribbon, stretch=1, width=None):
        """Start a new parameter-ribbon column (Word-ribbon style: a
        compact, borderless vertical stack of rows, captioned at the
        bottom) - replaces what used to be one QGroupBox in the old left
        parameter panel. Returns (widget, layout): the widget so callers
        can keep a self.box_X reference the way the old QGroupBox-based
        code did (e.g. some tabs' tpx3-gating findChildren sweeps), the
        layout to add the column's own rows into. Call _ribbon_group_end()
        once those rows are added.

        `width`, if given, fixes the column to that pixel width instead of
        letting it size to its content (rarely needed - most columns
        should just size naturally)."""
        widget = qtw.QWidget()
        if width:
            widget.setFixedWidth(width)
            widget.setSizePolicy(qtw.QSizePolicy.Preferred, qtw.QSizePolicy.Preferred)
        else:
            widget.setSizePolicy(qtw.QSizePolicy.Expanding, qtw.QSizePolicy.Preferred)
        layout_group = qtw.QVBoxLayout(widget)
        layout_group.setContentsMargins(6, 4, 6, 2)
        layout_group.setSpacing(3)
        layout_ribbon.addWidget(widget, stretch)
        return widget, layout_group

    def _ribbon_group_end(self, layout_ribbon, layout_group, caption, separator=True, stretch=True):
        """Finish a ribbon column (or one stacked sub-section of a combined
        column): pin its rows to the top, add its caption label at the
        bottom (Word-ribbon style - small, muted, centered), then a
        vertical separator before the next column (skip for the last
        column in the ribbon, or for a sub-section that isn't the last one
        stacked in its combined column - see `separator`)."""
        if stretch:
            layout_group.addStretch(1)
        label = qtw.QLabel(caption)
        label.setAlignment(Qt.AlignHCenter)
        label.setStyleSheet('color: #999999; font-size: 8pt;')
        layout_group.addWidget(label)
        if separator:
            sep = qtw.QFrame()
            sep.setFrameShape(qtw.QFrame.VLine)
            sep.setFrameShadow(qtw.QFrame.Sunken)
            layout_ribbon.addWidget(sep)

    def _ribbon_inline_separator(self, layout_row):
        """A small vertical line between two distinct concepts sharing one
        ribbon row (e.g. Scan Size | Dwell Time) - narrower-scope than
        _ribbon_group_end's inter-column separator, but the same idea."""
        sep = qtw.QFrame()
        sep.setFrameShape(qtw.QFrame.VLine)
        sep.setFrameShadow(qtw.QFrame.Sunken)
        layout_row.addWidget(sep)
