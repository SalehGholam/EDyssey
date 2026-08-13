# -*- coding: utf-8 -*-
"""PETS2 (.pts2) project-file generation for the "Make *.pts2" checkbox in
the SAM2/CV2 tabs' Extract 3DED box.

`BASIC_FIELDS`/`ADVANCED_FIELDS`/`AUTOTASK_OPTIONS` describe every editable
PETS setting (keyword, default value, and the exact comment from a
reference PETS2-generated file, reused as tooltips in ui_tabs/pets2_dialog.py)
so the dialog and this module stay in sync without duplicating the list.
Structural multi-line blocks that aren't really "parameters" in the same
sense (badpixels/icerings/distortions/distortionskeys/reconstruction/
cifentries/notes/celllist - all empty or PETS-refined starting points) are
written verbatim from the reference file and aren't user-editable here.
"""
import os
import json
import numpy as np
from .app_dirs import writable_data_dir

_CACHE_DIR = os.path.join(writable_data_dir(), 'temp')
_CACHE_FILE = os.path.join(_CACHE_DIR, 'pets2_last_params.json')

#%% physics

def electron_wavelength_angstrom(voltage_kv):
    """Relativistic electron wavelength (in Angstrom) for a given
    acceleration voltage (in kV) - the same quantity PETS calls `lambda`.

    lambda = h / sqrt(2*m0*e*V*(1 + e*V/(2*m0*c^2)))

    e.g. 200 kV -> ~0.02508 A, matching PETS' own reference value.
    """
    h = 6.62607015e-34   # Planck constant, J*s
    m0 = 9.1093837015e-31  # electron rest mass, kg
    e = 1.602176634e-19  # elementary charge, C
    c = 2.99792458e8     # speed of light, m/s
    V = voltage_kv * 1000.0
    lam_m = h / np.sqrt(2 * m0 * e * V * (1 + e * V / (2 * m0 * c ** 2)))
    return lam_m * 1e10  # m -> Angstrom

#%% parameter catalogue (shared with ui_tabs/pets2_dialog.py)

# Tasks PETS can run automatically at startup - exact wording from the
# reference file's `autotask` comment.
AUTOTASK_OPTIONS = [
    'peak search', 'tilt axis', 'peak analysis', 'find cell', 'find known cell',
    'refine cell', 'process frames', 'reflection profile', 'frame geometry',
    'finalize integration', 'generate 3d map', 'sections', 'check files',
    'global indexing', 'set', 'quit', 'break', 'none',
]

# The handful of settings surfaced directly on the dialog's initial ("basic")
# view (beyond acceleration voltage / aperpixel / geometry / phi / bin /
# center, which need their own typed widgets and physics conversion).
# `bin` isn't here - it gets its own integer QSpinBox in the dialog.
BASIC_FIELDS = [
    ('reflectionsize', 'reflectionsize', '8.0', 'Diameter of integration circle of the reflections (pixels, unbinned)'),
    ('dstarmax', 'dstarmax', '1.400', 'maximum dstar (rec.angstroms) for integration'),
    ('dstarmaxps', 'dstarmaxps', '1.400', 'maximum dstar (rec.angstroms) for peaksearch'),
    ('dstarmin', 'dstarmin', '0.050', 'minimum dstar (rec.angstroms) for integration and peaksearch'),
]

# Every other single-line/vector PETS setting - (keyword, placeholder, default
# value string, comment/tooltip verbatim from the reference file).
ADVANCED_FIELDS = [
    ('noiseparameters', 'noiseparameters', '5.0000      0.0000',
     "Parameters allowing to calculate the sigma(count) on every pixel. The first is G*gamma (gain*cascade factor), the second is sigma**2 of the dark image."),
    ('nonlinearity', 'nonlinearity', 'no    0.0000000000',
     "Detector non-linearity correction: yes/no, followed by the coefficient k in c(corr)=c(orig)*(1+k*c(orig)), with k in 1/(counts/sec)."),
    ('delta', 'delta', '0.0000 0',
     "angle between the positive main rotation axis and the x-y plane (perpendicular to the beam).Positive, if the angle between the axis and the beam is smaller than 90 degrees.The key defines, if delta shpould (1) or shoul not (0) be refined."),
    ('pixelsize', 'pixelsize', '0.0100',
     "size of the pixel in the 3D map of reciprocal space."),
    ('centerradius', 'centerradius', '0.150',
     "excluded region around the central beam. Obsolete, partly overlapping with dstarmin. Will be removed/replaced soon."),
    ('beamstop', 'beamstop', 'no',
     'beamstop settings. beamstop no with no beamstop. "beamstop followed by a list of beamstop points, if beamstop is used.'),
    ('correctbadpixels', 'correctbadpixels', 'yes',
     "apply interpolation of values at bad pixels"),
    ('avoidicerings', 'avoidicerings', 'no',
     "list of ice ring diameters. Peaks in the vicinity of these rings are ignored, if avoidicerings is yes."),
    ('peaksearchmode', 'peaksearchmode', 'auto',
     "Peak search mode - auto (default) = find automatically, manual = click in the picture to add or remove found peaks."),
    ('centermode', 'centermode', 'centralbeam 1',
     "Mode of center determination. centralbeam = use central beam, friedelpairs = use friedel pairs, useold = take whatever was previously set. Optional second argument: always accept center? (0/1). Applied only to Friedel pairs."),
    ('i/sigma', 'i_sigma', '2.00   10.00',
     "Intensity/sigma for peak search and for the calculation of the camel plot"),
    ('moreaveprofiles', 'moreaveprofiles', 'no',
     "Option for average peak profile calculation in the fit profile. If yes, each group of frames will have its own calculated average peak profile."),
    ('peakprofileparams', 'peakprofileparams', '100 1    10.0',
     "Parameters for average peak profile (APP) calculation. First: minimum number of peaks in one APP. Second: minimum I/sigma peaks (0=fixed/1=auto). Third: value of minimum I/sigma for the case fixed."),
    ('peakprofilesectors', 'peakprofilesectors', '0 1 0.40',
     "Parameters for specifying image sectors for APP calculation. First: distinguish image sectors (0/1). Second: radius of the central sector (0=fixed/1=auto). Third: radius value for the case fixed ."),
    ('background', 'background', '0    1.000000',
     "Constant background to be subtracted from all images after reading in. May be negative if needed. Second optional argument is the factor that multiplies the raw image at reading"),
    ('backgroundmethod', 'backgroundmethod', 'median',
     "Method for background subtraction. median = median filter (default, the best), average = average in concentric circles, fft = background by Gaussian smearing. Median should be used in normal circumstances."),
    ('peakanalysis', 'peakanalysis', '0 0    3.7400    0.0080',
     "Peak analysis options: estimate cluster limits automatically (0) or use preset values (1), process automatically (1) or wait for confirmation (0), xyz cluster limit (in pixels), 3D cluster limit (in rec. Angstroms)."),
    ('minclusterpoints', 'minclusterpoints', '2',
     "Minimum number of points in a cluster (used during peak analysis)."),
    ('showrfft', 'showrfft', 'no',
     "Display rfft file in indexing panel."),
    ('indexingmode', 'indexingmode', 'diffspace',
     "Algorithm for automatic cell determination: diffspace = algorithm using difference space clustering, triplets = old \"jana-style\" algorithm, fft = new algorithm using fourier transform of xyz file."),
    ('indexingparameters', 'indexingparameters', '0.0180  2.0000   10000.0  30.0000  3.0000  0.6000  1.0000',
     "Parameters for cell determination: indexing tolerance (rec. A), angular tolerance for rec. direction (deg), maximal primitive volume (rec. A^3), maximal cell length (A), relative length tolerance (%), angular tolerance (deg), maximum Sg for indexing (distortions only)."),
    ('automaxddiff', 'automaxddiff', 'yes',
     "Automatic determination of indexing tolerance. If yes, the indexing tolerance is calculated from the  maximal cell length as 0.45/(maximal cell length)."),
    ('knowncellmethod', 'knowncellmethod', 'refldir',
     "Method for find known cell: refldir = method for determining orientation by rotating around the directions of reflections.spacedir = method for determining orientation by testing all space directions."),
    ('knowncell', 'knowncell', '10.00000    10.00000    10.00000    90.00000    90.00000    90.00000',
     "Known cell parameters for orientation search."),
    ('knowncellparameters', 'knowncellparameters', '3.0000  200',
     "Parameters for searching for a known cell: angular rotation step (deg), number of test reflections for indexing at each position."),
    ('maxindex', 'maxindex', '2',
     "maximum satellite index used in integration. (Applies only to modulated structures.)"),
    ('cellrefinemode', 'cellrefinemode', 'cellandub',
     "Algorithm for cell refinement. cellandub = refinement from 3D peak coordinates (standard), cellfromd = refinement against lengths of the diffraction vectors, cellanddistort = cell from 2D peak positions plus, optionally, distortion parameters."),
    ('maxddiffrefine', 'maxddiffrefine', '0.0180',
     "Indexing tolerance (rec. A) for refine cell."),
    ('automaxddiffref', 'automaxddiffref', 'no',
     "Automatic determination of indexing tolerance in refine cell. If yes, the indexing tolerance is calculated from the maximal cell length actual or maximal fixed cell length as 0.3/(maximal cell length)."),
    ('cellrefineparameters', 'cellrefineparameters', '0 1 1',
     "Parameters for cell refinement: crystal system index, key to refine cell in cellanddistort (0/1), key to refine distortions in cellanddistort (0/1)."),
    ('refdistortncycles', 'refdistortncycles', '25',
     "Maximum number of cycles for distortion refinement procedure"),
    ('referencecell', 'referencecell', '10.00000    10.00000    10.00000    90.00000    90.00000    90.00000 0',
     "Cell reference to be used in cell refinement. 7th parameter says if lattice scaling is allowed (0/1)."),
    ('intensitymethod', 'intensitymethod', 'profilesum 0',
     "Method for peak intensity determination: profilesum or profilefit. Second parameter: if 1, a two-pass integration is performed."),
    ('ignoreoverlaps', 'ignoreoverlaps', 'no',
     "Ignoring peak overlap during integration. If yes, only the peak with the smallest excitation error will be considered, the others will be ignored."),
    ('adjustreflbox', 'adjustreflbox', 'yes',
     "If yes, integration circle will be shifted to maximum intensity during intensity extraction. Useful for distorted data, dangerous for dense patterns."),
    ('resshellfraction', 'resshellfraction', '0.5000',
     "Limit between inner and outer shell for the calculation of average I/sigma (shown in the plot made during integration + in the file .framestats)."),
    ('saturationlimit', 'saturationlimit', '64000',
     "Above this number the pixel is considered saturated."),
    ('skipsaturated', 'skipsaturated', 'no',
     "If yes, reflections containing saturated pixels are not written in the output reflection list."),
    ('minrotaxisdist', 'minrotaxisdist', '0.0250',
     "minimum distance from rotation axis in reciprocal angstroms."),
    ('minreflpartiality', 'minreflpartiality', '0.0',
     "minimum reflection partiality in percentage."),
    ('maxsigmamult', 'maxsigmamult', '6.0',
     "maximum allowed multiple of sigma for Sg range."),
    ('rcshape', 'rcshape', 'pv',
     "Shape of the reflection profile. gaussian, pv (pseudovoigt: RCwidth Lorentzian, mosaicity Gaussian), lorentzian"),
    ('integrationmode', 'integrationmode', 'fit 1  0  3  1.000',
     "Integration settings: first the mode: fit, integrate, bestpattern or maximum. Second: framescaling (0 = no scaling/1 = optimize against reflection profile/2 = optimize against Rint). Third: Laue class number for frame scaling. Fourth: range for frame scale restraints. Fifth: weight of the frame scale restraint."),
    ('integrationparams', 'integrationparams', '0.0010  0.1000  1  1.00',
     "parameters of the profile - rocking curve width (rec. angstroms), mosaicity(deg.), minimum number of measurements per reflection, range of integration (as a multiple of the sigma of reflection profile)."),
    ('intkinematical', 'intkinematical', 'yes',
     "If yes, kinematical intensity integration will be performed in Finalize Integration."),
    ('intdynamical', 'intdynamical', 'yes',
     "If yes, dynamical intensity integration will be performed in Finalize Integration."),
    ('dynamicalscales', 'dynamicalscales', 'no',
     "If yes, dynamical intensities for dynamical refinement are divided by the frame scales (= corrected like the kinematical intensities)."),
    ('dynamicalerrormodel', 'dynamicalerrormodel', 'no',
     "If yes, sigma(I) for dynamical processing is modified by the error model coefficients (= the same treatment as sigmas in the kinematical processing)."),
    ('errormodel', 'errormodel', '1.0000    0.0000    0.0000 1',
     "Error model parameters. The first is s_fac, the second s_b, the third s_add.sigma^2=s_fac^2*(sigmacnt^2+s_b^2*int+s_add^2*int**2, with int the maximum intensity of symmetry related reflections. Then the refinement flag (1= refine error model)."),
    ('outliers', 'outliers', '1.50   1000.00',
     "Outlier thresholds. First: threshold for outliers in symmetry averaging. Second: threshold for profile fitting."),
    ('refinecamelparams', 'refinecamelparams', '1 1 0 1  0.0020',
     "Keys of the camel refinement. 1=apply, 0= don't apply: refine RC width, refine mosaicity, refine precession angle. Automatic/User defined camel plot step (0/1), camel plot step."),
    ('orientationparams', 'orientationparams', '1 1 0 1 -3 0 2 1',
     "Keys of the orientation optimization. 1=apply, 0= don't apply: opt. of tilt angles, opt. of frame centers, RC width, mosaicity, frame-by-frame distortions (binary code - 1st bit magnification, second elliptical, third parabolic), transfer average distortions to global distortions. The next key selects integrated intensities (=1) or uniform (=2), the last key sets the refinement of in-plane sigma"),
    ('simulationpower', 'simulationpower', '0.500',
     "An exponent applied to experimental and simulated counts in orientation refinement. Typically <1 to suppress overweighting of the strong reflections in the fit."),
    ('interpolationparams', 'interpolationparams', '1  4        0.00',
     "Keys for interpolation of orientation optimization values. First: mode of interpolation (0 = none,1 = polynomial, 2 = moving average). Second: order (polynomial order for poly fit, half-range formoving average). Third: level for outlier rejection (in sigma."),
    ('distcenterasoffset', 'distcenterasoffset', 'yes',
     "If yes, center of distortions is assumed to be a constant offset from the center of the diffraction pattern. Otherwise constant coordinates of the distortion centers are assumed."),
    ('distortunits', 'distortunits', 'percent',
     "Units in which distortions are displayed and stored. pixels or percent."),
    ('mapformat', 'mapformat', 'xplor no',
     "Format of the 3D map of reciprocal space. May be xplor, ccp4 or hdf5. Second argument: apply symmetry in the reconstruction?"),
    ('reconstructionparams', 'reconstructionparams', '0.0070    0.0140 0',
     "Parameters for the sections of rec. space. Pixel size (rec. angstroms) and thickness of the summed region perpendicular to the section. Switch to apply symmetry in the reconstructions."),
    ('removebackground', 'removebackground', 'yes yes',
     "If yes, background subtraction is done in reciprocal space reconstructions. If no, raw images are used.First argument: 2D reconstructions (rec. space sections), 2nd argument: 3D map reconstruction"),
    ('serialed', 'serialed', '2.000 1.000   180.00 0 0',
     "Settings for global orientation search by template matching. Three reals: step between templates (deg.), max resolution of templates (rec.A), maximum deviation from the original orientation. Then a switch to use reference cell (1) or current orientation matrix (0).Finally a switch to perform or not fine refinements of the result (0/1)."),
    ('virtualframes', 'virtualframes', '1   1   0 1',
     "Virtual frames - first the number of summed frames, second the step between frames, third 1 to sum all intensities, 0 only intensities on the virtual frame, -1 - decide based on geometry; fourth: determine the first and second number automatically"),
    ('composition', 'composition', 'C1',
     "The assumed composition of the compound. For now, used only to generate the output ins file."),
]

_ADVANCED_KEYWORD_TO_PLACEHOLDER = {kw: ph for kw, ph, _default, _comment in ADVANCED_FIELDS}

#%% persistence - last-used parameters, so the dialog can pre-fill next time

def load_pets2_defaults():
    """Load the last-used PETS2 dialog values (see ui_tabs/pets2_dialog.py),
    cached in `<repo>/temp/pets2_last_params.json`. Deliberately excludes
    per-study values (center, alpha start/step) - those are re-derived fresh
    (or left blank) on every open. Returns {} if no cache exists yet or it
    can't be read."""
    if not os.path.isfile(_CACHE_FILE):
        return {}
    try:
        with open(_CACHE_FILE, 'r') as f:
            return json.load(f)
    except Exception:
        return {}

def save_pets2_defaults(params):
    """Cache `params` (a plain dict of dialog values) to
    `<repo>/temp/pets2_last_params.json` for next time. Creates the temp/
    folder if needed."""
    os.makedirs(_CACHE_DIR, exist_ok=True)
    with open(_CACHE_FILE, 'w') as f:
        json.dump(params, f, indent=4)

#%% .pts2 generation

_PTS2_TEMPLATE = """#version of PETS that created this file
version 2.3.20260720.0745

#List of tasks to be automaticaly performed after the start. options:
# peak search, tilt axis, peak analysis, find cell, find known cell, refine cell, process frames, reflection profile, frame geometry, finalize integration, generate 3d map, sections, check files, global indexing, set, quit, break, none,
autotask
{autotask_block}
endautotask

#If yes, list of tasks to be automaticaly performed after the start is not cleared when all tasks are finished.
keepautotasks {keepautotasks}

#wavelength in angstroms
lambda  {lam:.8f}

#image calibration in reciprocal angstroms/pixel
aperpixel  {aperpixel:.8f}

#detector type. Options: default, asi, olympus, cetad. More to come...
detector {detector}

#Parameters allowing to calculate the sigma(count) on every pixel. The first is G*gamma (gain*cascade factor), the second is sigma**2 of the dark image.
noiseparameters      {noiseparameters}

#Exposure time per frame in seconds. Used by the detector non-linearity correction.
exposureperframe    {exposure:.6f}

#Detector non-linearity correction: yes/no, followed by the coefficient k in c(corr)=c(orig)*(1+k*c(orig)), with k in 1/(counts/sec).
nonlinearity {nonlinearity}

#experimental geometry. Options: continuous,precession,static
geometry {geometry}

#precession or integration (semi)angle. Must be zero for static geometry.
phi  {phi:.5f}

#angle between the positive horizontal axis on the image and the projection of the tilt axis on the image. Two keys define, if global search and local refinement should be performed.
omega   {omega:.4f} {omega_global_search:d} {omega_local_refine:d}

#angle between the positive main rotation axis and the x-y plane (perpendicular to the beam).Positive, if the angle between the axis and the beam is smaller than 90 degrees.The key defines, if delta shpould (1) or shoul not (0) be refined.
delta    {delta}

#size of the pixel in the 3D map of reciprocal space.
pixelsize    {pixelsize}

#binning of the images after reading in. Should be an even number.
bin  {binning}

#Diameter of integration circle of the reflections (pixels, unbinned)
reflectionsize  {reflectionsize}

#maximum dstar (rec.angstroms) for integration
dstarmax  {dstarmax}

#maximum dstar (rec.angstroms) for peaksearch
dstarmaxps  {dstarmaxps}

#minimum dstar (rec.angstroms) for integration and peaksearch
dstarmin  {dstarmin}

#excluded region around the central beam. Obsolete, partly overlapping with dstarmin. Will be removed/replaced soon.
centerradius  {centerradius}

#beamstop settings. beamstop no with no beamstop. "beamstop followed by a list of beamstop points, if beamstop is used.
beamstop {beamstop}

#list of coordinates of bad pixels. These pixels are ignored and their value  is interpolated from the values of the neighboring pixels.
badpixels
endbadpixels

#apply interpolation of values at bad pixels
correctbadpixels {correctbadpixels}

avoidicerings {avoidicerings}

#list of ice ring diameters. Peaks in the vicinity of these rings are ignored, if avoidicerings is yes.
icerings
 0.2735
 0.4446
endicerings

#Peak search mode - auto (default) = find automatically, manual = click in the picture to add or remove found peaks.
peaksearchmode {peaksearchmode}

#center of the diffraction pattern. Options: AUTO=find center automatically; two numbers x y - position of the pattern center on the first frame in pixels.
center {center_line}

#Mode of center determination. centralbeam = use central beam, friedelpairs = use friedel pairs, useold = take whatever was previously set. Optional second argument: always accept center? (0/1). Applied only to Friedel pairs.
centermode {centermode}

#Intensity/sigma for peak search and for the calculation of the camel plot
i/sigma    {i_sigma}

#Option for average peak profile calculation in the fit profile. If yes, each group of frames will have its own calculated average peak profile.
moreaveprofiles {moreaveprofiles}

#Parameters for average peak profile (APP) calculation. First: minimum number of peaks in one APP. Second: minimum I/sigma peaks (0=fixed/1=auto). Third: value of minimum I/sigma for the case fixed.
peakprofileparams  {peakprofileparams}

#Parameters for specifying image sectors for APP calculation. First: distinguish image sectors (0/1). Second: radius of the central sector (0=fixed/1=auto). Third: radius value for the case fixed .
peakprofilesectors  {peakprofilesectors}

#Constant background to be subtracted from all images after reading in. May be negative if needed. Second optional argument is the factor that multiplies the raw image at reading
background       {background}

#Method for background subtraction. median = median filter (default, the best), average = average in concentric circles, fft = background by Gaussian smearing. Median should be used in normal circumstances.
backgroundmethod {backgroundmethod}

#Peak analysis options: estimate cluster limits automatically (0) or use preset values (1), process automatically (1) or wait for confirmation (0), xyz cluster limit (in pixels), 3D cluster limit (in rec. Angstroms).
peakanalysis {peakanalysis}

#Minimum number of points in a cluster (used during peak analysis).
minclusterpoints   {minclusterpoints}

#Display rfft file in indexing panel.
showrfft {showrfft}

#Algorithm for automatic cell determination: diffspace = algorithm using difference space clustering, triplets = old "jana-style" algorithm, fft = new algorithm using fourier transform of xyz file.
indexingmode {indexingmode}

#Parameters for cell determination: indexing tolerance (rec. A), angular tolerance for rec. direction (deg), maximal primitive volume (rec. A^3), maximal cell length (A), relative length tolerance (%), angular tolerance (deg), maximum Sg for indexing (distortions only).
indexingparameters  {indexingparameters}

#Automatic determination of indexing tolerance. If yes, the indexing tolerance is calculated from the  maximal cell length as 0.45/(maximal cell length).
automaxddiff {automaxddiff}

#Method for find known cell: refldir = method for determining orientation by rotating around the directions of reflections.spacedir = method for determining orientation by testing all space directions.
knowncellmethod {knowncellmethod}

#Known cell parameters for orientation search.
knowncell    {knowncell}

#Parameters for searching for a known cell: angular rotation step (deg), number of test reflections for indexing at each position.
knowncellparameters  {knowncellparameters}

#maximum satellite index used in integration. (Applies only to modulated structures.)
maxindex  {maxindex}

#Algorithm for cell refinement. cellandub = refinement from 3D peak coordinates (standard), cellfromd = refinement against lengths of the diffraction vectors, cellanddistort = cell from 2D peak positions plus, optionally, distortion parameters.
cellrefinemode {cellrefinemode}

#Indexing tolerance (rec. A) for refine cell.
maxddiffrefine  {maxddiffrefine}

#Automatic determination of indexing tolerance in refine cell. If yes, the indexing tolerance is calculated from the maximal cell length actual or maximal fixed cell length as 0.3/(maximal cell length).
automaxddiffref {automaxddiffref}

#Parameters for cell refinement: crystal system index, key to refine cell in cellanddistort (0/1), key to refine distortions in cellanddistort (0/1).
cellrefineparameters  {cellrefineparameters}

#Maximum number of cycles for distortion refinement procedure
refdistortncycles {refdistortncycles}

#Cell reference to be used in cell refinement. 7th parameter says if lattice scaling is allowed (0/1).
referencecell    {referencecell}

#Method for peak intensity determination: profilesum or profilefit. Second parameter: if 1, a two-pass integration is performed.
intensitymethod {intensitymethod}

#Ignoring peak overlap during integration. If yes, only the peak with the smallest excitation error will be considered, the others will be ignored.
ignoreoverlaps {ignoreoverlaps}

#If yes, integration circle will be shifted to maximum intensity during intensity extraction. Useful for distorted data, dangerous for dense patterns.
adjustreflbox {adjustreflbox}

#Limit between inner and outer shell for the calculation of average I/sigma (shown in the plot made during integration + in the file .framestats).
resshellfraction  {resshellfraction}

#Above this number the pixel is considered saturated.
saturationlimit      {saturationlimit}

#If yes, reflections containing saturated pixels are not written in the output reflection list.
skipsaturated {skipsaturated}

#minimum distance from rotation axis in reciprocal angstroms.
minrotaxisdist  {minrotaxisdist}

#minimum reflection partiality in percentage.
minreflpartiality   {minreflpartiality}

#maximum allowed multiple of sigma for Sg range.
maxsigmamult   {maxsigmamult}

#Shape of the reflection profile. gaussian, pv (pseudovoigt: RCwidth Lorentzian, mosaicity Gaussian), lorentzian
rcshape {rcshape}

#Integration settings: first the mode: fit, integrate, bestpattern or maximum. Second: framescaling (0 = no scaling/1 = optimize against reflection profile/2 = optimize against Rint). Third: Laue class number for frame scaling. Fourth: range for frame scale restraints. Fifth: weight of the frame scale restraint.
integrationmode {integrationmode}

#parameters of the profile - rocking curve width (rec. angstroms), mosaicity(deg.), minimum number of measurements per reflection, range of integration (as a multiple of the sigma of reflection profile).
integrationparams  {integrationparams}

#If yes, kinematical intensity integration will be performed in Finalize Integration.
intkinematical {intkinematical}

#If yes, dynamical intensity integration will be performed in Finalize Integration.
intdynamical {intdynamical}

#If yes, dynamical intensities for dynamical refinement are divided by the frame scales (= corrected like the kinematical intensities).
dynamicalscales {dynamicalscales}

#If yes, sigma(I) for dynamical processing is modified by the error model coefficients (= the same treatment as sigmas in the kinematical processing).
dynamicalerrormodel {dynamicalerrormodel}

#Error model parameters. The first is s_fac, the second s_b, the third s_add.sigma^2=s_fac^2*(sigmacnt^2+s_b^2*int+s_add^2*int**2, with int the maximum intensity of symmetry related reflections. Then the refinement flag (1= refine error model).
errormodel    {errormodel}

#Outlier thresholds. First: threshold for outliers in symmetry averaging. Second: threshold for profile fitting.
outliers      {outliers}

#Keys of the camel refinement. 1=apply, 0= don't apply: refine RC width, refine mosaicity, refine precession angle. Automatic/User defined camel plot step (0/1), camel plot step.
refinecamelparams {refinecamelparams}

#Keys of the orientation optimization. 1=apply, 0= don't apply: opt. of tilt angles, opt. of frame centers, RC width, mosaicity, frame-by-frame distortions (binary code - 1st bit magnification, second elliptical, third parabolic), transfer average distortions to global distortions. The next key selects integrated intensities (=1) or uniform (=2), the last key sets the refinement of in-plane sigma
orientationparams {orientationparams}

#An exponent applied to experimental and simulated counts in orientation refinement. Typically <1 to suppress overweighting of the strong reflections in the fit.
simulationpower   {simulationpower}

#Keys for interpolation of orientation optimization values. First: mode of interpolation (0 = none,1 = polynomial, 2 = moving average). Second: order (polynomial order for poly fit, half-range formoving average). Third: level for outlier rejection (in sigma.
interpolationparams  {interpolationparams}

#If yes, center of distortions is assumed to be a constant offset from the center of the diffraction pattern. Otherwise constant coordinates of the distortion centers are assumed.
distcenterasoffset {distcenterasoffset}

#Units in which distortions are displayed and stored. pixels or percent.
distortunits {distortunits}

#Distortion parameters.
distortions
      0.0000      0.0000      0.0000      0.0000
    0.000000    0.000000    0.000000    0.000000
    0.000000    0.000000    0.000000    0.000000
    0.000000    0.000000    0.000000    0.000000
    0.000000    0.000000    0.000000    0.000000
    0.000000    0.000000    0.000000    0.000000
    0.000000    0.000000    0.000000  -45.000000
    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000
enddistortions

#Refinement keys for the distortion parameters.
distortionskeys
  0  0  0  0
  0  0  1  0
  0  0  0  0
  1  0  0  1
  0  0  1  0
  0  0  0  0
  2  0  0  2
  0  0  0  0  0  0
enddistortionskeys

#Format of the 3D map of reciprocal space. May be xplor, ccp4 or hdf5. Second argument: apply symmetry in the reconstruction?
mapformat {mapformat}

Sections of reciprocal space. Each line contains name, first vector (3 coords), second vector (3 coords) and origin (3 coords).
reconstruction
endreconstruction

#Parameters for the sections of rec. space. Pixel size (rec. angstroms) and thickness of the summed region perpendicular to the section. Switch to apply symmetry in the reconstructions.
reconstructionparams    {reconstructionparams}

#If yes, background subtraction is done in reciprocal space reconstructions. If no, raw images are used.First argument: 2D reconstructions (rec. space sections), 2nd argument: 3D map reconstruction
removebackground {removebackground}

#Settings for global orientation search by template matching. Three reals: step between templates (deg.), max resolution of templates (rec.A), maximum deviation from the original orientation. Then a switch to use reference cell (1) or current orientation matrix (0).Finally a switch to perform or not fine refinements of the result (0/1).
serialed {serialed}

#Virtual frames - first the number of summed frames, second the step between frames, third 1 to sum all intensities, 0 only intensities on the virtual frame, -1 - decide based on geometry; fourth: determine the first and second number automatically
virtualframes   {virtualframes}

#List of entries to be propagated further to the structure analysis pipeline. May contain experimental info, PETS augments it with dtaa-processing related entries.
cifentries
endcifentries

#The assumed composition of the compound. For now, used only to generate the output ins file.
composition {composition}

# Order of items in each line of the image list
imagelistheader imgname alpha beta domega alphaorig betaorig domegaorig xcenter ycenter intscale diffbfac magcorr elliamp elliph paraamp paraph useforcalc dataset

#List of images
imagelist
{imagelist}
endimagelist

#Project notes (optional, may be multi-line, maximum length 5000 characters)
notes
{notes_block}
endnotes

#Orientation matrix and cell parameters
celllist
cellItem active
name Cell 1
ubmatrix
  0.100000  0.000000  0.000000
  0.000000  0.100000  0.000000
  0.000000  0.000000  0.100000
celldir
  1.000000  0.000000  0.000000
  0.000000  1.000000  0.000000
  0.000000  0.000000  1.000000
cell   10.0000   10.0000   10.0000   90.000   90.000   90.000
cellvol   1000.00
cellsu    0.0000    0.0000    0.0000    0.000    0.000    0.000
cellSymmetry       a
cellCentering       P
cellHistoryCodes  0  0  0  0
cellHistoryParams      0.000      0.000      0.000      0.000      0.000      0.000      0.000      0.000      0.000      0.000      0.000
endCellItem
endcelllist
"""


def write_pts2(fn, n_frames, aperpixel, voltage_kv, geometry='continuous', omega=0.0,
               omega_global_search=1, omega_local_refine=1,
               phi=0.0, exposure_s=0.0, detector='default', tilt_start_deg=0.0,
               tilt_step_deg=0.0, binning=1, reflectionsize=8.0, dstarmax=1.4,
               dstarmaxps=1.4, dstarmin=0.05, center=None, autotask=None,
               keepautotasks='no', advanced_overrides=None, frame_shape=None,
               roi_id=None):
    """Write a PETS2 project file (.pts2) ready to open directly in PETS,
    pointing at a sibling `frames/%04d.tif` stack (as already exported by
    `create_frames`).

    Args:
        fn: Output .pts2 file path.
        n_frames: Number of frames in the tilt series (imagelist length).
        aperpixel: Reciprocal-space calibration, in 1/Angstrom per pixel.
        voltage_kv: Acceleration voltage in kV - converted to `lambda` via
            `electron_wavelength_angstrom`.
        geometry: 'continuous', 'precession', or 'static'.
        omega: Angle (deg) between the horizontal image axis and the tilt
            axis projection.
        omega_global_search: 1 to have PETS perform a global search for omega
            on open, 0 to skip it (defaults to on - the recommended starting
            point for a fresh study). Written as the first key after omega.
        omega_local_refine: 1 to have PETS locally refine omega around the
            found/entered value, 0 to skip it. Written as the second key
            after omega.
        phi: Precession/integration semi-angle (deg) - must be 0 for static
            geometry.
        exposure_s: Exposure time per frame, in seconds.
        detector: PETS detector type ('default', 'asi', 'olympus', 'cetad').
        tilt_start_deg: Goniometer tilt angle (deg) at frame 1.
        tilt_step_deg: Tilt increment (deg) between consecutive frames.
        binning, reflectionsize, dstarmax, dstarmaxps, dstarmin: see
            BASIC_FIELDS for the matching PETS comment/meaning.
        center: (x, y) pixel position to seed PETS' center search, or None
            for `center AUTO` (auto-detect on the first frame).
        autotask: Ordered list of task names (see AUTOTASK_OPTIONS) to run
            automatically when PETS opens the project, or None/[] for none.
        keepautotasks: 'yes'/'no' - whether the autotask list survives after
            all listed tasks finish.
        advanced_overrides: Optional {pets_keyword: raw_value_string} dict
            overriding any entry in ADVANCED_FIELDS (unlisted keywords keep
            their ADVANCED_FIELDS default).
        frame_shape: (height, width) of each frame in pixels, used only when
            `center` is None to seed PETS' own frame-center-refinement.
        roi_id: ROI/object number this tilt series belongs to - written into
            the project notes so it stays traceable once the file is opened
            standalone in PETS.
    """
    lam = electron_wavelength_angstrom(voltage_kv)

    if center is not None:
        center_line = f'{center[0]:.4f} {center[1]:.4f}'
    else:
        center_line = 'AUTO'

    autotask_block = '\n'.join(autotask) if autotask else ''

    notes_block = f'ROI No {roi_id}' if roi_id is not None else ''

    # Only imgname + alpha are meaningful before PETS has processed anything
    # - the remaining imagelistheader columns (beta, domega, centers,
    # intensity scaling, distortions...) are left blank for PETS to fill in.
    rows = []
    for i in range(n_frames):
        angle = tilt_start_deg + i * tilt_step_deg
        rows.append(f'"frames/{i + 1:04d}.tif"  {angle:.4f}')

    fmt = {ph: default for _kw, ph, default, _comment in ADVANCED_FIELDS}
    if advanced_overrides:
        for kw, val in advanced_overrides.items():
            ph = _ADVANCED_KEYWORD_TO_PLACEHOLDER.get(kw)
            if ph is not None:
                fmt[ph] = val

    content = _PTS2_TEMPLATE.format(
        lam=lam, aperpixel=aperpixel, detector=detector, exposure=exposure_s,
        geometry=geometry, phi=phi, omega=omega,
        omega_global_search=int(omega_global_search), omega_local_refine=int(omega_local_refine),
        binning=binning,
        reflectionsize=reflectionsize, dstarmax=dstarmax, dstarmaxps=dstarmaxps,
        dstarmin=dstarmin, center_line=center_line, autotask_block=autotask_block,
        keepautotasks=keepautotasks, imagelist='\n'.join(rows), notes_block=notes_block,
        **fmt,
    )
    with open(fn, 'w') as f:
        f.write(content)
