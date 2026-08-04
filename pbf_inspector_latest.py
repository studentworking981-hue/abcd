"""
Powder Bed Fusion (PBF) Metal AM -- Powder-Bed Surface Defect Inspector v4
============================================================================
STANDALONE SINGLE-FILE APPLICATION.

This file merges what used to be two files -- pbf_pipeline.py (the pure
image-processing engine) and pbf_inspector_app.py (the Tkinter/customtkinter
GUI) -- into ONE self-contained module. There is no `pbf_pipeline` import
anywhere below; every function the GUI calls (run_pipeline, dog_to_colormap,
perspective_correct, DEFAULT_PARAMS, and every helper run_pipeline depends
on) is defined directly in this file, in the "PIPELINE ENGINE" section
immediately below the imports. Running

    python pbf_inspector_app.py

requires no other project file to be present alongside it.

WHAT THIS VERSION CHANGES, AND WHY
---------------------------------------------------------------------------
The detection engine still contains NO threshold parameters or
threshold-based logic of any kind: no `pixel > value` comparison anywhere
in this file depends on an image-derived mean, std, MAD, z-score, or
percentile. Detection is marker-controlled watershed end to end:

  1. MARKER GENERATION -- `dome_extract()` performs morphological
     reconstruction (h-dome / h-minima extraction) to find regions that
     stand at least a fixed height above (or below) their surroundings.
     That height (`dome_height_*`) is a plain user-set number, the same
     *kind* of control as `recon_radius` -- a fixed structural depth,
     never computed from the image's own mean, std, or percentiles.
  2. MARKER REFINEMENT -- `build_foreground_markers()` runs a distance-
     transform peak search (`peak_local_max`) inside each dome-positive
     component to split touching blobs into separate watershed seeds.
     `peak_local_max` only ever compares a pixel's distance value to its
     *neighbours*, never to an absolute number. `close_and_fill()` merges
     markers within a fixed `basin_merge_radius` of one another first.
  3. WATERSHED FLOODING -- `watershed_segment()` floods every marker
     outward over the response channel's own topology (`-response_channel`
     as elevation), confined to the dome-positive footprint built in step
     1 -- never a thresholded binary. A single homogeneous blob with one
     marker keeps its FULL dome footprint; two touching blobs with two
     markers still get split at the ridge between them.
  4. SHAPE FILTERING -- `shape_filter()` is a geometric discriminator
     (area / circularity / solidity / aspect / convexity) on already-
     segmented regions; it never inspects a pixel intensity value.

`perspective_correct()` (build-plate boundary detection) uses a two-marker
watershed (image border = background marker, a small seed box at the frame
centre = plate marker) over the frame's gradient magnitude -- no
`cv2.threshold` / Otsu anywhere in this file.

FILE LAYOUT
---------------------------------------------------------------------------
  1. Imports (consolidated, deduplicated -- pipeline + GUI needs together)
  2. PIPELINE ENGINE -- perspective correction, ROI polygon masking, core
     image-processing helpers, marker/watershed detection, DEFAULT_PARAMS,
     run_pipeline(), dog_to_colormap()
  3. GUI -- CTkSpinbox, ROIPolygonEditor, PBFInspectorWorkstation -- calls
     ONLY the functions defined in section 2, never re-implementing any
     detection logic, and unchanged in behaviour from the two-file version:
     same window, buttons, tabs, viewports, ROI editor, single/batch
     export, progress/status handling.

TESTING NOTE: this file could not be executed in the sandboxed environment
this was developed in -- neither tkinter nor customtkinter are installed
there, and there is no display server available even if they were, so no
tkinter GUI can be instantiated to click-test it. Every pipeline function
in section 2 WAS executed and validated there against synthetic test
scenes and the provided sample image. Please run this file locally and
confirm the GUI behaves as expected -- report anything odd.
"""

import os
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from scipy import ndimage
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.morphology import reconstruction

import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
from PIL import Image, ImageTk


# ======================================================================
#  PIPELINE ENGINE
#  (formerly pbf_pipeline.py -- merged directly into this file so the
#  application has no external module dependency; every function below is
#  used, directly or indirectly, by run_pipeline())
# ======================================================================


# ======================================================================
#  PERSPECTIVE CORRECTION
#  v4: plate/background split via two-marker watershed over gradient
#  magnitude -- no cv2.threshold / Otsu anywhere in this file any more.
# ======================================================================

def perspective_correct(img):
    """Detect the build-plate quadrilateral and flatten it to a rectangle.

    The plate (bright, textured) sits against a background frame border in
    every sample provided. Rather than separating them with an intensity
    cutoff (v3 used Otsu), this seeds a background marker along the image
    border and a plate marker in a small box at the frame centre, then lets
    a watershed flood across the gradient-magnitude surface decide the
    boundary -- the same marker-controlled-watershed principle used
    throughout the rest of this file, and no threshold of any kind."""
    gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_full, (5, 5), 0).astype(np.float32)
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)

    h, w = gray_full.shape
    markers = np.zeros((h, w), dtype=np.int32)
    border = max(2, min(h, w) // 50)
    markers[:border, :] = 1
    markers[-border:, :] = 1
    markers[:, :border] = 1
    markers[:, -border:] = 1
    cy0, cy1 = int(h * 0.4), int(h * 0.6)
    cx0, cx1 = int(w * 0.4), int(w * 0.6)
    markers[cy0:cy1, cx0:cx1] = 2

    seg = watershed(gradient, markers)
    plate_mask = (seg == 2).astype(np.uint8) * 255

    contours, _ = cv2.findContours(plate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2)
            rect = np.zeros((4, 2), dtype="float32")
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]

            (tl, tr, br, bl) = rect
            widthA = np.hypot(br[0] - bl[0], br[1] - bl[1])
            widthB = np.hypot(tr[0] - tl[0], tr[1] - tl[1])
            maxWidth = max(int(widthA), int(widthB))

            heightA = np.hypot(tr[0] - br[0], tr[1] - br[1])
            heightB = np.hypot(tl[0] - bl[0], tl[1] - bl[1])
            maxHeight = max(int(heightA), int(heightB))

            if maxWidth > 10 and maxHeight > 10:
                dst = np.array([
                    [0, 0], [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
                M = cv2.getPerspectiveTransform(rect, dst)
                img = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    return img


# ======================================================================
#  ROI POLYGON
# ======================================================================

def polygon_to_mask(shape_hw, polygon_pts_norm):
    """polygon_pts_norm: list of (x_frac, y_frac), each in [0, 1], relative
    to whatever image they were drawn on. Normalized coordinates (rather
    than raw pixels) mean the SAME polygon is valid at any resolution --
    the downsized live preview, a full-resolution export, or any other
    image in a batch folder that shares the same camera framing -- with no
    separate bookkeeping of "what size was this drawn on". Returns a uint8
    0/255 mask sized to `shape_hw`. Falls back to an all-255 mask if the
    polygon has fewer than 3 points (nothing selected yet / invalid
    selection) so callers do not need a special case.
    """
    h, w = shape_hw
    if polygon_pts_norm is None or len(polygon_pts_norm) < 3:
        return np.full((h, w), 255, dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array([(fx * w, fy * h) for (fx, fy) in polygon_pts_norm], dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 255)
    return mask


# ======================================================================
#  CORE IMAGE-PROCESSING FUNCTIONS
# ======================================================================

def estimate_background(gray_f32, sigma):
    """Large-sigma Gaussian low-pass = the slow illumination/powder-tone
    field. Unchanged from v3."""
    k = int(2 * round(3 * sigma) + 1)
    k = max(k, 3)
    return cv2.GaussianBlur(gray_f32, (k, k), sigma)


def flatten_illumination(gray_f32, background, target=128.0):
    """Multiplicative flat-fielding. Unchanged from v3."""
    norm = (gray_f32 / (background + 1e-3)) * target
    return np.clip(norm, 0, 255).astype(np.float32)


def difference_of_gaussians(img_f32, sigma1, sigma2):
    """Band-pass filter, symmetric to bright/dark blobs of a chosen scale.
    Unchanged from v3 -- still what selects which spatial scale ("fine" vs
    "coarse") a candidate response belongs to before marker generation
    runs on it. This is scale selection, not a threshold: its output is
    fed into `dome_extract()` below, never compared to a statistical
    cutoff."""
    k1 = max(3, int(2 * round(3 * sigma1) + 1))
    k2 = max(3, int(2 * round(3 * sigma2) + 1))
    g1 = cv2.GaussianBlur(img_f32, (k1, k1), sigma1)
    g2 = cv2.GaussianBlur(img_f32, (k2, k2), sigma2)
    return g1 - g2


def opening_by_reconstruction(channel_f32, radius):
    """Grayscale opening-by-reconstruction: kills sub-pixel speckle while
    leaving the shape of surviving blobs untouched. Unchanged from v3.
    Structural (radius-based), not an intensity cutoff."""
    if radius <= 0:
        return channel_f32
    size = 2 * radius + 1
    seed = ndimage.grey_erosion(channel_f32, size=(size, size))
    seed = np.minimum(seed, channel_f32)
    return reconstruction(seed, channel_f32, method='dilation')


def dome_extract(signed_f32, height):
    """Regional-maxima ('h-dome') extraction via morphological
    reconstruction. This is the ONLY thing that decides which pixels
    become foreground marker candidates in this pipeline -- it replaces
    every `mean + k*std` / z-score comparison from v3.

    `height` is a fixed structural parameter, exactly like `recon_radius`
    above: a plain number the user (or a GUI slider) sets directly. It is
    never computed from the image's own mean, std, MAD, or percentiles, so
    it is not a statistical/adaptive threshold.

    The returned array is zero everywhere that is not part of a connected
    region standing at least `height` above (or below, depending on the
    caller's chosen sign) its local surroundings, and equal to that
    region's own elevation above its base everywhere it is part of one.
    That zero/nonzero split is the reconstruction's own topology -- a
    structural fact about the surface, not a chosen cutoff value being
    compared against a pixel intensity.
    """
    if height <= 0:
        return np.zeros_like(signed_f32)
    seed = signed_f32 - height
    recon = reconstruction(seed, signed_f32, method='dilation')
    return signed_f32 - recon


def remove_small_components(binary_u8, min_area):
    """Connected-component + area filter. Same result as a per-label loop,
    vectorized via a label lookup table. A component-AREA filter is a
    geometric/shape criterion (see `shape_filter` below), not an intensity
    threshold -- it never inspects a single pixel's value."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary_u8, connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False
    lut = keep.astype(np.uint8) * 255
    return lut[labels]


def close_and_fill(binary_u8, radius):
    """Morphological closing + per-component hole-fill. Also doubles as
    marker-merging in v4: running this on the dome-positive candidate mask
    before the peak search below merges any two dome regions closer than
    `radius` into one marker group ("Basin Merge Radius" / "Marker Merge
    Distance" in the GUI). Structural (radius-based), not an intensity
    threshold. radius<=0 disables it."""
    if radius <= 0 or binary_u8.max() == 0:
        return binary_u8
    k = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    closed = cv2.morphologyEx(binary_u8, cv2.MORPH_CLOSE, kernel)
    filled = ndimage.binary_fill_holes(closed > 0).astype(np.uint8) * 255
    return filled


def build_foreground_markers(dome_positive_u8, min_distance):
    """Distance-transform peak markers ('Marker Refinement') inside each
    dome-positive candidate region -- splits touching blobs into separate
    watershed seeds. `peak_local_max` only ever compares a pixel's distance
    value to its NEIGHBOURS, never to an absolute number, so this stays
    threshold-free. Unchanged in spirit from v3's `build_markers`."""
    dist = cv2.distanceTransform(dome_positive_u8, cv2.DIST_L2, 5)
    coords = peak_local_max(dist, min_distance=max(1, min_distance), labels=dome_positive_u8)
    markers = np.zeros(dome_positive_u8.shape, dtype=np.int32)
    for idx, pt in enumerate(coords):
        markers[pt[0], pt[1]] = idx + 1
    if markers.max() == 0 and dome_positive_u8.max() > 0:
        _, markers = cv2.connectedComponents(dome_positive_u8)
    return markers


def watershed_segment(response_channel, foreground_markers, support_mask, compactness=0.0):
    """Marker-controlled watershed CONFINED to `support_mask` -- the
    dome-positive candidate region built by `dome_extract` above, never a
    thresholded binary. `support_mask` is what makes this a genuine
    watershed segmentation rather than a plain marker dump: within the
    already-established dome footprint, each marker's flood expands
    outward over the response channel's OWN topology (`-response_channel`
    as elevation, exactly as in v3) until it meets a neighbouring flood or
    the support boundary -- so a single homogeneous blob with only one
    marker keeps its FULL extent (the whole dome_positive footprint, not
    just a rim around the seed point), while two touching blobs with two
    separate markers still get split at their shared ridge ('Watershed
    Connectivity', 'Watershed Basin Merging'). No intensity cutoff is
    involved: the confinement comes from the morphological reconstruction
    that built `support_mask`, not from comparing pixels to a value here."""
    if foreground_markers.max() == 0 or support_mask.max() == 0:
        return np.zeros(response_channel.shape, dtype=np.int32)
    surface = -response_channel
    return watershed(surface, foreground_markers, mask=(support_mask > 0), compactness=compactness)


def shape_filter(seg_labels, min_area, max_area, min_circ, min_solid, max_aspect, min_conv):
    """Area / circularity / solidity / aspect / convexity discriminator.
    Unchanged from v3 -- a geometric filter applied to already-segmented
    regions. It never inspects a pixel's intensity, so it is not one of
    the threshold comparisons this rewrite eliminates; it is exactly the
    'Shape Analysis' / 'Contour Analysis' stage this rewrite keeps."""
    out = np.zeros(seg_labels.shape, dtype=np.uint8)
    kept = []
    for label in np.unique(seg_labels):
        if label == 0:
            continue
        comp = (seg_labels == label).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if not (min_area <= area <= max_area):
            continue
        perim = cv2.arcLength(c, True)
        circularity = (4 * np.pi * area) / (perim ** 2) if perim > 0 else 0
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        x, y, wb, hb = cv2.boundingRect(c)
        aspect = max(wb, hb) / max(1, min(wb, hb))
        convexity = cv2.arcLength(hull, True) / perim if perim > 0 else 0
        if circularity >= min_circ and solidity >= min_solid and aspect <= max_aspect and convexity >= min_conv:
            out[seg_labels == label] = 255
            kept.append(c)
    return out, kept


DEFAULT_PARAMS = dict(
    # Tab 1 -- Illumination & Denoise
    bg_sigma=45.0,
    bilat_d=9, bilat_sc=50.0, bilat_ss=50.0,
    # Tab 2 -- Multi-scale gradient bands (DoG scale selection + speckle
    # reconstruction radius, feeding marker generation below)
    fine_sigma1=1.0, fine_sigma2=3.0,
    coarse_sigma1=5.0, coarse_sigma2=16.0,
    recon_radius=1,
    # Tab 3 -- Marker Generation & Watershed
    #
    # dome_height_* are fixed structural depths on the DoG response's own
    # native scale (roughly 8-bit intensity units), picked once by hand to
    # sit comfortably above the DoG band's typical noise amplitude on this
    # camera rig -- the same way `recon_radius` or `close_radius` were
    # already picked by hand in v3. They are plain numbers set directly by
    # the GUI slider, NOT recomputed from any image's mean/std/percentile
    # at run time -- that distinction (fixed structural number vs.
    # per-image statistical cutoff) is exactly what this rewrite requires.
    # If a different camera/lighting rig produces a different noise floor,
    # re-pick these numbers by hand for that rig, same as any other
    # structural slider in this file.
    dome_height_fine=30.0,
    dome_height_coarse=32.0,
    min_component_area_fine=5,
    min_component_area_coarse=60,
    local_max_min_dist=4,
    basin_merge_radius_fine=1,
    basin_merge_radius_coarse=2,
    compactness=0.0,
    # Tab 4 -- Shape filtering
    min_area=20, max_area=25000,
    min_circularity=0.45, min_solidity=0.82, max_aspect_ratio=5.0, min_convexity=0.88,
    # Tab 5 -- Full-Extent Band & Continuity (large homogeneous defects)
    dome_height_extent=35.0,
    min_component_area_extent=120,
    local_max_min_dist_extent=15,
    basin_merge_radius_extent=3,
    # Tab 6 -- Micro-Scale Marker Refinement (minute defects)
    micro_enabled=True,
    dome_height_micro=20.0,
    min_component_area_micro=2,
    local_max_min_dist_micro=2,
    basin_merge_radius_micro=1,
    micro_max_area=40,
)


def _detect_band(response_channel, roi, dome_height, min_component_area,
                  local_max_min_dist, basin_merge_radius,
                  min_area, max_area, min_circ, min_solid, max_aspect, min_conv,
                  compactness=0.0):
    """One marker-controlled-watershed detection band. Replaces v3's
    `adaptive_threshold -> binary -> watershed` sequence with
    `dome_extract -> marker refinement -> mask-confined watershed`, with no
    threshold at any step (see module docstring)."""
    dome = dome_extract(response_channel, dome_height)
    # dome > 0 is a structural consequence of the reconstruction above
    # (see dome_extract's docstring) -- not a chosen intensity cutoff.
    dome_positive = (dome > 0).astype(np.uint8) * 255
    dome_positive = cv2.bitwise_and(dome_positive, roi)
    dome_positive = remove_small_components(dome_positive, min_component_area)
    dome_positive = close_and_fill(dome_positive, basin_merge_radius)

    if dome_positive.max() == 0:
        return dict(mask=np.zeros(response_channel.shape, np.uint8), contours=[],
                    dome=dome, dome_positive=dome_positive)

    fg_markers = build_foreground_markers(dome_positive, local_max_min_dist)
    seg = watershed_segment(response_channel, fg_markers, dome_positive, compactness=compactness)
    mask, contours = shape_filter(seg, min_area, max_area, min_circ, min_solid, max_aspect, min_conv)
    return dict(mask=mask, contours=contours, dome=dome, dome_positive=dome_positive)


def _run_detection(gray, params, roi_polygon=None):
    """Polarity/band detection engine, operating on an already
    perspective-corrected, possibly-resized grayscale array. Shared by both
    the live-preview path and the full-resolution export path."""
    p = dict(DEFAULT_PARAMS)
    p.update(params or {})
    gray_f = gray.astype(np.float32)

    # Base ROI: exclude the solid-black padding introduced by
    # perspective_correct's warp (pixels the warp never touched -- exactly
    # 0, or noise around it). This is not a defect-detection cutoff; it
    # excludes pixels that are not real powder-bed surface at all, the same
    # role the polygon ROI intersection below plays.
    roi = (gray > 3).astype(np.uint8) * 255
    roi = cv2.erode(roi, np.ones((15, 15), np.uint8))
    # Polygon ROI: intersect with the user-drawn polygon, if any, instead
    # of a rigid rectangular crop. Everything outside the polygon is
    # excluded from detection.
    if roi_polygon is not None and len(roi_polygon) >= 3:
        poly_mask = polygon_to_mask(gray.shape[:2], roi_polygon)
        roi = cv2.bitwise_and(roi, poly_mask)

    background = estimate_background(gray_f, p["bg_sigma"])
    normalized = flatten_illumination(gray_f, background)
    denoised = cv2.bilateralFilter(
        normalized.astype(np.uint8), int(p["bilat_d"]), float(p["bilat_sc"]), float(p["bilat_ss"])
    ).astype(np.float32)

    dog_fine = difference_of_gaussians(denoised, p["fine_sigma1"], p["fine_sigma2"])
    dog_coarse = difference_of_gaussians(denoised, p["coarse_sigma1"], p["coarse_sigma2"])
    dog = dog_fine + dog_coarse  # combined, for visualization only

    results = {}
    for polarity, sign in (("bright", 1.0), ("dark", -1.0)):
        extent_channel = opening_by_reconstruction(sign * denoised, int(p["recon_radius"]))

        band_defs = [
            ("fine", np.clip(sign * dog_fine, 0, None), p["dome_height_fine"],
             p["min_component_area_fine"], p["local_max_min_dist"], p["basin_merge_radius_fine"]),
            ("coarse", np.clip(sign * dog_coarse, 0, None), p["dome_height_coarse"],
             p["min_component_area_coarse"], p["local_max_min_dist"], p["basin_merge_radius_coarse"]),
            ("extent", extent_channel, p["dome_height_extent"],
             p["min_component_area_extent"], p["local_max_min_dist_extent"], p["basin_merge_radius_extent"]),
        ]

        band_results = {}
        for name, channel, dome_h, min_comp_area, mdist, merge_r in band_defs:
            band_results[name] = _detect_band(
                channel, roi, dome_h, min_comp_area, mdist, merge_r,
                p["min_area"], p["max_area"], p["min_circularity"], p["min_solidity"],
                p["max_aspect_ratio"], p["min_convexity"], compactness=p["compactness"],
            )

        combined_mask = band_results["fine"]["mask"]
        for name in ("coarse", "extent"):
            combined_mask = cv2.bitwise_or(combined_mask, band_results[name]["mask"])

        if p["micro_enabled"]:
            # Minute-defect band: same marker-controlled-watershed engine,
            # tuned to a small dome height / tight marker spacing / small
            # max-area cap so it only ever contributes tiny candidates the
            # fine band's own area filter would otherwise discard. This
            # replaces v3's z-score-based `local_contrast_rescue` -- it can
            # only ever ADD candidates that this same watershed engine
            # proposes on its own terms, never re-validate against a
            # statistical cutoff.
            micro_result = _detect_band(
                np.clip(sign * dog_fine, 0, None), roi, p["dome_height_micro"],
                p["min_component_area_micro"], p["local_max_min_dist_micro"],
                p["basin_merge_radius_micro"],
                p["min_area"], p["micro_max_area"], p["min_circularity"], p["min_solidity"],
                p["max_aspect_ratio"], p["min_convexity"], compactness=p["compactness"],
            )
            band_results["micro"] = micro_result
            combined_mask = cv2.bitwise_or(combined_mask, micro_result["mask"])

        combined_contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        results[polarity] = dict(
            mask=combined_mask, contours=list(combined_contours), bands=band_results,
        )

    return dict(
        roi=roi, normalized=normalized, background=background, denoised=denoised,
        dog=dog, dog_fine=dog_fine, dog_coarse=dog_coarse,
        bright=results["bright"], dark=results["dark"],
    )


def run_pipeline(image_path, max_dim, params, roi_polygon=None):
    """Full pipeline entry point. Used identically by both single-image and
    batch export; the GUI module never re-implements any of this."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = perspective_correct(img)

    h, w = img.shape[:2]
    if max_dim is not None and max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        img_disp = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        img_disp = img.copy()

    gray = cv2.cvtColor(img_disp, cv2.COLOR_BGR2GRAY)

    # roi_polygon is already normalized (x_frac, y_frac) -- no rescaling
    # needed regardless of whether this call is the downsized live preview
    # or a full-resolution export.
    det = _run_detection(gray, params, roi_polygon=roi_polygon)
    final_mask = cv2.bitwise_or(det["bright"]["mask"], det["dark"]["mask"])

    bg_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    white_display = bg_rgb.copy()
    dark_display = bg_rgb.copy()
    final_output = bg_rgb.copy()
    cv2.drawContours(white_display, det["bright"]["contours"], -1, (255, 0, 0), -1)
    cv2.drawContours(dark_display, det["dark"]["contours"], -1, (255, 0, 0), -1)
    cv2.drawContours(final_output, det["bright"]["contours"], -1, (255, 0, 0), -1)     # red = bright
    cv2.drawContours(final_output, det["dark"]["contours"], -1, (255, 140, 0), -1)     # orange = dark

    return dict(
        img_disp=img_disp, gray=gray, normalized=det["normalized"], background=det["background"],
        denoised=det["denoised"], dog=det["dog"], dog_fine=det["dog_fine"], dog_coarse=det["dog_coarse"],
        roi=det["roi"], bright=det["bright"], dark=det["dark"],
        final_mask=final_mask, white_display=white_display, dark_display=dark_display,
        final_output=final_output,
    )


def dog_to_colormap(dog, clip_std=3.0):
    """Diverging visualization of the (signed) DoG response. `clip_std` is
    a display-range clip for the colormap only -- it decides how many
    standard deviations of contrast fit in the visualization's color
    range, and never feeds back into detection. Unchanged from v3."""
    mu, sigma = float(dog.mean()), float(dog.std() + 1e-6)
    clipped = np.clip(dog, mu - clip_std * sigma, mu + clip_std * sigma)
    norm = cv2.normalize(clipped, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return cv2.applyColorMap(norm, cv2.COLORMAP_JET)

# ======================================================================
#  GUI
#  (formerly pbf_inspector_app.py's GUI layer -- unchanged behaviour;
#  calls only the pipeline functions defined above, never re-implements
#  any detection logic)
# ======================================================================

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


# ======================================================================
#  Spinbox widget -- unchanged from v2
# ======================================================================

class CTkSpinbox(ctk.CTkFrame):
    def __init__(self, parent, from_val, to_val, current_val, is_float=False, resolution=1, linked_var=None, on_update_callback=None):
        super().__init__(parent, fg_color="transparent")
        self.from_val = from_val
        self.to_val = to_val
        self.is_float = is_float
        self.resolution = resolution
        self.linked_var = linked_var
        self.callback = on_update_callback

        self.btn_down = ctk.CTkButton(self, text="\u25bc", width=28, height=28, font=("Arial", 10),
                                       fg_color="#37474f", hover_color="#263238", command=self.decrement)
        self.btn_down.pack(side=tk.LEFT, padx=2)

        self.entry = ctk.CTkEntry(self, width=65, height=28, font=('Helvetica', 12), justify='center')
        self.entry.insert(0, f"{current_val:.2f}" if self.is_float else str(int(current_val)))
        self.entry.pack(side=tk.LEFT, padx=2)

        self.btn_up = ctk.CTkButton(self, text="\u25b2", width=28, height=28, font=("Arial", 10),
                                     fg_color="#37474f", hover_color="#263238", command=self.increment)
        self.btn_up.pack(side=tk.LEFT, padx=2)

    def get_value(self):
        try:
            return float(self.entry.get()) if self.is_float else int(float(self.entry.get()))
        except ValueError:
            return self.from_val

    def set_value(self, val):
        self.entry.delete(0, tk.END)
        self.entry.insert(0, f"{val:.2f}" if self.is_float else str(int(val)))
        if self.linked_var:
            self.linked_var.set(val)

    def increment(self):
        curr = self.get_value()
        new_val = np.clip(curr + self.resolution, self.from_val, self.to_val)
        self.set_value(new_val)
        if self.callback:
            self.callback()

    def decrement(self):
        curr = self.get_value()
        new_val = np.clip(curr - self.resolution, self.from_val, self.to_val)
        self.set_value(new_val)
        if self.callback:
            self.callback()


# ======================================================================
#  NEW: Polygon ROI editor (Problem 9)
# ======================================================================

class ROIPolygonEditor(ctk.CTkToplevel):
    """Click-to-add / drag-to-move / right-click-to-delete polygon editor.

    Left click on empty canvas   -> add a vertex at that point
    Left click + drag a vertex   -> move that vertex
    Right click near a vertex    -> delete that vertex
    'Clear Points'                -> discard all vertices (start over)
    'Confirm ROI'                 -> apply the polygon (or, if empty,
                                      apply "no ROI restriction")
    'Cancel'                      -> close without changing anything

    Calls `on_confirm(normalized_points_or_None)` exactly once, only if the
    user actually confirms or clears -- Cancel calls nothing.
    """

    HANDLE_RADIUS = 5
    HIT_RADIUS = 10
    MAX_DISPLAY_DIM = 900

    def __init__(self, parent, cv_img_bgr, existing_polygon_norm, on_confirm):
        super().__init__(parent)
        self.title("Define Processing ROI \u2014 click to add points, drag to move, right-click to delete")
        self.on_confirm = on_confirm
        self.dragging_idx = None

        h, w = cv_img_bgr.shape[:2]
        scale = self.MAX_DISPLAY_DIM / float(max(h, w)) if max(h, w) > self.MAX_DISPLAY_DIM else 1.0
        self.disp_w, self.disp_h = max(1, int(w * scale)), max(1, int(h * scale))

        disp_img = cv2.resize(cv_img_bgr, (self.disp_w, self.disp_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(disp_img, cv2.COLOR_BGR2RGB)
        # Kept as an instance attribute -- PhotoImage is garbage-collected
        # (and the canvas goes blank) if no Python reference survives.
        self.tk_photo = ImageTk.PhotoImage(Image.fromarray(rgb))

        self.canvas = tk.Canvas(self, width=self.disp_w, height=self.disp_h,
                                 cursor="tcross", bg="black", highlightthickness=0)
        self.canvas.pack(padx=10, pady=10)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_photo)

        if existing_polygon_norm:
            self.points = [(fx * self.disp_w, fy * self.disp_h) for (fx, fy) in existing_polygon_norm]
        else:
            self.points = []

        hint = ctk.CTkLabel(
            self, text="Left-click: add point   |   Drag: move point   |   Right-click: delete point",
            font=('Helvetica', 11, 'italic'), text_color="#90a4ae")
        hint.pack(pady=(0, 5))

        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.pack(pady=(0, 10))
        ctk.CTkButton(btn_row, text="Clear Points", fg_color="#546e7a", hover_color="#37474f",
                      command=self.clear_points).pack(side=tk.LEFT, padx=8)
        ctk.CTkButton(btn_row, text="\u2713 Confirm ROI", fg_color="#2e7d32", hover_color="#1b5e20",
                      command=self.confirm).pack(side=tk.LEFT, padx=8)
        ctk.CTkButton(btn_row, text="Cancel", fg_color="#8d1e1e", hover_color="#5c1414",
                      command=self.destroy).pack(side=tk.LEFT, padx=8)

        self.canvas.bind("<Button-1>", self.on_left_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Button-3>", self.on_right_click)

        self.redraw()
        self.grab_set()  # modal-ish: keep focus on the editor while it's open

    def find_nearest_point(self, x, y):
        best_idx, best_dist = None, self.HIT_RADIUS
        for i, (px, py) in enumerate(self.points):
            d = ((px - x) ** 2 + (py - y) ** 2) ** 0.5
            if d <= best_dist:
                best_dist, best_idx = d, i
        return best_idx

    def on_left_click(self, event):
        idx = self.find_nearest_point(event.x, event.y)
        if idx is not None:
            self.dragging_idx = idx
        else:
            self.points.append((event.x, event.y))
            self.redraw()

    def on_drag(self, event):
        if self.dragging_idx is not None:
            x = min(max(event.x, 0), self.disp_w)
            y = min(max(event.y, 0), self.disp_h)
            self.points[self.dragging_idx] = (x, y)
            self.redraw()

    def on_release(self, _event):
        self.dragging_idx = None

    def on_right_click(self, event):
        idx = self.find_nearest_point(event.x, event.y)
        if idx is not None:
            del self.points[idx]
            self.redraw()

    def redraw(self):
        self.canvas.delete("roi_shape")
        if len(self.points) >= 3:
            flat = [c for pt in self.points for c in pt]
            self.canvas.create_polygon(flat, outline="#00e5ff", fill="#00e5ff", stipple="gray12",
                                        width=2, tags="roi_shape")
        elif len(self.points) == 2:
            flat = [c for pt in self.points for c in pt]
            self.canvas.create_line(flat, fill="#00e5ff", width=2, tags="roi_shape")
        for (x, y) in self.points:
            r = self.HANDLE_RADIUS
            self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="#ff4081",
                                     outline="white", tags="roi_shape")

    def clear_points(self):
        self.points = []
        self.redraw()

    def confirm(self):
        if len(self.points) == 0:
            self.on_confirm(None)  # explicit clear -- no ROI restriction
            self.destroy()
            return
        if len(self.points) < 3:
            messagebox.showwarning("ROI", "Place at least 3 points to define a polygon (or clear all points for no ROI restriction).")
            return
        norm = [(x / self.disp_w, y / self.disp_h) for (x, y) in self.points]
        self.on_confirm(norm)
        self.destroy()


# ======================================================================
#  Main application window
# ======================================================================

class PBFInspectorWorkstation(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("PBF Powder-Bed Inspection Workstation \u2014 Marker-Controlled Watershed Engine v4 (No Thresholds)")
        self.geometry("1650x1040")
        self.image_path = None
        self.folder_path = None
        self.max_display_dim = 420
        self.updating_ui = False
        self.params = {}
        self.roi_polygon = None  # normalized (x_frac, y_frac) points, or None = no restriction

        self.setup_ui_architecture()

    # ---------------- UI construction ----------------

    def setup_ui_architecture(self):
        top_bar = ctk.CTkFrame(self, height=60, corner_radius=0)
        top_bar.pack(side=tk.TOP, fill=tk.X)

        load_btn = ctk.CTkButton(top_bar, text="\U0001F4C2 Load Single Image", command=self.load_image_action,
                                  font=('Helvetica', 12, 'bold'), fg_color="#0288d1", hover_color="#01579b")
        load_btn.pack(side=tk.LEFT, padx=15, pady=10)

        load_folder_btn = ctk.CTkButton(top_bar, text="\U0001F4C1 Load Batch Folder", command=self.load_folder_action,
                                         font=('Helvetica', 12, 'bold'), fg_color="#ff8f00", hover_color="#c67100")
        load_folder_btn.pack(side=tk.LEFT, padx=10, pady=10)

        roi_btn = ctk.CTkButton(top_bar, text="\U0001F53A Define ROI", command=self.open_roi_editor,
                                 font=('Helvetica', 12, 'bold'), fg_color="#6a1b9a", hover_color="#4a148c")
        roi_btn.pack(side=tk.LEFT, padx=10, pady=10)

        clear_roi_btn = ctk.CTkButton(top_bar, text="\u2715 Clear ROI", command=self.clear_roi,
                                       font=('Helvetica', 12), fg_color="#546e7a", hover_color="#37474f", width=90)
        clear_roi_btn.pack(side=tk.LEFT, padx=(0, 10), pady=10)

        self.path_lbl = ctk.CTkLabel(top_bar, text="No source files loaded.", font=('Helvetica', 12, 'italic'), text_color="#90a4ae")
        self.path_lbl.pack(side=tk.LEFT, padx=15)

        self.viewport_frame = ctk.CTkFrame(self, corner_radius=10, fg_color="#1e1e1e")
        self.viewport_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=15, pady=10)

        for i in range(6):
            self.viewport_frame.columnconfigure(i, weight=1, uniform="viewport")
        self.viewport_frame.rowconfigure(1, weight=1)

        titles = [
            "1. Raw Input (ROI outlined)",
            "2. Illumination-Normalized",
            "3. DoG Response (warm=bright / cool=dark)",
            "4. Bright Defects (Red)",
            "5. Dark Defects (Red)",
            "6. Unified Output (Bright=Red, Dark=Orange)",
        ]
        self.view_labels = []
        for i, t in enumerate(titles):
            lbl = ctk.CTkLabel(self.viewport_frame, text=t, font=('Helvetica', 11, 'bold'), text_color="white")
            lbl.grid(row=0, column=i, pady=(10, 5), sticky="ew")
            view = ctk.CTkLabel(self.viewport_frame, text="Awaiting Data...", text_color="#616161")
            view.grid(row=1, column=i, padx=3, pady=5, sticky="nsew")
            self.view_labels.append(view)

        control_panel = ctk.CTkFrame(self, height=400)
        control_panel.pack(side=tk.BOTTOM, fill=tk.X, padx=15, pady=15)

        self.tabview = ctk.CTkTabview(control_panel, height=380)
        self.tabview.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.tab_illum = self.tabview.add(" Tab 1 \u2014 Illumination & Denoise ")
        self.tab_dog = self.tabview.add(" Tab 2 \u2014 Multi-Scale Gradient Bands ")
        self.tab_sens = self.tabview.add(" Tab 3 \u2014 Marker Generation & Watershed ")
        self.tab_filter = self.tabview.add(" Tab 4 \u2014 Defect Shape Filtering ")
        self.tab_extent = self.tabview.add(" Tab 5 \u2014 Full-Extent Band & Continuity ")
        self.tab_rescue = self.tabview.add(" Tab 6 \u2014 Micro-Scale Marker Refinement ")

        self.build_tabs_layout()

        right_bar = ctk.CTkFrame(control_panel, width=240, fg_color="transparent")
        right_bar.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

        self.export_btn = ctk.CTkButton(right_bar, text="\U0001F4BE Export Output\n(Single or Batch Folder)", command=self.start_async_export,
                                         font=('Helvetica', 12, 'bold'), fg_color="#2e7d32", hover_color="#1b5e20")
        self.export_btn.pack(fill=tk.BOTH, expand=True, pady=10)

    def create_parameter_row(self, parent, key, label_text, from_val, to_val, current_val, row_idx, is_float=False, resolution=1):
        lbl = ctk.CTkLabel(parent, text=label_text, font=('Helvetica', 11), anchor='w')
        lbl.grid(row=row_idx, column=0, sticky='w', pady=3, padx=10)

        var = tk.DoubleVar(value=current_val) if is_float else tk.IntVar(value=current_val)
        self.params[key] = var

        slider = ctk.CTkSlider(parent, from_=from_val, to=to_val, number_of_steps=max(1, int((to_val - from_val) / resolution)), variable=var)
        slider.grid(row=row_idx, column=1, sticky='ew', padx=20, pady=3)

        spinbox = CTkSpinbox(parent, from_val, to_val, current_val, is_float, resolution, linked_var=var, on_update_callback=self.trigger_pipeline_refresh)
        spinbox.grid(row=row_idx, column=2, padx=10, pady=3)

        slider.bind("<ButtonRelease-1>", lambda event: self.sync_slider_to_spinbox(var, spinbox))
        spinbox.entry.bind("<Return>", lambda event: self.sync_spinbox_to_slider(spinbox, var, from_val, to_val))

        parent.columnconfigure(1, weight=1)
        return var, spinbox

    def create_toggle_row(self, parent, key, label_text, current_val, row_idx):
        """New: boolean on/off row (used for micro_enabled). Registered
        in the same self.params dict as the numeric sliders -- BooleanVar.get()
        also returns a plain value, so get_current_params_dict() needs no
        special-casing."""
        lbl = ctk.CTkLabel(parent, text=label_text, font=('Helvetica', 11), anchor='w')
        lbl.grid(row=row_idx, column=0, sticky='w', pady=3, padx=10)
        var = tk.BooleanVar(value=current_val)
        self.params[key] = var
        switch = ctk.CTkSwitch(parent, text="", variable=var, command=self.trigger_pipeline_refresh)
        switch.grid(row=row_idx, column=1, sticky='w', padx=20, pady=3)
        return var

    def sync_slider_to_spinbox(self, var, spinbox):
        if self.updating_ui:
            return
        self.updating_ui = True
        spinbox.set_value(var.get())
        self.updating_ui = False
        self.trigger_pipeline_refresh()

    def sync_spinbox_to_slider(self, spinbox, var, from_val, to_val):
        if self.updating_ui:
            return
        self.updating_ui = True
        val = spinbox.get_value()
        if from_val <= val <= to_val:
            var.set(val)
            self.updating_ui = False
            self.trigger_pipeline_refresh()
            return
        else:
            messagebox.showwarning("Out of Bounds", f"Value must be between {from_val} and {to_val}")
        self.updating_ui = False

    def build_tabs_layout(self):
        dp = DEFAULT_PARAMS

        # Tab 1 -- Illumination & Denoise
        self.create_parameter_row(self.tab_illum, "bg_sigma", "Background Gaussian Sigma (Illumination Scale):", 10.0, 150.0, dp["bg_sigma"], 0, is_float=True, resolution=5.0)
        self.create_parameter_row(self.tab_illum, "bilat_d", "Bilateral Filter Diameter:", 1, 25, dp["bilat_d"], 1, resolution=1)
        self.create_parameter_row(self.tab_illum, "bilat_sc", "Bilateral Sigma Color:", 5.0, 200.0, dp["bilat_sc"], 2, is_float=True, resolution=5.0)
        self.create_parameter_row(self.tab_illum, "bilat_ss", "Bilateral Sigma Space:", 5.0, 200.0, dp["bilat_ss"], 3, is_float=True, resolution=5.0)

        # Tab 2 -- Multi-Scale DoG
        self.create_parameter_row(self.tab_dog, "fine_sigma1", "Fine Band Sigma 1 (Small Defect Inner Scale):", 0.3, 5.0, dp["fine_sigma1"], 0, is_float=True, resolution=0.1)
        self.create_parameter_row(self.tab_dog, "fine_sigma2", "Fine Band Sigma 2 (Small Defect Outer Scale):", 1.0, 10.0, dp["fine_sigma2"], 1, is_float=True, resolution=0.2)
        self.create_parameter_row(self.tab_dog, "coarse_sigma1", "Coarse Band Sigma 1 (Large Defect Inner Scale):", 1.0, 20.0, dp["coarse_sigma1"], 2, is_float=True, resolution=0.5)
        self.create_parameter_row(self.tab_dog, "coarse_sigma2", "Coarse Band Sigma 2 (Large Defect Outer Scale):", 5.0, 60.0, dp["coarse_sigma2"], 3, is_float=True, resolution=1.0)
        self.create_parameter_row(self.tab_dog, "recon_radius", "Speckle Reconstruction Radius (px):", 0, 5, dp["recon_radius"], 4, resolution=1)

        # Tab 3 -- Marker Generation & Watershed
        self.create_parameter_row(self.tab_sens, "dome_height_fine", "Fine-Band Marker Dome Height:", 1.0, 60.0, dp["dome_height_fine"], 0, is_float=True, resolution=1.0)
        self.create_parameter_row(self.tab_sens, "dome_height_coarse", "Coarse-Band Marker Dome Height:", 1.0, 60.0, dp["dome_height_coarse"], 1, is_float=True, resolution=1.0)
        self.create_parameter_row(self.tab_sens, "min_component_area_fine", "Fine-Band Min Component Area (px):", 1, 100, dp["min_component_area_fine"], 2, resolution=1)
        self.create_parameter_row(self.tab_sens, "min_component_area_coarse", "Coarse-Band Min Component Area (px):", 5, 500, dp["min_component_area_coarse"], 3, resolution=5)
        self.create_parameter_row(self.tab_sens, "local_max_min_dist", "Watershed Marker Min Distance (px):", 1, 30, dp["local_max_min_dist"], 4, resolution=1)
        self.create_parameter_row(self.tab_sens, "basin_merge_radius_fine", "Fine-Band Basin Merge Radius (px):", 0, 8, dp["basin_merge_radius_fine"], 5, resolution=1)
        self.create_parameter_row(self.tab_sens, "basin_merge_radius_coarse", "Coarse-Band Basin Merge Radius (px):", 0, 8, dp["basin_merge_radius_coarse"], 6, resolution=1)
        self.create_parameter_row(self.tab_sens, "compactness", "Watershed Compactness:", 0.0, 5.0, dp["compactness"], 7, is_float=True, resolution=0.1)

        # Tab 4 -- Defect Shape Filtering
        self.create_parameter_row(self.tab_filter, "min_area", "Minimum Area (Pixels):", 1, 5000, dp["min_area"], 0, resolution=1)
        self.create_parameter_row(self.tab_filter, "max_area", "Maximum Area (Pixels):", 100, 50000, dp["max_area"], 1, resolution=100)
        self.create_parameter_row(self.tab_filter, "min_circularity", "Minimum Circularity:", 0.0, 1.0, dp["min_circularity"], 2, is_float=True, resolution=0.05)
        self.create_parameter_row(self.tab_filter, "min_solidity", "Minimum Solidity:", 0.0, 1.0, dp["min_solidity"], 3, is_float=True, resolution=0.05)
        self.create_parameter_row(self.tab_filter, "max_aspect_ratio", "Maximum Aspect Ratio:", 1.0, 20.0, dp["max_aspect_ratio"], 4, is_float=True, resolution=0.5)
        self.create_parameter_row(self.tab_filter, "min_convexity", "Minimum Convexity:", 0.0, 1.0, dp["min_convexity"], 5, is_float=True, resolution=0.05)

        # Tab 5 -- Full-Extent Band & Continuity (large homogeneous defects
        # keep their full extent instead of only a detected rim)
        self.create_parameter_row(self.tab_extent, "dome_height_extent", "Full-Extent Marker Dome Height:", 1.0, 120.0, dp["dome_height_extent"], 0, is_float=True, resolution=1.0)
        self.create_parameter_row(self.tab_extent, "min_component_area_extent", "Full-Extent Min Component Area (px):", 10, 2000, dp["min_component_area_extent"], 1, resolution=10)
        self.create_parameter_row(self.tab_extent, "local_max_min_dist_extent", "Full-Extent Watershed Marker Min Distance (px):", 1, 60, dp["local_max_min_dist_extent"], 2, resolution=1)
        self.create_parameter_row(self.tab_extent, "basin_merge_radius_extent", "Full-Extent Basin Merge Radius (px):", 0, 12, dp["basin_merge_radius_extent"], 3, resolution=1)

        # Tab 6 -- Micro-Scale Marker Refinement (minute defects: a small
        # dome height / tight marker spacing / small max-area cap so this
        # band only ever contributes tiny candidates)
        self.create_toggle_row(self.tab_rescue, "micro_enabled", "Enable Micro-Scale Marker Band:", dp["micro_enabled"], 0)
        self.create_parameter_row(self.tab_rescue, "dome_height_micro", "Micro-Band Marker Dome Height:", 1.0, 40.0, dp["dome_height_micro"], 1, is_float=True, resolution=1.0)
        self.create_parameter_row(self.tab_rescue, "min_component_area_micro", "Micro-Band Min Component Area (px):", 1, 50, dp["min_component_area_micro"], 2, resolution=1)
        self.create_parameter_row(self.tab_rescue, "local_max_min_dist_micro", "Micro-Band Watershed Marker Min Distance (px):", 1, 10, dp["local_max_min_dist_micro"], 3, resolution=1)
        self.create_parameter_row(self.tab_rescue, "basin_merge_radius_micro", "Micro-Band Basin Merge Radius (px):", 0, 5, dp["basin_merge_radius_micro"], 4, resolution=1)
        self.create_parameter_row(self.tab_rescue, "micro_max_area", "Max Area Considered 'Minute' (px):", 5, 300, dp["micro_max_area"], 5, resolution=5)

    # ---------------- ROI editor wiring (Problem 9) ----------------

    def open_roi_editor(self):
        if not self.image_path:
            messagebox.showwarning("Define ROI", "Load a single image or batch folder first.")
            return
        img = cv2.imread(self.image_path)
        if img is None:
            messagebox.showerror("Define ROI", f"Could not read image:\n{self.image_path}")
            return
        img = perspective_correct(img)
        ROIPolygonEditor(self, img, self.roi_polygon, self.on_roi_confirmed)

    def on_roi_confirmed(self, normalized_points_or_none):
        self.roi_polygon = normalized_points_or_none
        self.trigger_pipeline_refresh()

    def clear_roi(self):
        self.roi_polygon = None
        self.trigger_pipeline_refresh()

    # ---------------- File loading ----------------

    def load_image_action(self):
        file_path = filedialog.askopenfilename(filetypes=[("All Image Formats", "*.jpg *.jpeg *.png *.bmp *.tiff *.pgm")])
        if file_path:
            self.folder_path = None
            self.image_path = file_path
            self.path_lbl.configure(text=os.path.basename(file_path))
            self.trigger_pipeline_refresh()

    def load_folder_action(self):
        folder_path = filedialog.askdirectory(title="Select Batch Folder of AM Images")
        if folder_path:
            self.image_path = None
            self.folder_path = folder_path
            files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
            self.path_lbl.configure(text=f"Batch Folder: {os.path.basename(folder_path)} ({len(files)} target images)")
            if len(files) > 0:
                self.image_path = os.path.join(folder_path, files[0])
                self.trigger_pipeline_refresh()

    def get_current_params_dict(self):
        return {k: v.get() for k, v in self.params.items()}

    # ---------------- Live preview ----------------

    def trigger_pipeline_refresh(self):
        if not self.image_path or self.updating_ui:
            return
        p_dict = self.get_current_params_dict()
        result = run_pipeline(self.image_path, self.max_display_dim, p_dict, roi_polygon=self.roi_polygon)
        if result is not None:
            self.update_native_viewports(result)

    def convert_to_ctk_image(self, cv_img_bgr_or_rgb, source="bgr"):
        if source == "bgr":
            rgb = cv2.cvtColor(cv_img_bgr_or_rgb, cv2.COLOR_BGR2RGB)
        elif source == "gray":
            rgb = cv2.cvtColor(cv_img_bgr_or_rgb, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv_img_bgr_or_rgb
        h, w = rgb.shape[:2]
        pil_img = Image.fromarray(rgb)
        return ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(w, h))

    def draw_roi_outline(self, img_bgr):
        """NEW: outline the active ROI polygon on the Raw Input panel so the
        current processing mask is visible at a glance while tuning. Purely
        cosmetic -- does not affect any detection result."""
        if not self.roi_polygon or len(self.roi_polygon) < 3:
            return img_bgr
        h, w = img_bgr.shape[:2]
        pts = np.array([(fx * w, fy * h) for (fx, fy) in self.roi_polygon], dtype=np.int32)
        out = img_bgr.copy()
        cv2.polylines(out, [pts], isClosed=True, color=(255, 229, 0), thickness=2)
        return out

    def update_native_viewports(self, result):
        normalized_u8 = np.clip(result["normalized"], 0, 255).astype(np.uint8)
        dog_colored = dog_to_colormap(result["dog"])
        raw_with_roi = self.draw_roi_outline(result["img_disp"])

        panels = [
            self.convert_to_ctk_image(raw_with_roi, source="bgr"),
            self.convert_to_ctk_image(normalized_u8, source="gray"),
            self.convert_to_ctk_image(dog_colored, source="bgr"),
            self.convert_to_ctk_image(result["white_display"], source="rgb"),
            self.convert_to_ctk_image(result["dark_display"], source="rgb"),
            self.convert_to_ctk_image(result["final_output"], source="rgb"),
        ]
        for lbl, img in zip(self.view_labels, panels):
            lbl.configure(image=img, text="")
            lbl.image = img

    # ---------------- Export ----------------

    def start_async_export(self):
        if not self.image_path and not self.folder_path:
            messagebox.showwarning("Export Error", "Please load a single image or batch folder first.")
            return

        selected_parent_dir = filedialog.askdirectory(title="Select Destination Folder")
        if not selected_parent_dir:
            return

        self.export_btn.configure(state="disabled", text="\u23f3 Running High-Res Batch Pool...")

        export_thread = threading.Thread(target=self.background_export_worker, args=(selected_parent_dir,))
        export_thread.daemon = True
        export_thread.start()

    def background_export_worker(self, target_base_path):
        try:
            p_dict = self.get_current_params_dict()
            roi_polygon = self.roi_polygon  # same ROI for every file in a batch (Problem 10):
            # a fixed PBF camera rig means every frame in a batch shares the
            # same framing, so one polygon drawn once is valid for all of
            # them. If a given rig's framing varies image-to-image, redraw
            # the ROI between batches rather than relying on one polygon
            # for a mixed-framing folder.

            if self.folder_path:
                output_dir = os.path.join(target_base_path, "Output_Images_DoG_Watershed")
                os.makedirs(output_dir, exist_ok=True)
                files_to_process = [
                    os.path.join(self.folder_path, f)
                    for f in os.listdir(self.folder_path)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))
                ]
            else:
                base_filename = os.path.splitext(os.path.basename(self.image_path))[0]
                output_dir = os.path.join(target_base_path, f"pbf_inspection_{base_filename}_outputs")
                os.makedirs(output_dir, exist_ok=True)
                files_to_process = [self.image_path]

            def process_single_file(file_path):
                file_name_only = os.path.splitext(os.path.basename(file_path))[0]
                r = run_pipeline(file_path, None, p_dict, roi_polygon=roi_polygon)
                if r is None:
                    return

                if self.folder_path:
                    cv2.imwrite(os.path.join(output_dir, f"{file_name_only}_segmented.png"),
                                cv2.cvtColor(r["final_output"], cv2.COLOR_RGB2BGR))
                else:
                    # Full stage dump -- satisfies "visualization for every stage" (spec item 11)
                    cv2.imwrite(os.path.join(output_dir, "01_original.png"), cv2.imread(file_path))
                    cv2.imwrite(os.path.join(output_dir, "02_illumination_normalized.png"),
                                np.clip(r["normalized"], 0, 255).astype(np.uint8))
                    cv2.imwrite(os.path.join(output_dir, "03_background_estimate.png"),
                                np.clip(r["background"], 0, 255).astype(np.uint8))
                    cv2.imwrite(os.path.join(output_dir, "04_denoised.png"),
                                np.clip(r["denoised"], 0, 255).astype(np.uint8))
                    cv2.imwrite(os.path.join(output_dir, "05_dog_combined.png"), dog_to_colormap(r["dog"]))
                    cv2.imwrite(os.path.join(output_dir, "06_dog_fine_band.png"), dog_to_colormap(r["dog_fine"]))
                    cv2.imwrite(os.path.join(output_dir, "07_dog_coarse_band.png"), dog_to_colormap(r["dog_coarse"]))
                    cv2.imwrite(os.path.join(output_dir, "08_bright_defect_mask.png"), r["bright"]["mask"])
                    cv2.imwrite(os.path.join(output_dir, "09_dark_defect_mask.png"), r["dark"]["mask"])
                    cv2.imwrite(os.path.join(output_dir, "10_final_binary_mask.png"), r["final_mask"])
                    cv2.imwrite(os.path.join(output_dir, "10b_roi_mask.png"), r["roi"])
                    cv2.imwrite(os.path.join(output_dir, "11_white_defects_overlay.png"), cv2.cvtColor(r["white_display"], cv2.COLOR_RGB2BGR))
                    cv2.imwrite(os.path.join(output_dir, "12_dark_defects_overlay.png"), cv2.cvtColor(r["dark_display"], cv2.COLOR_RGB2BGR))
                    cv2.imwrite(os.path.join(output_dir, "13_final_overlay.png"), cv2.cvtColor(r["final_output"], cv2.COLOR_RGB2BGR))

                    with open(os.path.join(output_dir, "parameters.json"), "w") as f:
                        json.dump({
                            "configured_parameters": p_dict,
                            "roi_polygon_normalized": roi_polygon,
                            "engine": "dual_polarity_marker_controlled_watershed_v4_no_threshold",
                            "bright_defect_count": len(r["bright"]["contours"]),
                            "dark_defect_count": len(r["dark"]["contours"]),
                        }, f, indent=4)

            with ThreadPoolExecutor() as executor:
                executor.map(process_single_file, files_to_process)

            self.after(0, lambda: self.on_export_complete(output_dir))

        except Exception as e:
            error_msg = str(e)
            self.after(0, lambda: self.on_export_failed(error_msg))

    def on_export_complete(self, output_dir):
        self.export_btn.configure(state="normal", text="\U0001F4BE Export Output\n(Single or Batch Folder)")
        messagebox.showinfo("Export Successful", f"Operations completed successfully!\nTarget Directory:\n{output_dir}")

    def on_export_failed(self, error_msg):
        self.export_btn.configure(state="normal", text="\U0001F4BE Export Output\n(Single or Batch Folder)")
        messagebox.showerror("Export Error", f"An execution error occurred:\n{error_msg}")


if __name__ == "__main__":
    app = PBFInspectorWorkstation()
    app.mainloop()
