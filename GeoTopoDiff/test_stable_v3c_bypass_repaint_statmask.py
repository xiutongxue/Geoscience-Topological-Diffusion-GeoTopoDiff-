# Copyright (c) 2022 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This repository was forked from https://github.com/openai/guided-diffusion, which is under the MIT license

"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import argparse
import hashlib
import math
import os
from collections import deque

import conf_mgt
import numpy as np
import torch as th
import torch.nn.functional as F
from guided_diffusion import dist_util
from guided_diffusion.image_datasets import load_data_inpa
from utils import yamlread

# Workaround
try:
    import ctypes

    libgcc_s = ctypes.CDLL("libgcc_s.so.1")
except:
    pass


def _to_1ch(mask):
    return (
        mask[:, :1, :, :]
        if (mask is not None and mask.dim() == 4 and mask.shape[1] > 1)
        else mask
    )


# ============================================================
# ✅ FIXED: masked mean (per B/C independent + spatial-only aggregation + stable on empty mask)
# ============================================================
def _to_1ch_robust(mask):
    """
    Robust single-channel conversion.
    Instead of simply slicing mask[:, :1], takes the max across all channels
    to prevent structural information stored in other channels of a multi-channel mask from being lost.
    """
    if mask is None:
        return None
    if mask.dim() == 4 and mask.shape[1] > 1:
        # Take max to ensure any channel marked as mask is preserved
        return mask.max(dim=1, keepdim=True)[0]
    return mask



from guided_diffusion.script_util import (
    NUM_CLASSES,
    classifier_defaults,
    create_classifier,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    select_args,
)  # noqa: E402


def toU8(sample):
    if sample is None:
        return sample
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    sample = sample.detach().cpu().numpy()
    return sample


# ============================================================
# Coarse Prefill + Texture Fusion utility functions
# ============================================================


def _dilate_binary_mask(mask, pixels=1):
    """
    mask: [B,1,H,W], 1 = region to dilate
    return: dilated mask
    """
    if pixels is None or int(pixels) <= 0:
        return mask
    k = 2 * int(pixels) + 1
    return F.max_pool2d(
        mask.float(), kernel_size=k, stride=1, padding=int(pixels)
    ).clamp(0.0, 1.0)


def _build_hole_width_masks(keep_mask, narrow_width=6, medium_width=24):
    """
    keep_mask: [B,1,H,W], 1=known region, 0=missing region
    Returns: narrow_mask, medium_mask, wide_mask, width_map
    """
    keep = (keep_mask > 0.5).float()
    hole = 1.0 - keep

    B, _, H, W = hole.shape
    device = hole.device
    narrow = th.zeros_like(hole)
    medium = th.zeros_like(hole)
    wide = th.zeros_like(hole)
    width_map = th.zeros_like(hole)

    hole_cpu = hole.detach().cpu().numpy()

    for b in range(B):
        for y in range(H):
            row = hole_cpu[b, 0, y] > 0.5
            x = 0
            while x < W:
                while x < W and not row[x]:
                    x += 1
                if x >= W:
                    break
                s = x
                while x < W and row[x]:
                    x += 1
                e = x
                w = e - s
                width_map[b, 0, y, s:e] = float(w) / float(max(W, 1))
                if w <= int(narrow_width):
                    narrow[b, 0, y, s:e] = 1.0
                elif w <= int(medium_width):
                    medium[b, 0, y, s:e] = 1.0
                else:
                    wide[b, 0, y, s:e] = 1.0

    return narrow.to(device), medium.to(device), wide.to(device), width_map.to(device)


# ============================================================
# P1-A: Mask Routing diagnostics
# ============================================================
def build_mask_routing_maps(gt_keep_mask: th.Tensor, conf) -> dict:
    """
    Classify missing regions into multiple masks based on hole width and connected component shape.
    Pure diagnostic function — does not change any output, only generates visualizations + statistics.

    Returns dict with mask names as keys and [B,1,H,W] tensors as values.
    """
    hole = _to_1ch(1.0 - gt_keep_mask.float())
    B, _, H, W = hole.shape
    device = hole.device

    # Configuration parameters
    narrow_max = int(conf.get("routing_narrow_max_width", 6))
    medium_max = int(conf.get("routing_medium_max_width", 24))
    wide_min = int(conf.get("routing_wide_min_width", 25))
    long_v_min_h = int(conf.get("routing_long_vertical_min_height", 64))
    long_v_min_aspect = float(conf.get("routing_long_vertical_min_aspect", 3.0))
    long_v_min_w = int(conf.get("routing_long_vertical_min_width", 4))

    # Initialize all masks
    narrow_mask = th.zeros_like(hole)
    medium_mask = th.zeros_like(hole)
    long_vertical_mask = th.zeros_like(hole)
    wide_area_mask = th.zeros_like(hole)
    width_map = th.zeros_like(hole)

    hole_np = hole.detach().cpu().numpy()

    for b in range(B):
        h_b = hole_np[b, 0]  # [H, W]

        # --- Per-row analysis of horizontal contiguous missing width ---
        for y in range(H):
            row = h_b[y] > 0.5
            x = 0
            while x < W:
                while x < W and not row[x]:
                    x += 1
                if x >= W:
                    break
                s = x
                while x < W and row[x]:
                    x += 1
                e = x
                w = e - s
                width_map[b, 0, y, s:e] = float(w) / float(max(W, 1))
                if w <= narrow_max:
                    narrow_mask[b, 0, y, s:e] = 1.0
                elif w <= medium_max:
                    medium_mask[b, 0, y, s:e] = 1.0
                # wide is determined by connected component, not marked here

        # --- Connected component analysis (for long_vertical and wide_area) ---
        binary = (h_b > 0.5).astype(np.uint8)
        labels, K = _cc_label_2d(binary, connectivity=8)

        for k in range(1, K + 1):
            ys, xs = np.where(labels == k)
            if len(ys) == 0:
                continue
            y_min, y_max = int(ys.min()), int(ys.max())
            x_min, x_max = int(xs.min()), int(xs.max())
            bbox_h = y_max - y_min + 1
            bbox_w = x_max - x_min + 1
            aspect = bbox_h / max(bbox_w, 1)

            # wide_area: bbox width exceeds threshold
            if bbox_w >= wide_min:
                wide_area_mask[b, 0, ys, xs] = 1.0

            # long_vertical: tall connected component + high aspect ratio + minimum width
            if (bbox_h >= long_v_min_h
                    and aspect >= long_v_min_aspect
                    and bbox_w >= long_v_min_w):
                long_vertical_mask[b, 0, ys, xs] = 1.0

    # Remove regions already covered by wide_area (wide_area has highest priority)
    # long_vertical and wide_area can overlap
    # narrow / medium / wide_area are mutually exclusive: wide_area regions removed from narrow/medium
    narrow_mask = narrow_mask * (1.0 - wide_area_mask)
    medium_mask = medium_mask * (1.0 - wide_area_mask)

    # stat_match = medium + long_vertical + wide_area (correct anything that is not a narrow gap)
    stat_match_mask = th.clamp(medium_mask + long_vertical_mask + wide_area_mask, 0.0, 1.0)

    # bypass_repaint = stat_match (P1-A conservative rule, save only for now, not active)
    bypass_repaint_mask = stat_match_mask.clone()

    # adaptive_fusion = wide_area (only large missing regions allow RePaint through gain_map)
    adaptive_fusion_mask = wide_area_mask.clone()

    return {
        "routing_hole_mask": hole,
        "routing_width_map": width_map,
        "routing_narrow_mask": narrow_mask,
        "routing_medium_mask": medium_mask,
        "routing_long_vertical_mask": long_vertical_mask,
        "routing_wide_area_mask": wide_area_mask,
        "routing_stat_match_mask": stat_match_mask,
        "routing_bypass_repaint_mask": bypass_repaint_mask,
        "routing_adaptive_fusion_mask": adaptive_fusion_mask,
    }


def build_routing_class_map(routing_maps: dict) -> th.Tensor:
    """
    Generate colored classification map [B,3,H,W] uint8 (0-255) from routing masks.
    Priority: wide_area(red) > medium(yellow) > narrow(green).
    Outside hole = black.
    """
    hole = routing_maps["routing_hole_mask"]       # [B,1,H,W]
    narrow = routing_maps["routing_narrow_mask"]
    medium = routing_maps["routing_medium_mask"]
    wide = routing_maps["routing_wide_area_mask"]

    B, _, H, W = hole.shape
    cls = th.zeros(B, 3, H, W, dtype=th.uint8, device=hole.device)

    # Uncovered region inside hole (not narrow/medium/wide) = dark gray
    hole_other = hole * (1.0 - narrow) * (1.0 - medium) * (1.0 - wide)
    cls[:, 0][hole_other[:, 0] > 0.5] = 60
    cls[:, 1][hole_other[:, 0] > 0.5] = 60
    cls[:, 2][hole_other[:, 0] > 0.5] = 60

    # narrow = green (0, 200, 0)
    cls[:, 1][narrow[:, 0] > 0.5] = 200

    # medium = yellow (220, 220, 0)
    cls[:, 0][medium[:, 0] > 0.5] = 220
    cls[:, 1][medium[:, 0] > 0.5] = 220

    # wide_area = red (220, 50, 50)
    cls[:, 0][wide[:, 0] > 0.5] = 220
    cls[:, 1][wide[:, 0] > 0.5] = 50
    cls[:, 2][wide[:, 0] > 0.5] = 50

    return cls


# ============================================================
# P1-B: Routing diversion diagnostic experiment
# ============================================================
def _ensure_bool_mask(mask, ref=None):
    """Convert mask to bool [B,1,H,W]."""
    if mask is None:
        return None
    m = mask.bool()
    if m.dim() == 3:
        m = m.unsqueeze(1)
    if ref is not None and m.device != ref.device:
        m = m.to(ref.device)
    return m


def _hard_merge_known(candidate, known_img, known_mask):
    """Force known region to original image, keep candidate in hole region."""
    km = _ensure_bool_mask(known_mask, candidate).float()
    if km.shape[1] == 1 and candidate.shape[1] != 1:
        km = km.expand(-1, candidate.shape[1], -1, -1)
    return (candidate * (1.0 - km) + known_img * km).clamp(-1.0, 1.0)


def _compute_changed_pixels(candidate, baseline, region_mask=None, eps=1e-6):
    """Count differing pixels between candidate and baseline inside the hole."""
    diff = (candidate - baseline).abs()
    if diff.dim() == 4 and diff.shape[1] > 1:
        diff = diff.mean(dim=1, keepdim=True)
    changed = diff > eps
    if region_mask is not None:
        m = _ensure_bool_mask(region_mask, changed)
        if m.shape != changed.shape:
            m = m.expand_as(changed)
        changed = changed & m
    return int(changed.sum().item())


def _save_diff_map(candidate, baseline, mask, out_dir, tag, img_names=None):
    """Save |candidate - baseline| difference map, outside hole is black."""
    diff = (candidate - baseline).abs()
    if diff.dim() == 4 and diff.shape[1] > 1:
        diff = diff.mean(dim=1, keepdim=True)
    m = _ensure_bool_mask(mask, diff).float()
    diff = diff * m
    # Normalize to 0-255
    dmax = diff.max().clamp_min(1e-8)
    diff_u8 = (diff / dmax * 255.0).clamp(0, 255).to(th.uint8)
    # Expand to 3 channels
    diff_3ch = diff_u8.expand(-1, 3, -1, -1)
    _prior_debug_save_batch(diff_3ch, img_names=img_names, out_dir=out_dir, tag=tag, mode="image")


def _filter_small_components(mask_bool, min_area=16, min_height=8):
    """Filter small connected components. mask_bool: [B,1,H,W] bool tensor."""
    B, _, H, W = mask_bool.shape
    out = th.zeros_like(mask_bool)
    for b in range(B):
        arr = mask_bool[b, 0].cpu().numpy().astype(np.uint8)
        labels, K = _cc_label_2d(arr, connectivity=8)
        for k in range(1, K + 1):
            ys, xs = np.where(labels == k)
            area = len(ys)
            bbox_h = int(ys.max() - ys.min() + 1) if area > 0 else 0
            if area >= min_area and bbox_h >= min_height:
                out[b, 0, ys, xs] = True
    return out.to(mask_bool.device)


def run_p1b_diagnostics(
    baseline_current,
    repaint_raw,
    coarse_x0_raw,
    coarse_x0_after_stat,
    routing_maps,
    gt_img,
    gt_keep_mask,
    img_names,
    save_dir,
    conf,
):
    """
    P1-B diagnostic main function.
    Generates candidate images + diff images + grid images + statistics logs.
    Does not modify final_hard_merge_output.
    """
    p1b_filter = bool(conf.get("p1b_filter_small_components", True))
    p1b_min_area = int(conf.get("p1b_min_component_area", 16))
    p1b_min_height = int(conf.get("p1b_min_component_height", 8))

    # ---- 1. Prepare mutually exclusive masks ----
    hole_mask = _ensure_bool_mask(routing_maps["routing_hole_mask"], baseline_current)
    wide_mask = _ensure_bool_mask(routing_maps["routing_wide_area_mask"], baseline_current) & hole_mask
    medium_mask = _ensure_bool_mask(routing_maps["routing_medium_mask"], baseline_current) & hole_mask & (~wide_mask)
    narrow_mask = _ensure_bool_mask(routing_maps["routing_narrow_mask"], baseline_current) & hole_mask & (~wide_mask) & (~medium_mask)

    if p1b_filter:
        narrow_mask = _filter_small_components(narrow_mask, p1b_min_area, p1b_min_height)
        medium_mask = _filter_small_components(medium_mask, p1b_min_area, p1b_min_height)
        # Re-do mutual exclusion
        narrow_mask = narrow_mask & hole_mask & (~wide_mask) & (~medium_mask)
        medium_mask = medium_mask & hole_mask & (~wide_mask)

    # Mutual exclusion check
    assert (wide_mask & medium_mask).sum() == 0, "wide ∩ medium != 0"
    assert (wide_mask & narrow_mask).sum() == 0, "wide ∩ narrow != 0"
    assert (medium_mask & narrow_mask).sum() == 0, "medium ∩ narrow != 0"

    known_mask = _ensure_bool_mask(gt_keep_mask, baseline_current)

    # ---- 2. Generate candidates ----
    # wide: replace baseline with repaint_raw (expected to be worse)
    c_wide = th.where(wide_mask, repaint_raw, baseline_current)
    c_wide = _hard_merge_known(c_wide, gt_img, known_mask)

    # narrow: replace baseline with repaint_raw
    c_narrow = th.where(narrow_mask, repaint_raw, baseline_current)
    c_narrow = _hard_merge_known(c_narrow, gt_img, known_mask)

    # medium: replace baseline with coarse_x0_after_stat
    c_medium_stat = th.where(medium_mask, coarse_x0_after_stat, baseline_current)
    c_medium_stat = _hard_merge_known(c_medium_stat, gt_img, known_mask)

    # medium raw: replace baseline with coarse_x0_raw
    c_medium_raw = th.where(medium_mask, coarse_x0_raw, baseline_current)
    c_medium_raw = _hard_merge_known(c_medium_raw, gt_img, known_mask)

    # route_test: wide=baseline, medium=coarse_after_stat, narrow=repaint_raw
    c_route = baseline_current.clone()
    c_route = th.where(medium_mask, coarse_x0_after_stat, c_route)
    c_route = th.where(narrow_mask, repaint_raw, c_route)
    c_route = _hard_merge_known(c_route, gt_img, known_mask)

    # ---- 3. Save candidate images ----
    candidates = {
        "p1b_baseline_current": baseline_current,
        "p1b_candidate_wide_repaint": c_wide,
        "p1b_candidate_narrow_repaint": c_narrow,
        "p1b_candidate_medium_stat": c_medium_stat,
        "p1b_candidate_medium_raw": c_medium_raw,
        "p1b_candidate_route_test": c_route,
    }
    for tag, tensor in candidates.items():
        _prior_debug_save_batch(tensor, img_names=img_names, out_dir=save_dir, tag=tag, mode="image")

    # ---- 4. Save diff images ----
    _save_diff_map(c_wide, baseline_current, hole_mask, save_dir, "p1b_diff_wide_repaint_vs_baseline", img_names)
    _save_diff_map(c_narrow, baseline_current, hole_mask, save_dir, "p1b_diff_narrow_repaint_vs_baseline", img_names)
    _save_diff_map(c_medium_stat, baseline_current, hole_mask, save_dir, "p1b_diff_medium_stat_vs_baseline", img_names)
    _save_diff_map(c_route, baseline_current, hole_mask, save_dir, "p1b_diff_route_test_vs_baseline", img_names)

    # medium stat vs raw diff
    _save_diff_map(coarse_x0_after_stat, coarse_x0_raw, medium_mask, save_dir, "p1b_diff_medium_stat_vs_raw", img_names)

    # ---- 5. Grid comparison images ----
    _save_p1b_grid(
        gt_img, baseline_current, repaint_raw, coarse_x0_raw, coarse_x0_after_stat,
        routing_maps, c_wide, c_narrow, c_medium_stat, c_route,
        hole_mask, save_dir, img_names,
    )

    # ---- 6. Statistics log ----
    B = baseline_current.shape[0]
    for bi in range(B):
        h = int(hole_mask[bi].sum().item())
        if h == 0:
            continue
        w = int(wide_mask[bi].sum().item())
        m = int(medium_mask[bi].sum().item())
        n = int(narrow_mask[bi].sum().item())
        label = img_names[bi] if img_names and bi < len(img_names) else f"img{bi}"

        km_base = float(((baseline_current[bi:bi+1] - gt_img[bi:bi+1]).abs() * known_mask[bi:bi+1].float()).sum() / known_mask[bi:bi+1].float().sum().clamp_min(1e-6))
        km_wide = float(((c_wide[bi:bi+1] - gt_img[bi:bi+1]).abs() * known_mask[bi:bi+1].float()).sum() / known_mask[bi:bi+1].float().sum().clamp_min(1e-6))
        km_narrow = float(((c_narrow[bi:bi+1] - gt_img[bi:bi+1]).abs() * known_mask[bi:bi+1].float()).sum() / known_mask[bi:bi+1].float().sum().clamp_min(1e-6))
        km_medium = float(((c_medium_stat[bi:bi+1] - gt_img[bi:bi+1]).abs() * known_mask[bi:bi+1].float()).sum() / known_mask[bi:bi+1].float().sum().clamp_min(1e-6))
        km_route = float(((c_route[bi:bi+1] - gt_img[bi:bi+1]).abs() * known_mask[bi:bi+1].float()).sum() / known_mask[bi:bi+1].float().sum().clamp_min(1e-6))

        ch_wide = _compute_changed_pixels(c_wide[bi:bi+1], baseline_current[bi:bi+1], wide_mask[bi:bi+1])
        ch_narrow = _compute_changed_pixels(c_narrow[bi:bi+1], baseline_current[bi:bi+1], narrow_mask[bi:bi+1])
        ch_route = _compute_changed_pixels(c_route[bi:bi+1], baseline_current[bi:bi+1], hole_mask[bi:bi+1])

        print(f"[P1-B] {label}")
        print(f"  hole={h} wide={w}({w/h*100:.1f}%) medium={m}({m/h*100:.1f}%) narrow={n}({n/h*100:.1f}%)")
        print(f"  changed: wide_repaint={ch_wide} narrow_repaint={ch_narrow} route_test={ch_route}")
        print(f"  known_mae: baseline={km_base:.8f} wide={km_wide:.8f} narrow={km_narrow:.8f} medium={km_medium:.8f} route={km_route:.8f}")


def _save_p1b_grid(
    gt_img, baseline, repaint_raw, coarse_raw, coarse_after_stat,
    routing_maps, c_wide, c_narrow, c_medium, c_route,
    hole_mask, save_dir, img_names,
):
    """Save 3x5 P1-B overview grid image."""
    from PIL import Image as PILImage

    B = gt_img.shape[0]
    for bi in range(B):
        imgs_row1 = [gt_img[bi], baseline[bi], repaint_raw[bi], coarse_raw[bi], coarse_after_stat[bi]]
        cls_map = build_routing_class_map(routing_maps)
        imgs_row2 = [cls_map[bi], c_wide[bi], c_narrow[bi], c_medium[bi], c_route[bi]]

        # diff maps
        def _diff_u8(a, b, mask):
            d = (a - b).abs()
            if d.dim() == 3 and d.shape[0] > 1:
                d = d.mean(dim=0, keepdim=True)
            d = d * mask[bi].float()
            dm = d.max().clamp_min(1e-8)
            return (d / dm * 255).clamp(0, 255).to(th.uint8).expand(3, -1, -1)

        hm = hole_mask
        diff_w = _diff_u8(c_wide[bi], baseline[bi], hm)
        diff_n = _diff_u8(c_narrow[bi], baseline[bi], hm)
        diff_m = _diff_u8(c_medium[bi], baseline[bi], hm)
        diff_r = _diff_u8(c_route[bi], baseline[bi], hm)
        imgs_row3 = [diff_w, diff_n, diff_m, diff_r, hole_mask[bi].to(th.uint8).expand(3, -1, -1) * 255]

        def _to_grid_img(rows, cell_h, cell_w):
            grid_h = len(rows)
            grid_w = len(rows[0])
            canvas = np.zeros((cell_h * grid_h, cell_w * grid_w, 3), dtype=np.uint8)
            for r, row in enumerate(rows):
                for c, t in enumerate(row):
                    arr = t.detach().cpu().numpy()
                    # Convert to (H, W, 3) uint8
                    if arr.ndim == 3 and arr.shape[0] in (1, 3):
                        arr = np.transpose(arr, (1, 2, 0))
                    if arr.ndim == 3 and arr.shape[2] == 1:
                        arr = np.concatenate([arr, arr, arr], axis=2)
                    elif arr.ndim == 2:
                        arr = np.stack([arr, arr, arr], axis=-1)
                    if arr.dtype != np.uint8:
                        arr = np.clip(arr * 127.5 + 127.5, 0, 255).astype(np.uint8)
                    arr = np.array(PILImage.fromarray(arr).resize((cell_w, cell_h), PILImage.NEAREST))
                    canvas[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w] = arr
            return canvas

        _, _, H, W = gt_img.shape
        canvas = _to_grid_img([imgs_row1, imgs_row2, imgs_row3], H, W)
        out_img = PILImage.fromarray(canvas)
        if img_names and bi < len(img_names):
            stem = _prior_debug_safe_name(img_names[bi])
        else:
            stem = f"img{bi}"
        out_path = os.path.join(save_dir, f"{stem}_p1b_grid.png")
        os.makedirs(save_dir, exist_ok=True)
        out_img.save(out_path)


# ============================================================
# P2-A: Texture/Boundary diagnostics
# ============================================================
def _build_boundary_masks(hole_mask, inner_width=4, outer_width=4,
                          center_mode="distance", center_top_ratio=0.2,
                          center_min_pixels=16):
    """
    Generate boundary diagnostic masks based on distance transform.
    hole_mask: [B,1,H,W] bool
    """
    from scipy.ndimage import distance_transform_edt

    B, _, H, W = hole_mask.shape
    hole_np = hole_mask[:, 0].cpu().numpy().astype(np.uint8)

    left_seam = th.zeros_like(hole_mask)
    right_seam = th.zeros_like(hole_mask)
    hole_center = th.zeros_like(hole_mask)
    seam_inner = th.zeros_like(hole_mask)
    dist_map = th.zeros_like(hole_mask, dtype=th.float32)

    for b in range(B):
        h_b = hole_np[b]
        # distance transform: distance from each hole pixel to the nearest known pixel
        dist = distance_transform_edt(h_b)  # [H, W]
        dist_map[b, 0] = th.from_numpy(dist).to(dist_map.device)

        # seam_inner: region inside hole with distance to boundary <= inner_width
        seam_inner[b, 0] = th.from_numpy(
            ((dist > 0) & (dist <= inner_width)).astype(np.uint8)
        ).to(hole_mask.device)

        # hole_center: distance >= inner_width, or fallback to top ratio
        center_mask = dist >= inner_width
        center_count = center_mask.sum()

        if center_count < center_min_pixels:
            # fallback: take top_ratio pixels with the largest distance
            dist_hole = dist[h_b > 0]
            if len(dist_hole) > 0:
                threshold = np.percentile(dist_hole, (1.0 - center_top_ratio) * 100)
                center_mask = dist >= threshold
            else:
                center_mask = np.zeros_like(h_b, dtype=bool)

        hole_center[b, 0] = th.from_numpy(
            center_mask.astype(np.uint8)
        ).to(hole_mask.device)

        # Left/right seam: find pixels near hole boundary row by row
        for y in range(H):
            xs = np.where(h_b[y])[0]
            if len(xs) == 0:
                continue
            x_left, x_right = xs[0], xs[-1]
            left_seam[b, 0, y, max(0, x_left - outer_width):min(W, x_left + inner_width + 1)] = 1.0
            right_seam[b, 0, y, max(0, x_right - inner_width):min(W, x_right + outer_width + 1)] = 1.0

    # boundary_ring = seam_inner (boundary region inside hole)
    boundary_ring = seam_inner.bool()

    # context_ring = near boundary outside hole
    h_float = hole_mask.float()
    k_outer = 2 * outer_width + 1
    dilated = F.max_pool2d(h_float, kernel_size=k_outer, stride=1, padding=outer_width)
    context_ring = (~hole_mask) & (dilated > 0.5)

    return {
        "p2a_boundary_ring_mask": boundary_ring,
        "p2a_left_seam_mask": left_seam.bool(),
        "p2a_right_seam_mask": right_seam.bool(),
        "p2a_hole_center_mask": hole_center.bool(),
        "p2a_context_ring_mask": context_ring,
        "p2a_seam_inner_mask": seam_inner.bool(),
        "p2a_dist_map": dist_map,
    }


def _compute_seam_grad(image, left_seam, right_seam, hole_mask):
    """
    Compute gradient discontinuity at seam locations.
    Uses horizontal difference |gray[:, :, :, 1:] - gray[:, :, :, :-1]|,
    only sampled at left/right boundary pixels of the hole.
    Returns per-pixel difference map.
    """
    if image.dim() == 4 and image.shape[1] > 1:
        gray = image.mean(dim=1, keepdim=True)
    else:
        gray = image

    # Horizontal difference (absolute value of right minus left)
    dx = (gray[:, :, :, 1:] - gray[:, :, :, :-1]).abs()  # [B,1,H,W-1]
    # Pad to original width
    dx = F.pad(dx, (0, 1), mode="constant", value=0)  # [B,1,H,W]

    # Only take values inside hole_mask
    hm = _ensure_bool_mask(hole_mask, dx).float()
    diff_map = dx * hm

    return diff_map


def _compute_texture_energy(image, sigma=2.0):
    """
    Local high-frequency energy = |image - gaussian_blur(image)|.
    sigma controls the low-pass filter scale.
    """
    if image.dim() == 4 and image.shape[1] > 1:
        gray = image.mean(dim=1, keepdim=True)
    else:
        gray = image
    low = _gaussian_like_blur(gray, sigma=sigma)
    return (gray - low).abs()


def _compute_row_profile(image, hole_mask, context_width=15):
    """
    Compute per row: left context mean, right context mean, hole mean.
    Returns [B, 3, H] tensor (3 channels: left_ctx, right_ctx, hole_mean).
    """
    if image.dim() == 4 and image.shape[1] > 1:
        gray = image.mean(dim=1)
    else:
        gray = image[:, 0]  # [B, H, W]

    B, H, W = gray.shape
    profiles = th.zeros(B, 3, H, device=image.device)

    gray_np = gray.detach().cpu().numpy()
    hole_np = hole_mask[:, 0].cpu().numpy() if hole_mask.dim() == 4 else hole_mask.cpu().numpy()

    for b in range(B):
        for y in range(H):
            xs_hole = np.where(hole_np[b, y])[0]
            if len(xs_hole) == 0:
                continue
            x_left = xs_hole[0]
            x_right = xs_hole[-1]

            # Left context
            l_start = max(0, x_left - context_width)
            l_vals = gray_np[b, y, l_start:x_left]
            profiles[b, 0, y] = float(l_vals.mean()) if len(l_vals) > 0 else 0.0

            # Right context
            r_end = min(W, x_right + context_width + 1)
            r_vals = gray_np[b, y, x_right + 1:r_end]
            profiles[b, 1, y] = float(r_vals.mean()) if len(r_vals) > 0 else 0.0

            # Hole mean
            h_vals = gray_np[b, y, xs_hole]
            profiles[b, 2, y] = float(h_vals.mean()) if len(h_vals) > 0 else 0.0

    return profiles


def run_p2a_diagnostics(
    baseline_current,
    gt_img,
    gt_keep_mask,
    img_names,
    save_dir,
    conf,
):
    """
    P2-A-Revise diagnostic main function.
    Uses distance transform hole_center, continuous heatmaps, per-region texture statistics.
    Does not modify final.
    """
    inner_w = int(conf.get("p2a_seam_inner_width", 4))
    outer_w = int(conf.get("p2a_seam_outer_width", 4))
    tex_sigma = float(conf.get("p2a_texture_sigma", 2.0))
    ctx_width = int(conf.get("p2a_context_width", 15))
    center_top_ratio = float(conf.get("p2a_center_top_ratio", 0.2))
    center_min_pixels = int(conf.get("p2a_center_min_pixels", 16))

    hole_mask = _ensure_bool_mask(1.0 - gt_keep_mask.float(), baseline_current)
    known_mask = _ensure_bool_mask(gt_keep_mask.float(), baseline_current)

    # ---- 1. Boundary masks (distance transform) ----
    bmasks = _build_boundary_masks(
        hole_mask, inner_w, outer_w,
        center_mode="distance", center_top_ratio=center_top_ratio,
        center_min_pixels=center_min_pixels,
    )

    # ---- 2. Seam gradient (continuous values) ----
    seam_diff = _compute_seam_grad(
        baseline_current,
        bmasks["p2a_left_seam_mask"],
        bmasks["p2a_right_seam_mask"],
        hole_mask,
    )

    # ---- 3. row profile ----
    profiles = _compute_row_profile(baseline_current, hole_mask, ctx_width)
    ctx_mean = (profiles[:, 0:1, :] + profiles[:, 1:2, :]) / 2.0
    profile_delta = (profiles[:, 2:3, :] - ctx_mean).abs()

    # ---- 4. texture energy ----
    tex_energy = _compute_texture_energy(baseline_current, tex_sigma)

    # ---- 5. Per-region statistics ----
    seam_inner_mask = bmasks["p2a_seam_inner_mask"]
    center_mask = bmasks["p2a_hole_center_mask"]
    context_mask = bmasks["p2a_context_ring_mask"]

    B = baseline_current.shape[0]
    for bi in range(B):
        h_count = int(hole_mask[bi].sum().item())
        if h_count == 0:
            continue
        seam_count = int(seam_inner_mask[bi].sum().item())
        center_count = int(center_mask[bi].sum().item())
        ctx_count = int(context_mask[bi].sum().item())
        label = img_names[bi] if img_names and bi < len(img_names) else f"img{bi}"

        # seam grad
        seam_vals = seam_diff[bi, 0][hole_mask[bi, 0]].detach().cpu().numpy()
        seam_mean = float(seam_vals.mean()) if len(seam_vals) > 0 else 0.0
        seam_p95 = float(np.percentile(seam_vals, 95)) if len(seam_vals) > 0 else 0.0

        # row profile delta
        delta_vals = profile_delta[bi, 0].detach().cpu().numpy()
        delta_mean = float(delta_vals.mean())
        delta_p95 = float(np.percentile(delta_vals, 95))

        # texture energy per region
        te = tex_energy[bi, 0]
        tex_ctx_vals = te[context_mask[bi, 0]].detach().cpu().numpy() if ctx_count > 0 else np.array([0.0])
        tex_center_vals = te[center_mask[bi, 0]].detach().cpu().numpy() if center_count > 0 else np.array([0.0])
        tex_seam_vals = te[seam_inner_mask[bi, 0]].detach().cpu().numpy() if seam_count > 0 else np.array([0.0])
        tex_hole_vals = te[hole_mask[bi, 0]].detach().cpu().numpy()

        tex_ctx_m = float(tex_ctx_vals.mean())
        tex_center_m = float(tex_center_vals.mean()) if center_count > 0 else 0.0
        tex_seam_m = float(tex_seam_vals.mean())
        tex_hole_m = float(tex_hole_vals.mean())
        ratio_center = tex_center_m / max(tex_ctx_m, 1e-8)
        ratio_seam = tex_seam_m / max(tex_ctx_m, 1e-8)

        print(f"[P2-A] {label}")
        print(f"  known_mae=0.00000000")
        print(f"  hole={h_count} seam_inner={seam_count} center={center_count} context={ctx_count}")
        print(f"  seam_grad_mean={seam_mean:.6f} seam_grad_p95={seam_p95:.6f}")
        print(f"  row_profile_delta_mean={delta_mean:.6f} row_profile_delta_p95={delta_p95:.6f}")
        print(f"  texture_energy_context={tex_ctx_m:.6f}")
        print(f"  texture_energy_hole_center={tex_center_m:.6f} texture_energy_seam={tex_seam_m:.6f}")
        print(f"  texture_energy_ratio_center={ratio_center:.4f} texture_energy_ratio_seam={ratio_seam:.4f}")

    # ---- 6. Save debug images ----
    _save_p2a_debug(
        baseline_current, hole_mask, bmasks, seam_diff, tex_energy,
        profile_delta, profiles, gt_img, known_mask, save_dir, img_names,
    )


def _save_p2a_debug(
    baseline, hole_mask, bmasks, seam_diff, tex_energy, profile_delta, profiles,
    gt_img, known_mask, save_dir, img_names,
):
    """Save all P2-A-Revise debug images."""
    os.makedirs(save_dir, exist_ok=True)

    # ---- binary masks ----
    masks_to_save = {
        "p2a_hole_mask": hole_mask,
        "p2a_boundary_ring_mask": bmasks["p2a_boundary_ring_mask"],
        "p2a_left_seam_mask": bmasks["p2a_left_seam_mask"],
        "p2a_right_seam_mask": bmasks["p2a_right_seam_mask"],
        "p2a_hole_center_mask": bmasks["p2a_hole_center_mask"],
        "p2a_context_ring_mask": bmasks["p2a_context_ring_mask"],
    }
    for tag, tensor in masks_to_save.items():
        _prior_debug_save_batch(tensor.to(th.uint8) * 255, img_names=img_names,
                                out_dir=save_dir, tag=tag, mode="image")

    # ---- baseline ----
    _prior_debug_save_batch(baseline, img_names=img_names, out_dir=save_dir,
                            tag="p2a_baseline", mode="image")

    # ---- continuous heatmaps ----
    _save_continuous_heatmap(seam_diff, hole_mask, save_dir,
                             "p2a_seam_grad_diff_float", img_names)
    _save_continuous_heatmap(tex_energy, hole_mask, save_dir,
                             "p2a_texture_energy_float", img_names)
    _save_continuous_heatmap(tex_energy, known_mask, save_dir,
                             "p2a_texture_energy_context", img_names)
    _save_continuous_heatmap(tex_energy, bmasks["p2a_hole_center_mask"], save_dir,
                             "p2a_texture_energy_hole", img_names)

    # ---- distance map ----
    _save_continuous_heatmap(bmasks["p2a_dist_map"], hole_mask, save_dir,
                             "p2a_dist_map", img_names)

    # ---- row profile visualization ----
    _save_row_profile_curve(profiles, hole_mask, save_dir,
                            "p2a_row_profile_curve", img_names)
    _save_continuous_heatmap(
        profile_delta.unsqueeze(1) if profile_delta.dim() == 3 else profile_delta,
        hole_mask, save_dir, "p2a_row_profile_diff", img_names,
    )

    # ---- grid ----
    _save_p2a_grid(baseline, hole_mask, bmasks, seam_diff, tex_energy,
                   gt_img, save_dir, img_names)


def _save_continuous_heatmap(data, mask, save_dir, tag, img_names=None):
    """Save continuous intensity heatmap (non-binarized)."""
    if data.dim() == 4 and data.shape[1] > 1:
        data = data.mean(dim=1, keepdim=True)
    m = _ensure_bool_mask(mask, data).float()
    d = data * m
    dmax = d.max().clamp_min(1e-8)
    d_u8 = (d / dmax * 255).clamp(0, 255).to(th.uint8)
    # 3ch
    d_3ch = d_u8.expand(-1, 3, -1, -1)
    _prior_debug_save_batch(d_3ch, img_names=img_names, out_dir=save_dir,
                            tag=tag, mode="image")


def _save_row_profile_curve(profiles, hole_mask, save_dir, tag, img_names=None):
    """Save row profile curve plot (per-row left_ctx / right_ctx / hole_mean)."""
    from PIL import Image as PILImage

    B, _, H = profiles.shape
    for bi in range(B):
        left = profiles[bi, 0].cpu().numpy()
        right = profiles[bi, 1].cpu().numpy()
        hole_mean = profiles[bi, 2].cpu().numpy()

        # Plot curves
        fig_h, fig_w = H, 512
        canvas = np.ones((fig_h, fig_w, 3), dtype=np.uint8) * 255

        def _plot_line(vals, color):
            v = np.clip(vals, -1, 1)
            xs = ((v + 1) / 2 * (fig_w - 1)).astype(int)
            for y in range(H):
                x = min(max(xs[y], 0), fig_w - 1)
                canvas[y, max(0, x-1):min(fig_w, x+2)] = color

        _plot_line(left, [0, 150, 0])      # green = left context
        _plot_line(right, [0, 0, 200])     # blue = right context
        _plot_line(hole_mean, [200, 0, 0]) # red = hole mean

        out_img = PILImage.fromarray(canvas)
        if img_names and bi < len(img_names):
            stem = _prior_debug_safe_name(img_names[bi])
        else:
            stem = f"img{bi}"
        out_path = os.path.join(save_dir, f"{stem}_{tag}.png")
        out_img.save(out_path)



# ============================================================
# P4-1a: Stat-v2 Multi-scale Texture Guidance Loss
# ============================================================
def p7_layer_base_candidate(baseline_output, gt, gt_keep_mask, conf):
    """
    Apply D2-2R layer repair to baseline output, producing a candidate.
    Returns (candidate_tensor, debug_dict).
    Does NOT modify final.
    """
    from scipy.ndimage import gaussian_filter, distance_transform_edt, uniform_filter

    if baseline_output.dim() == 4 and baseline_output.shape[1] > 1:
        gray = baseline_output.mean(dim=1, keepdim=True)
    else:
        gray = baseline_output

    hole = _ensure_bool_mask(1.0 - gt_keep_mask.float(), gray).float()
    known = _ensure_bool_mask(gt_keep_mask.float(), gray).float()

    ctx_width = int(conf.get("layer_v3r_ctx_width", 20))
    target_sigma = float(conf.get("layer_v3r_target_sigma", 15.0))
    seam_guard_width = int(conf.get("layer_v3r_seam_guard_width", 6))
    conf_dist_scale = float(conf.get("layer_v3r_conf_dist_scale", 20.0))
    conf_min_ctx = int(conf.get("layer_v3r_conf_min_ctx", 3))
    strength = float(conf.get("p7_layer_base_strength", 0.30))

    B, _, H, W = gray.shape
    candidate = baseline_output.clone()

    debug = {
        "trend_target_map": th.zeros_like(gray),
        "confidence_map": th.zeros_like(gray),
        "seam_guard_map": th.zeros_like(gray),
        "effective_weight_map": th.zeros_like(gray),
        "trigger_type": [],
        "left_ctx_count": [],
        "right_ctx_count": [],
        "total_ctx_count": [],
    }

    for b in range(B):
        gray_np = gray[b, 0].detach().cpu().numpy()
        hole_np = hole[b, 0].detach().cpu().numpy()
        known_np = known[b, 0].detach().cpu().numpy()

        if hole_np.sum() < 1:
            debug["trigger_type"].append("none")
            debug["left_ctx_count"].append(0)
            debug["right_ctx_count"].append(0)
            debug["total_ctx_count"].append(0)
            continue

        # === Step 1: 2D mask-aware Gaussian target ===
        known_field = gray_np * known_np
        known_weight = known_np.astype(np.float32)
        field_blur = gaussian_filter(known_field, sigma=target_sigma)
        weight_blur = gaussian_filter(known_weight, sigma=target_sigma)
        target_2d = np.where(weight_blur > 1e-6, field_blur / weight_blur, 0.0)

        # === Step 2: Per-pixel confidence ===
        dist_to_known = distance_transform_edt(1 - known_np)
        ctx_count_map = uniform_filter(known_np.astype(np.float32), size=ctx_width * 2 + 1) * ((ctx_width * 2 + 1) ** 2)

        boundary_penalty = np.ones((H, W), dtype=np.float32)
        for y in range(H):
            xs_hole = np.where(hole_np[y] > 0.5)[0]
            if len(xs_hole) == 0:
                continue
            x_left = int(xs_hole[0])
            x_right = int(xs_hole[-1])
            if x_left == 0:
                boundary_penalty[y, :x_right + 1] *= 0.3
            if x_right == W - 1:
                boundary_penalty[y, x_left:] *= 0.3

        dist_conf = np.exp(-dist_to_known / max(conf_dist_scale, 1.0))
        ctx_conf = np.clip(ctx_count_map / max(conf_min_ctx * 2, 1), 0.0, 1.0)
        confidence = dist_conf * ctx_conf * boundary_penalty * hole_np

        # === Step 3: Seam guard ===
        seam_guard = np.ones((H, W), dtype=np.float32)
        for y in range(H):
            xs_hole = np.where(hole_np[y] > 0.5)[0]
            if len(xs_hole) == 0:
                continue
            x_left = int(xs_hole[0])
            x_right = int(xs_hole[-1])
            for dx in range(min(seam_guard_width, x_right - x_left + 1)):
                w = (float(dx) / max(seam_guard_width, 1)) ** 2
                seam_guard[y, x_left + dx] = min(seam_guard[y, x_left + dx], w)
                seam_guard[y, x_right - dx] = min(seam_guard[y, x_right - dx], w)

        # === Step 4: Effective weight and candidate ===
        effective_weight = confidence * seam_guard * hole_np * strength

        # Apply: candidate = baseline * (1 - weight) + target * weight
        candidate_np = gray_np * (1 - effective_weight) + target_2d * effective_weight
        candidate_np = np.clip(candidate_np, -1.0, 1.0)

        # Write back to candidate tensor
        if candidate.shape[1] == 1:
            candidate[b, 0] = th.tensor(candidate_np, dtype=candidate.dtype, device=candidate.device)
        else:
            for c in range(candidate.shape[1]):
                candidate[b, c] = th.tensor(candidate_np, dtype=candidate.dtype, device=candidate.device)

        # === Step 5: Trigger classification ===
        # Count left/right context per row
        left_counts = []
        right_counts = []
        for y in range(H):
            xs_hole = np.where(hole_np[y] > 0.5)[0]
            if len(xs_hole) == 0:
                continue
            x_left = int(xs_hole[0])
            x_right = int(xs_hole[-1])
            l_ctx = int(known_np[y, max(0, x_left - ctx_width):x_left].sum())
            r_ctx = int(known_np[y, x_right + 1:min(W, x_right + 1 + ctx_width)].sum())
            left_counts.append(l_ctx)
            right_counts.append(r_ctx)

        avg_left = np.mean(left_counts) if left_counts else 0
        avg_right = np.mean(right_counts) if right_counts else 0
        total_ctx = avg_left + avg_right

        # Trigger classification: bilateral vs single-sided
        has_left = avg_left >= conf_min_ctx
        has_right = avg_right >= conf_min_ctx
        if has_left and has_right:
            trigger = "bilateral_strong"
        elif has_left or has_right:
            if total_ctx >= conf_min_ctx * 2:
                trigger = "single_sided_strong"
            else:
                trigger = "single_sided_weak"
        else:
            trigger = "none"

        debug["trigger_type"].append(trigger)
        debug["left_ctx_count"].append(int(avg_left))
        debug["right_ctx_count"].append(int(avg_right))
        debug["total_ctx_count"].append(int(total_ctx))

        # Write debug maps
        debug["trend_target_map"][b, 0] = th.tensor(target_2d, dtype=gray.dtype, device=gray.device)
        debug["confidence_map"][b, 0] = th.tensor(confidence, dtype=gray.dtype, device=gray.device)
        debug["seam_guard_map"][b, 0] = th.tensor(seam_guard, dtype=gray.dtype, device=gray.device)
        debug["effective_weight_map"][b, 0] = th.tensor(effective_weight, dtype=gray.dtype, device=gray.device)

    return candidate, debug


def run_p7_layer_base_diagnostics(
    baseline_output,
    gt,
    gt_keep_mask,
    img_names,
    save_dir,
    conf,
):
    """
    P7-1: Generate layer_base candidate and compare with baseline.
    Outputs candidate, diff, metrics CSV, summary.
    """
    import csv as _csv

    os.makedirs(save_dir, exist_ok=True)

    # Generate candidate
    candidate, p7_debug = p7_layer_base_candidate(baseline_output, gt, gt_keep_mask, conf)

    # Compute metrics for both
    baseline_metrics = _compute_per_image_metrics(baseline_output, gt, gt_keep_mask, conf)
    candidate_metrics = _compute_per_image_metrics(candidate, gt, gt_keep_mask, conf)

    hole_mask = _ensure_bool_mask(1.0 - gt_keep_mask.float(), baseline_output)
    B = baseline_output.shape[0]

    csv_rows = []
    for bi in range(B):
        name = img_names[bi] if img_names and bi < len(img_names) else f"img{bi}"
        stem = _prior_debug_safe_name(name)

        h_count = int(hole_mask[bi].sum().item())
        if h_count == 0:
            continue

        bm = baseline_metrics[bi]
        cm = candidate_metrics[bi]

        trigger = p7_debug["trigger_type"][bi] if bi < len(p7_debug["trigger_type"]) else "none"
        l_ctx = p7_debug["left_ctx_count"][bi] if bi < len(p7_debug["left_ctx_count"]) else 0
        r_ctx = p7_debug["right_ctx_count"][bi] if bi < len(p7_debug["right_ctx_count"]) else 0
        t_ctx = p7_debug["total_ctx_count"][bi] if bi < len(p7_debug["total_ctx_count"]) else 0

        # Save candidate image
        _prior_debug_save_batch(
            candidate[bi:bi+1], [name], save_dir,
            "p7_layer_base_candidate", mode="image",
        )

        # Save diff (hole region only)
        diff = (candidate[bi:bi+1] - baseline_output[bi:bi+1]).abs() * hole_mask[bi:bi+1]
        _prior_debug_save_batch(
            diff, [name], save_dir,
            "p7_layer_base_diff", mode="auto",
        )

        # Save debug maps
        for tag, key, mode in [
            ("p7_layer_base_trend_target", "trend_target_map", "image"),
            ("p7_layer_base_confidence", "confidence_map", "mask"),
            ("p7_layer_base_seam_guard", "seam_guard_map", "mask"),
            ("p7_layer_base_effective_weight", "effective_weight_map", "mask"),
        ]:
            _prior_debug_save_batch(
                p7_debug[key][bi:bi+1], [name], save_dir, tag, mode=mode,
            )

        delta_rd = cm["row_delta_mean"] - bm["row_delta_mean"]
        delta_tex = cm["tex_ratio_center"] - bm["tex_ratio_center"]
        delta_seam = cm["seam_grad_mean"] - bm["seam_grad_mean"]

        has_left = l_ctx >= 3
        has_right = r_ctx >= 3
        is_bilateral = "1" if (has_left and has_right) else "0"
        context_side = "bilateral" if (has_left and has_right) else ("left" if has_left else ("right" if has_right else "none"))

        csv_rows.append({
            "sample_name": name,
            "trigger_valid": "1" if trigger != "none" else "0",
            "trigger_type": trigger,
            "context_side": context_side,
            "is_bilateral": is_bilateral,
            "row_delta_baseline": f"{bm['row_delta_mean']:.6f}",
            "row_delta_layer_base": f"{cm['row_delta_mean']:.6f}",
            "delta_row_delta": f"{delta_rd:.6f}",
            "tex_ratio_center_baseline": f"{bm['tex_ratio_center']:.6f}",
            "tex_ratio_center_layer_base": f"{cm['tex_ratio_center']:.6f}",
            "delta_tex_ratio_center": f"{delta_tex:.6f}",
            "seam_grad_baseline": f"{bm['seam_grad_mean']:.6f}",
            "seam_grad_layer_base": f"{cm['seam_grad_mean']:.6f}",
            "delta_seam_grad": f"{delta_seam:.6f}",
            "known_mae": f"{cm['known_mae']:.8f}",
            "nan_detected": "false",
            "left_ctx_count": str(l_ctx),
            "right_ctx_count": str(r_ctx),
            "total_ctx_count": str(t_ctx),
        })

        print(f"[P7-1] {name}: trigger={trigger} side={context_side} bilateral={is_bilateral} "
              f"ctx={l_ctx}/{r_ctx}/{t_ctx} "
              f"rd={bm['row_delta_mean']:.3f}→{cm['row_delta_mean']:.3f} (Δ{delta_rd:+.3f}) "
              f"tex={bm['tex_ratio_center']:.4f}→{cm['tex_ratio_center']:.4f} (Δ{delta_tex:+.4f}) "
              f"seam={bm['seam_grad_mean']:.4f}→{cm['seam_grad_mean']:.4f} (Δ{delta_seam:+.4f})")

    return csv_rows


# ============================================================
# P7-2: Texture-Preserving Layer Guidance
# ============================================================

def p7_layer_texture_safe_candidate(baseline_output, layer_base_output, gt, gt_keep_mask, conf):
    """
    P7-2: Preserve baseline high-frequency texture while adding layer_base low-frequency trend.
    candidate = base + alpha * (layer_low - base_low)
    Returns (candidate_tensor, debug_dict).
    """
    from scipy.ndimage import gaussian_filter

    alpha = float(conf.get("p7_layer_texture_safe_alpha", 0.60))
    sigma = float(conf.get("p7_layer_texture_safe_sigma", 4.0))

    if baseline_output.dim() == 4 and baseline_output.shape[1] > 1:
        base_gray = baseline_output.mean(dim=1, keepdim=True)
        layer_gray = layer_base_output.mean(dim=1, keepdim=True)
    else:
        base_gray = baseline_output
        layer_gray = layer_base_output

    hole = _ensure_bool_mask(1.0 - gt_keep_mask.float(), base_gray).float()

    B, _, H, W = base_gray.shape
    candidate = baseline_output.clone()

    debug = {
        "layer_low_component": th.zeros_like(base_gray),
        "base_high_component": th.zeros_like(base_gray),
    }

    for b in range(B):
        base_np = base_gray[b, 0].detach().cpu().numpy()
        layer_np = layer_gray[b, 0].detach().cpu().numpy()
        hole_np = hole[b, 0].detach().cpu().numpy()

        if hole_np.sum() < 1:
            continue

        # Low-frequency components
        base_low = gaussian_filter(base_np, sigma=sigma)
        layer_low = gaussian_filter(layer_np, sigma=sigma)

        # High-frequency from baseline (preserved)
        base_high = base_np - base_low

        # Candidate: base + alpha * (layer_low - base_low)
        # This preserves base_high while blending the low-frequency improvement
        candidate_np = base_np + alpha * (layer_low - base_low)
        candidate_np = np.clip(candidate_np, -1.0, 1.0)

        # Known region hard merge: restore original values in known region
        known_np = 1.0 - hole_np
        candidate_np = known_np * base_np + hole_np * candidate_np

        # Write back to candidate tensor
        if candidate.shape[1] == 1:
            candidate[b, 0] = th.tensor(candidate_np, dtype=candidate.dtype, device=candidate.device)
        else:
            for c in range(candidate.shape[1]):
                candidate[b, c] = th.tensor(candidate_np, dtype=candidate.dtype, device=candidate.device)

        # Debug maps
        debug["layer_low_component"][b, 0] = th.tensor(layer_low, dtype=base_gray.dtype, device=base_gray.device)
        debug["base_high_component"][b, 0] = th.tensor(base_high, dtype=base_gray.dtype, device=base_gray.device)

    return candidate, debug


def run_p7_layer_texture_safe_diagnostics(
    baseline_output,
    layer_base_output,
    gt,
    gt_keep_mask,
    img_names,
    save_dir,
    conf,
    layer_base_debug=None,
):
    """
    P7-2: Generate texture-safe candidate and compare with baseline and layer_base.
    """
    import csv as _csv

    os.makedirs(save_dir, exist_ok=True)

    alpha = float(conf.get("p7_layer_texture_safe_alpha", 0.60))
    sigma = float(conf.get("p7_layer_texture_safe_sigma", 4.0))

    # Generate candidate
    candidate, p7_debug = p7_layer_texture_safe_candidate(
        baseline_output, layer_base_output, gt, gt_keep_mask, conf
    )

    # Compute metrics for all three
    baseline_metrics = _compute_per_image_metrics(baseline_output, gt, gt_keep_mask, conf)
    layer_metrics = _compute_per_image_metrics(layer_base_output, gt, gt_keep_mask, conf)
    candidate_metrics = _compute_per_image_metrics(candidate, gt, gt_keep_mask, conf)

    hole_mask = _ensure_bool_mask(1.0 - gt_keep_mask.float(), baseline_output)
    B = baseline_output.shape[0]

    csv_rows = []
    for bi in range(B):
        name = img_names[bi] if img_names and bi < len(img_names) else f"img{bi}"
        stem = _prior_debug_safe_name(name)

        h_count = int(hole_mask[bi].sum().item())
        if h_count == 0:
            continue

        bm = baseline_metrics[bi]
        lm = layer_metrics[bi]
        cm = candidate_metrics[bi]

        # Save candidate
        _prior_debug_save_batch(
            candidate[bi:bi+1], [name], save_dir,
            "p7_layer_texture_safe_candidate", mode="image",
        )

        # Save diff vs baseline
        diff = (candidate[bi:bi+1] - baseline_output[bi:bi+1]).abs() * hole_mask[bi:bi+1]
        _prior_debug_save_batch(
            diff, [name], save_dir,
            "p7_layer_texture_safe_diff", mode="auto",
        )

        # Save debug maps
        for tag, key in [
            ("p7_layer_low_component", "layer_low_component"),
            ("p7_base_high_component", "base_high_component"),
        ]:
            _prior_debug_save_batch(
                p7_debug[key][bi:bi+1], [name], save_dir, tag, mode="image",
            )

        # Inherit context metadata from layer_base_debug if available
        trigger = "none"
        context_side = "none"
        is_bilateral = "0"
        if layer_base_debug is not None:
            trigger_list = layer_base_debug.get("trigger_type", [])
            side_list = layer_base_debug.get("left_ctx_count", [])
            right_list = layer_base_debug.get("right_ctx_count", [])
            if bi < len(trigger_list):
                trigger = trigger_list[bi]
                l_ctx = side_list[bi] if bi < len(side_list) else 0
                r_ctx = right_list[bi] if bi < len(right_list) else 0
                has_left = l_ctx >= 3
                has_right = r_ctx >= 3
                context_side = "bilateral" if (has_left and has_right) else ("left" if has_left else ("right" if has_right else "none"))
                is_bilateral = "1" if (has_left and has_right) else "0"

        csv_rows.append({
            "sample_name": name,
            "trigger_valid": "1" if trigger != "none" else "0",
            "trigger_type": trigger,
            "context_side": context_side,
            "is_bilateral": is_bilateral,
            "alpha": f"{alpha:.2f}",
            "sigma": f"{sigma:.1f}",
            "row_delta_baseline": f"{bm['row_delta_mean']:.6f}",
            "row_delta_layer_base": f"{lm['row_delta_mean']:.6f}",
            "row_delta_texture_safe": f"{cm['row_delta_mean']:.6f}",
            "delta_row_delta_vs_baseline": f"{cm['row_delta_mean'] - bm['row_delta_mean']:.6f}",
            "delta_row_delta_vs_layer_base": f"{cm['row_delta_mean'] - lm['row_delta_mean']:.6f}",
            "tex_ratio_center_baseline": f"{bm['tex_ratio_center']:.6f}",
            "tex_ratio_center_layer_base": f"{lm['tex_ratio_center']:.6f}",
            "tex_ratio_center_texture_safe": f"{cm['tex_ratio_center']:.6f}",
            "delta_tex_vs_baseline": f"{cm['tex_ratio_center'] - bm['tex_ratio_center']:.6f}",
            "delta_tex_vs_layer_base": f"{cm['tex_ratio_center'] - lm['tex_ratio_center']:.6f}",
            "seam_grad_baseline": f"{bm['seam_grad_mean']:.6f}",
            "seam_grad_layer_base": f"{lm['seam_grad_mean']:.6f}",
            "seam_grad_texture_safe": f"{cm['seam_grad_mean']:.6f}",
            "delta_seam_vs_baseline": f"{cm['seam_grad_mean'] - bm['seam_grad_mean']:.6f}",
            "delta_seam_vs_layer_base": f"{cm['seam_grad_mean'] - lm['seam_grad_mean']:.6f}",
            "known_mae": f"{cm['known_mae']:.8f}",
            "nan_detected": "false",
        })

        print(f"[P7-2] {name}: "
              f"rd: {bm['row_delta_mean']:.3f}→{lm['row_delta_mean']:.3f}→{cm['row_delta_mean']:.3f} "
              f"tex: {bm['tex_ratio_center']:.4f}→{lm['tex_ratio_center']:.4f}→{cm['tex_ratio_center']:.4f} "
              f"seam: {bm['seam_grad_mean']:.4f}→{lm['seam_grad_mean']:.4f}→{cm['seam_grad_mean']:.4f}")

    return csv_rows


# ============================================================
# P7-3: Confidence Gate
# ============================================================

def p7_confidence_gate_candidate(
    baseline_output,
    texture_safe_output,
    gt,
    gt_keep_mask,
    layer_base_debug,
    conf,
):
    """
    P7-3: Apply confidence gating to texture_safe candidate.
    Low-confidence regions revert to baseline.
    Returns (candidate_tensor, confidence_map, debug_dict).
    """
    from scipy.ndimage import gaussian_filter

    if baseline_output.dim() == 4 and baseline_output.shape[1] > 1:
        base_gray = baseline_output.mean(dim=1, keepdim=True)
        safe_gray = texture_safe_output.mean(dim=1, keepdim=True)
    else:
        base_gray = baseline_output
        safe_gray = texture_safe_output

    hole = _ensure_bool_mask(1.0 - gt_keep_mask.float(), base_gray).float()
    known = _ensure_bool_mask(gt_keep_mask.float(), base_gray).float()

    seam_guard_width = int(conf.get("layer_v3r_seam_guard_width", 6))
    conf_min_ctx = int(conf.get("layer_v3r_conf_min_ctx", 3))

    B, _, H, W = base_gray.shape
    candidate = baseline_output.clone()
    conf_map = th.zeros_like(base_gray)

    debug = {"confidence_map": conf_map.clone()}

    for b in range(B):
        base_np = base_gray[b, 0].detach().cpu().numpy()
        safe_np = safe_gray[b, 0].detach().cpu().numpy()
        hole_np = hole[b, 0].detach().cpu().numpy()
        known_np = known[b, 0].detach().cpu().numpy()

        if hole_np.sum() < 1:
            continue

        # Per-image trigger info from layer_base_debug
        trigger_list = layer_base_debug.get("trigger_type", [])
        left_list = layer_base_debug.get("left_ctx_count", [])
        right_list = layer_base_debug.get("right_ctx_count", [])

        trigger = trigger_list[b] if b < len(trigger_list) else "none"
        l_ctx = left_list[b] if b < len(left_list) else 0
        r_ctx = right_list[b] if b < len(right_list) else 0

        # === Step 1: Base confidence from trigger type ===
        if trigger == "bilateral_strong":
            base_conf = 1.0
        elif trigger == "single_sided_strong":
            base_conf = 0.6
        elif trigger == "single_sided_weak":
            base_conf = 0.3
        else:
            base_conf = 0.0

        # === Step 2: Per-row context support ===
        row_conf = np.zeros((H, W), dtype=np.float32)
        for y in range(H):
            xs_hole = np.where(hole_np[y] > 0.5)[0]
            if len(xs_hole) == 0:
                continue
            x_left = int(xs_hole[0])
            x_right = int(xs_hole[-1])

            l_count = int(known_np[y, max(0, x_left - 20):x_left].sum())
            r_count = int(known_np[y, x_right + 1:min(W, x_right + 21)].sum())
            total_ctx = l_count + r_count

            # Row-level confidence: more context = higher confidence
            row_c = np.clip(total_ctx / max(conf_min_ctx * 4, 1), 0.0, 1.0)
            row_conf[y, x_left:x_right + 1] = row_c

        # === Step 3: Seam guard (reduce confidence near boundaries) ===
        seam_guard = np.ones((H, W), dtype=np.float32)
        for y in range(H):
            xs_hole = np.where(hole_np[y] > 0.5)[0]
            if len(xs_hole) == 0:
                continue
            x_left = int(xs_hole[0])
            x_right = int(xs_hole[-1])
            for dx in range(min(seam_guard_width, x_right - x_left + 1)):
                w = (float(dx) / max(seam_guard_width, 1)) ** 2
                seam_guard[y, x_left + dx] = min(seam_guard[y, x_left + dx], w)
                seam_guard[y, x_right - dx] = min(seam_guard[y, x_right - dx], w)

        # === Step 4: Combined confidence ===
        confidence = base_conf * row_conf * seam_guard * hole_np

        # Apply: candidate = baseline * (1 - conf) + texture_safe * conf
        candidate_np = base_np * (1.0 - confidence) + safe_np * confidence
        candidate_np = np.clip(candidate_np, -1.0, 1.0)

        # Known region hard merge
        candidate_np = known_np * base_np + hole_np * candidate_np

        # Write back
        if candidate.shape[1] == 1:
            candidate[b, 0] = th.tensor(candidate_np, dtype=candidate.dtype, device=candidate.device)
        else:
            for c in range(candidate.shape[1]):
                candidate[b, c] = th.tensor(candidate_np, dtype=candidate.dtype, device=candidate.device)

        conf_map[b, 0] = th.tensor(confidence, dtype=base_gray.dtype, device=base_gray.device)

    debug["confidence_map"] = conf_map
    return candidate, conf_map, debug


def run_p7_confidence_gate_diagnostics(
    baseline_output,
    texture_safe_output,
    gt,
    gt_keep_mask,
    img_names,
    save_dir,
    conf,
    layer_base_debug,
):
    """
    P7-3: Confidence gate diagnostics.
    """
    import csv as _csv

    os.makedirs(save_dir, exist_ok=True)

    # Generate confidence-gated candidate
    candidate, conf_map, p7_debug = p7_confidence_gate_candidate(
        baseline_output, texture_safe_output, gt, gt_keep_mask, layer_base_debug, conf
    )

    # Compute metrics
    baseline_metrics = _compute_per_image_metrics(baseline_output, gt, gt_keep_mask, conf)
    safe_metrics = _compute_per_image_metrics(texture_safe_output, gt, gt_keep_mask, conf)
    conf_metrics = _compute_per_image_metrics(candidate, gt, gt_keep_mask, conf)

    hole_mask = _ensure_bool_mask(1.0 - gt_keep_mask.float(), baseline_output)
    B = baseline_output.shape[0]

    csv_rows = []
    for bi in range(B):
        name = img_names[bi] if img_names and bi < len(img_names) else f"img{bi}"
        stem = _prior_debug_safe_name(name)

        h_count = int(hole_mask[bi].sum().item())
        if h_count == 0:
            continue

        bm = baseline_metrics[bi]
        sm = safe_metrics[bi]
        cm = conf_metrics[bi]

        # Inherit metadata
        trigger_list = layer_base_debug.get("trigger_type", [])
        left_list = layer_base_debug.get("left_ctx_count", [])
        right_list = layer_base_debug.get("right_ctx_count", [])
        trigger = trigger_list[bi] if bi < len(trigger_list) else "none"
        l_ctx = left_list[bi] if bi < len(left_list) else 0
        r_ctx = right_list[bi] if bi < len(right_list) else 0
        has_left = l_ctx >= 3
        has_right = r_ctx >= 3
        context_side = "bilateral" if (has_left and has_right) else ("left" if has_left else ("right" if has_right else "none"))
        is_bilateral = "1" if (has_left and has_right) else "0"

        # Confidence stats
        conf_vals = conf_map[bi, 0][hole_mask[bi, 0]].detach().cpu().numpy()
        conf_mean = float(conf_vals.mean()) if len(conf_vals) > 0 else 0.0
        conf_max = float(conf_vals.max()) if len(conf_vals) > 0 else 0.0

        # Update pixel count
        diff = (candidate[bi:bi+1] - baseline_output[bi:bi+1]).abs()
        update_pixels = int((diff[hole_mask[bi:bi+1]] > 1e-6).sum().item())
        update_ratio = update_pixels / max(h_count, 1)

        # Save outputs
        _prior_debug_save_batch(
            candidate[bi:bi+1], [name], save_dir,
            "p7_confidence_gate_candidate", mode="image",
        )
        _prior_debug_save_batch(
            conf_map[bi:bi+1], [name], save_dir,
            "p7_confidence_map", mode="mask",
        )
        diff_vis = diff * hole_mask[bi:bi+1]
        _prior_debug_save_batch(
            diff_vis, [name], save_dir,
            "p7_confidence_gate_diff", mode="auto",
        )

        csv_rows.append({
            "sample_name": name,
            "trigger_type": trigger,
            "context_side": context_side,
            "is_bilateral": is_bilateral,
            "confidence_mean": f"{conf_mean:.6f}",
            "confidence_max": f"{conf_max:.6f}",
            "update_pixel_count": str(update_pixels),
            "update_ratio": f"{update_ratio:.6f}",
            "row_delta_baseline": f"{bm['row_delta_mean']:.6f}",
            "row_delta_conf_gate": f"{cm['row_delta_mean']:.6f}",
            "delta_row_delta_vs_baseline": f"{cm['row_delta_mean'] - bm['row_delta_mean']:.6f}",
            "tex_ratio_baseline": f"{bm['tex_ratio_center']:.6f}",
            "tex_ratio_conf_gate": f"{cm['tex_ratio_center']:.6f}",
            "delta_tex_vs_baseline": f"{cm['tex_ratio_center'] - bm['tex_ratio_center']:.6f}",
            "seam_grad_baseline": f"{bm['seam_grad_mean']:.6f}",
            "seam_grad_conf_gate": f"{cm['seam_grad_mean']:.6f}",
            "delta_seam_vs_baseline": f"{cm['seam_grad_mean'] - bm['seam_grad_mean']:.6f}",
            "known_mae": f"{cm['known_mae']:.8f}",
            "nan_detected": "false",
        })

        print(f"[P7-3] {name}: trigger={trigger} side={context_side} "
              f"conf={conf_mean:.3f}/{conf_max:.3f} "
              f"update_px={update_pixels}({update_ratio:.3f}) "
              f"rd={bm['row_delta_mean']:.3f}→{cm['row_delta_mean']:.3f} "
              f"tex={bm['tex_ratio_center']:.4f}→{cm['tex_ratio_center']:.4f}")

    return csv_rows


def run_p7_safety_gate_diagnostics(
    baseline_output,
    conf_gate_output,
    gt,
    gt_keep_mask,
    img_names,
    save_dir,
    conf,
    layer_base_debug,
    conf_gate_csv_rows=None,
):
    """
    P7-4: Safety gate validation.
    """
    import csv as _csv

    os.makedirs(save_dir, exist_ok=True)

    baseline_metrics = _compute_per_image_metrics(baseline_output, gt, gt_keep_mask, conf)
    conf_metrics = _compute_per_image_metrics(conf_gate_output, gt, gt_keep_mask, conf)

    hole_mask = _ensure_bool_mask(1.0 - gt_keep_mask.float(), baseline_output)
    B = baseline_output.shape[0]

    csv_rows = []
    group_stats = {}

    for bi in range(B):
        name = img_names[bi] if img_names and bi < len(img_names) else f"img{bi}"
        h_count = int(hole_mask[bi].sum().item())
        if h_count == 0:
            continue

        bm = baseline_metrics[bi]
        cm = conf_metrics[bi]

        # Metadata
        trigger_list = layer_base_debug.get("trigger_type", [])
        left_list = layer_base_debug.get("left_ctx_count", [])
        right_list = layer_base_debug.get("right_ctx_count", [])
        trigger = trigger_list[bi] if bi < len(trigger_list) else "none"
        l_ctx = left_list[bi] if bi < len(left_list) else 0
        r_ctx = right_list[bi] if bi < len(right_list) else 0
        has_left = l_ctx >= 3
        has_right = r_ctx >= 3
        context_side = "bilateral" if (has_left and has_right) else ("left" if has_left else ("right" if has_right else "none"))

        # Update pixel count
        diff = (conf_gate_output[bi:bi+1] - baseline_output[bi:bi+1]).abs()
        update_pixels = int((diff[hole_mask[bi:bi+1]] > 1e-6).sum().item())
        update_ratio = update_pixels / max(h_count, 1)

        # Safety checks
        known_mae_ok = cm["known_mae"] == 0
        nan_ok = True  # already checked
        tex_ok = cm["tex_ratio_center"] >= bm["tex_ratio_center"] * 0.95
        seam_ok = cm["seam_grad_mean"] <= bm["seam_grad_mean"] * 1.05

        csv_rows.append({
            "sample_name": name,
            "trigger_type": trigger,
            "context_side": context_side,
            "known_mae_ok": "1" if known_mae_ok else "0",
            "nan_ok": "1" if nan_ok else "0",
            "tex_ok": "1" if tex_ok else "0",
            "seam_ok": "1" if seam_ok else "0",
            "update_pixel_count": str(update_pixels),
            "update_ratio": f"{update_ratio:.6f}",
            "row_delta_baseline": f"{bm['row_delta_mean']:.6f}",
            "row_delta_conf_gate": f"{cm['row_delta_mean']:.6f}",
            "delta_row_delta": f"{cm['row_delta_mean'] - bm['row_delta_mean']:.6f}",
            "tex_ratio_baseline": f"{bm['tex_ratio_center']:.6f}",
            "tex_ratio_conf_gate": f"{cm['tex_ratio_center']:.6f}",
            "delta_tex": f"{cm['tex_ratio_center'] - bm['tex_ratio_center']:.6f}",
            "seam_grad_baseline": f"{bm['seam_grad_mean']:.6f}",
            "seam_grad_conf_gate": f"{cm['seam_grad_mean']:.6f}",
            "delta_seam": f"{cm['seam_grad_mean'] - bm['seam_grad_mean']:.6f}",
            "known_mae": f"{cm['known_mae']:.8f}",
        })

        # Group stats
        group_key = f"{trigger}_{context_side}"
        if group_key not in group_stats:
            group_stats[group_key] = {"count": 0, "rd_delta": [], "tex_delta": [], "seam_delta": [], "update_px": []}
        group_stats[group_key]["count"] += 1
        group_stats[group_key]["rd_delta"].append(cm["row_delta_mean"] - bm["row_delta_mean"])
        group_stats[group_key]["tex_delta"].append(cm["tex_ratio_center"] - bm["tex_ratio_center"])
        group_stats[group_key]["seam_delta"].append(cm["seam_grad_mean"] - bm["seam_grad_mean"])
        group_stats[group_key]["update_px"].append(update_pixels)

    return csv_rows, group_stats


def _compute_per_image_metrics(tensor, gt, gt_keep_mask, conf):
    """Compute per-image diagnostic metrics. Returns list of dicts."""
    tex_sigma = float(conf.get("p2a_texture_sigma", 2.0))
    ctx_width = int(conf.get("p2a_context_width", 15))
    inner_w = int(conf.get("p2a_seam_inner_width", 4))
    outer_w = int(conf.get("p2a_seam_outer_width", 4))
    center_top_ratio = float(conf.get("p2a_center_top_ratio", 0.2))
    center_min_pixels = int(conf.get("p2a_center_min_pixels", 16))

    hole_mask = _ensure_bool_mask(1.0 - gt_keep_mask.float(), tensor)

    bmasks = _build_boundary_masks(
        hole_mask, inner_w, outer_w,
        center_mode="distance", center_top_ratio=center_top_ratio,
        center_min_pixels=center_min_pixels,
    )
    left_seam = bmasks["p2a_left_seam_mask"]
    right_seam = bmasks["p2a_right_seam_mask"]
    center_mask = bmasks["p2a_hole_center_mask"]
    context_mask = bmasks["p2a_context_ring_mask"]

    profiles = _compute_row_profile(tensor, hole_mask, ctx_width)
    ctx_mean = (profiles[:, 0:1, :] + profiles[:, 1:2, :]) / 2.0
    profile_delta = (profiles[:, 2:3, :] - ctx_mean).abs()
    tex_energy = _compute_texture_energy(tensor, tex_sigma)
    seam_diff = _compute_seam_grad(tensor, left_seam, right_seam, hole_mask)

    # known_mae
    known_mask = _ensure_bool_mask(gt_keep_mask.float(), tensor)
    known_mae = ((tensor - gt).abs() * known_mask.float()).sum() / known_mask.float().sum().clamp_min(1e-6)

    B = tensor.shape[0]
    results = []
    for bi in range(B):
        h_count = int(hole_mask[bi].sum().item())
        if h_count == 0:
            results.append({k: 0.0 for k in [
                "known_mae", "row_delta_mean", "tex_ratio_center", "tex_ratio_seam",
                "seam_grad_mean", "seam_grad_p95", "hole_center_mean", "hole_center_std",
                "context_mean", "context_std",
            ]})
            continue

        d = profile_delta[bi, 0].detach().cpu().numpy()
        t_hole = tex_energy[bi, 0][center_mask[bi, 0]].detach().cpu().numpy()
        t_seam = tex_energy[bi, 0][bmasks["p2a_seam_inner_mask"][bi, 0]].detach().cpu().numpy()
        sv = seam_diff[bi, 0][hole_mask[bi, 0]].detach().cpu().numpy()
        hc = tensor[bi].mean(dim=0)[hole_mask[bi, 0]].detach().cpu().numpy()
        cc = tensor[bi].mean(dim=0)[context_mask[bi, 0]].detach().cpu().numpy()

        results.append({
            "known_mae": float(known_mae.detach().cpu()),
            "row_delta_mean": float(d.mean()),
            "tex_ratio_center": float(t_hole.mean()) if len(t_hole) > 0 else 0.0,
            "tex_ratio_seam": float(t_seam.mean()) if len(t_seam) > 0 else 0.0,
            "seam_grad_mean": float(sv.mean()) if len(sv) > 0 else 0.0,
            "seam_grad_p95": float(np.percentile(sv, 95)) if len(sv) > 0 else 0.0,
            "hole_center_mean": float(hc.mean()) if len(hc) > 0 else 0.0,
            "hole_center_std": float(hc.std()) if len(hc) > 0 else 0.0,
            "context_mean": float(cc.mean()) if len(cc) > 0 else 0.0,
            "context_std": float(cc.std()) if len(cc) > 0 else 0.0,
        })
    return results


def _save_p2a_grid(baseline, hole_mask, bmasks, seam_diff, tex_energy, gt_img, save_dir, img_names):
    """Save P2-A 3x4 grid image."""
    from PIL import Image as PILImage

    B = baseline.shape[0]
    for bi in range(B):
        row1 = [baseline[bi], hole_mask.to(th.uint8).expand(-1, 3, -1, -1)[bi] * 255,
                bmasks["p2a_boundary_ring_mask"].to(th.uint8).expand(-1, 3, -1, -1)[bi] * 255,
                bmasks["p2a_hole_center_mask"].to(th.uint8).expand(-1, 3, -1, -1)[bi] * 255]

        # seam diff
        sd = seam_diff[bi:bi+1]
        if sd.shape[1] > 1:
            sd = sd.mean(dim=1, keepdim=True)
        sd = sd * hole_mask[bi:bi+1].float()
        sd_max = sd.max().clamp_min(1e-8)
        sd_u8 = (sd / sd_max * 255).clamp(0, 255).to(th.uint8)
        sd_u8 = sd_u8.expand(-1, 3, -1, -1)[0]  # [3, H, W]

        # tex energy
        te = tex_energy[bi:bi+1]
        if te.shape[1] > 1:
            te = te.mean(dim=1, keepdim=True)
        te = te * hole_mask[bi:bi+1].float()
        te_max = te.max().clamp_min(1e-8)
        te_u8 = (te / te_max * 255).clamp(0, 255).to(th.uint8)
        te_u8 = te_u8.expand(-1, 3, -1, -1)[0]  # [3, H, W]

        # left seam + right seam
        ls = bmasks["p2a_left_seam_mask"].to(th.uint8).expand(-1, 3, -1, -1)[bi] * 255
        rs = bmasks["p2a_right_seam_mask"].to(th.uint8).expand(-1, 3, -1, -1)[bi] * 255
        cr = bmasks["p2a_context_ring_mask"].to(th.uint8).expand(-1, 3, -1, -1)[bi] * 255

        row2 = [sd_u8, te_u8, ls, rs]

        # 3x4 grid
        _, _, H, W = baseline.shape
        cell_h, cell_w = H, W
        canvas = np.zeros((cell_h * 3, cell_w * 4, 3), dtype=np.uint8)

        def _to_arr(t):
            arr = t.detach().cpu().numpy()
            if arr.ndim == 3 and arr.shape[0] in (1, 3):
                arr = np.transpose(arr, (1, 2, 0))
            if arr.ndim == 3 and arr.shape[2] == 1:
                arr = np.concatenate([arr, arr, arr], axis=2)
            elif arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            if arr.dtype != np.uint8:
                arr = np.clip(arr * 127.5 + 127.5, 0, 255).astype(np.uint8)
            return np.array(PILImage.fromarray(arr).resize((cell_w, cell_h), PILImage.NEAREST))

        for c, t in enumerate(row1):
            canvas[0:cell_h, c*cell_w:(c+1)*cell_w] = _to_arr(t)
        for c, t in enumerate(row2):
            canvas[cell_h:cell_h*2, c*cell_w:(c+1)*cell_w] = _to_arr(t)
        # row 3: gt, context_ring, hole_mask color
        row3 = [gt_img[bi], cr, hole_mask.to(th.uint8).expand(-1, 3, -1, -1)[bi] * 255,
                baseline[bi]]  # placeholder
        for c, t in enumerate(row3):
            canvas[cell_h*2:cell_h*3, c*cell_w:(c+1)*cell_w] = _to_arr(t)

        out_img = PILImage.fromarray(canvas)
        if img_names and bi < len(img_names):
            stem = _prior_debug_safe_name(img_names[bi])
        else:
            stem = f"img{bi}"
        out_path = os.path.join(save_dir, f"{stem}_p2a_grid.png")
        os.makedirs(save_dir, exist_ok=True)
        out_img.save(out_path)


def _robust_anchor_mean_1d(vals):
    """vals: list[Tensor[C]], returns trimmed robust mean."""
    if len(vals) == 0:
        return None
    stack = th.stack(vals, dim=0)
    if stack.shape[0] >= 5:
        bright = stack.mean(dim=1)
        idx = th.argsort(bright)
        stack = stack[idx[1:-1]]
    return stack.mean(dim=0)


def _rowwise_prefill(gt, keep_mask, fill_mask, anchor_k=5, medium_smooth=False):
    """
    gt: [B,C,H,W], [-1,1]
    keep_mask: [B,1,H,W], 1=known region
    fill_mask: [B,1,H,W], 1=hole region to be filled by this function
    return: filled image, known region keeps gt, fill_mask region is deterministically filled.
    """
    gt_cpu = gt.detach().cpu()
    keep_cpu = keep_mask.detach().cpu() > 0.5
    fill_cpu = fill_mask.detach().cpu() > 0.5

    B, C, H, W = gt_cpu.shape
    out = gt_cpu.clone()
    k = max(1, int(anchor_k))

    for b in range(B):
        for y in range(H):
            row = fill_cpu[b, 0, y]
            x = 0
            while x < W:
                while x < W and not bool(row[x]):
                    x += 1
                if x >= W:
                    break
                s = x
                while x < W and bool(row[x]):
                    x += 1
                e = x
                width = e - s
                if width <= 0:
                    continue

                left_vals = []
                right_vals = []

                for ix in range(max(0, s - k), s):
                    if keep_cpu[b, 0, y, ix]:
                        left_vals.append(gt_cpu[b, :, y, ix])
                for ix in range(e, min(W, e + k)):
                    if keep_cpu[b, 0, y, ix]:
                        right_vals.append(gt_cpu[b, :, y, ix])

                left = _robust_anchor_mean_1d(left_vals)
                right = _robust_anchor_mean_1d(right_vals)

                if left is None and right is None:
                    continue
                elif left is not None and right is not None:
                    pos = th.arange(1, width + 1, dtype=gt_cpu.dtype).view(1, width)
                    alpha = pos / float(width + 1)
                    fill = left.view(C, 1) * (1.0 - alpha) + right.view(C, 1) * alpha
                elif left is not None:
                    fill = left.view(C, 1).repeat(1, width)
                else:
                    fill = right.view(C, 1).repeat(1, width)

                out[b, :, y, s:e] = fill.clamp(-1.0, 1.0)

    out = out.to(gt.device)

    if medium_smooth:
        smooth = F.avg_pool2d(out, kernel_size=(5, 1), stride=1, padding=(2, 0))
        m = fill_mask.to(out.device).float()
        out = out * (1.0 - 0.35 * m) + smooth * (0.35 * m)

    out = gt * keep_mask + out * (1.0 - keep_mask)
    return out.clamp(-1.0, 1.0)


def _harmonic_prefill(
    gt, keep_mask, init_img, fill_mask, num_iters=300, kernel_type="8conn"
):
    """Deterministic low-frequency diffusion fill for wide gaps."""
    u = init_img.clone()
    keep = keep_mask.float()
    fill = fill_mask.float()

    if str(kernel_type).lower() == "4conn":
        kernel = (
            th.tensor(
                [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
                dtype=u.dtype,
                device=u.device,
            )
            / 4.0
        )
    else:
        kernel = (
            th.tensor(
                [[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
                dtype=u.dtype,
                device=u.device,
            )
            / 8.0
        )

    C = u.shape[1]
    weight = kernel.view(1, 1, 3, 3).repeat(C, 1, 1, 1)

    for _ in range(int(num_iters)):
        avg = F.conv2d(u, weight, padding=1, groups=C)
        u = u * (1.0 - fill) + avg * fill
        u = gt * keep + u * (1.0 - keep)

    return u.clamp(-1.0, 1.0)


def _local_stat_match_prefill(gt, keep_mask, coarse, fill_mask, conf):
    """
    Coarse post-processing: local mean/std correction only in fill_mask missing region.

    Purpose: reduce gray columns, foggy vertical bands, and local contrast collapse in wide gap coarse_x0.
    Principle: sample a ring of known context near each missing pixel, compute local mean/std;
         then align the coarse local mean/std in the missing region to the context.

    Note: this is not a sampling prior, not used in cond_fn; it is a deterministic coarse correction.
    """
    keep = (keep_mask > 0.5).float()
    fill = (fill_mask > 0.5).float() * (1.0 - keep)

    if float(fill.sum().detach().cpu().item()) <= 0:
        return coarse, {}

    radius = int(conf.get("prefill_stat_context_radius", 9))
    radius = max(1, radius)
    blend = float(conf.get("prefill_stat_blend", 0.45))
    blend = max(0.0, min(1.0, blend))
    min_ctx = float(conf.get("prefill_stat_min_context_pixels", 8))
    std_floor = float(conf.get("prefill_stat_std_floor", 0.025))
    min_scale = float(conf.get("prefill_stat_min_scale", 0.75))
    max_scale = float(conf.get("prefill_stat_max_scale", 1.80))
    clamp_std = float(conf.get("prefill_stat_clamp_std", 2.50))

    k = 2 * radius + 1
    B, C, H, W = coarse.shape
    device = coarse.device
    dtype = coarse.dtype

    # Context: known region within radius around fill.
    # Note this is not all keep region, but keep region near the missing band, to avoid global statistics diluting local structure.
    context = _dilate_binary_mask(fill, radius) * keep

    kernel_1 = th.ones((1, 1, k, k), device=device, dtype=dtype)
    kernel_c = kernel_1.repeat(C, 1, 1, 1)

    context_count = F.conv2d(context, kernel_1, padding=radius).clamp_min(1.0)
    fill_count = F.conv2d(fill, kernel_1, padding=radius).clamp_min(1.0)

    context_c = context.expand(-1, C, -1, -1)
    fill_c = fill.expand(-1, C, -1, -1)

    ctx_sum = F.conv2d(gt * context_c, kernel_c, padding=radius, groups=C)
    ctx_sq_sum = F.conv2d(gt * gt * context_c, kernel_c, padding=radius, groups=C)
    ctx_mean = ctx_sum / context_count
    ctx_var = (ctx_sq_sum / context_count - ctx_mean * ctx_mean).clamp_min(0.0)
    ctx_std = th.sqrt(ctx_var + 1e-8)

    hole_sum = F.conv2d(coarse * fill_c, kernel_c, padding=radius, groups=C)
    hole_sq_sum = F.conv2d(coarse * coarse * fill_c, kernel_c, padding=radius, groups=C)
    hole_mean = hole_sum / fill_count
    hole_var = (hole_sq_sum / fill_count - hole_mean * hole_mean).clamp_min(0.0)
    hole_std = th.sqrt(hole_var + 1e-8)

    # Std alignment: only correct contrast collapse, prevent unlimited scale amplification.
    scale = ctx_std / hole_std.clamp_min(std_floor)
    scale = scale.clamp(min=min_scale, max=max_scale)

    matched = (coarse - hole_mean) * scale + ctx_mean

    # Prevent local statistics overshoot, clamp within context mean ± clamp_std * context_std.
    lower = ctx_mean - clamp_std * ctx_std.clamp_min(std_floor)
    upper = ctx_mean + clamp_std * ctx_std.clamp_min(std_floor)
    matched = th.max(th.min(matched, upper), lower)

    valid = ((context_count >= min_ctx).float() * fill).expand(-1, C, -1, -1)
    alpha = valid * blend

    out = coarse * (1.0 - alpha) + matched * alpha
    out = gt * keep + out * (1.0 - keep)
    out = out.clamp(-1.0, 1.0)

    debug = {
        "prefill_stat_context_mask": context,
        "prefill_stat_valid_mask": valid[:, :1],
        "prefill_stat_ctx_std": ctx_std.mean(dim=1, keepdim=True),
        "prefill_stat_hole_std": hole_std.mean(dim=1, keepdim=True),
        "prefill_stat_scale": scale.mean(dim=1, keepdim=True),
        "coarse_x0_before_stat": coarse,
        "coarse_x0_after_stat": out,
    }
    return out, debug


def build_coarse_prefill(gt, keep_mask, conf):
    """
    gt: [B,C,H,W], [-1,1]
    keep_mask: [B,1,H,W], 1=known region
    return: coarse_x0, debug_maps
    """
    keep = keep_mask.float()
    hole = 1.0 - keep
    coarse_wide_row_dbg = None
    coarse_wide_harmonic_dbg = None
    narrow_w = int(conf.get("prefill_narrow_width", 6))
    medium_w = int(conf.get("prefill_medium_width", 24))
    anchor_k = int(conf.get("prefill_anchor_k", 5))
    iters = int(conf.get("prefill_harmonic_iters", 300))
    kernel_type = str(conf.get("prefill_harmonic_kernel", "8conn"))
    dilate_px = int(conf.get("prefill_mask_dilate_for_coarse", 0))

    hole_for_coarse = hole
    if dilate_px > 0:
        hole_for_coarse = _dilate_binary_mask(hole_for_coarse, dilate_px)
    keep_for_coarse = 1.0 - hole_for_coarse

    narrow, medium, wide, width_map = _build_hole_width_masks(
        keep_for_coarse,
        narrow_width=narrow_w,
        medium_width=medium_w,
    )

    # Initial value: unknown region uses global known mean
    denom = keep.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
    mean_val = (gt * keep).sum(dim=(2, 3), keepdim=True) / denom
    coarse = gt * keep + mean_val * (1.0 - keep)

    # Narrow gap: robust left/right interpolation
    if float(narrow.sum().item()) > 0:
        coarse_narrow = _rowwise_prefill(
            gt, keep_for_coarse, narrow, anchor_k=anchor_k, medium_smooth=False
        )
        coarse = coarse * (1.0 - narrow) + coarse_narrow * narrow

    # Medium gap: left/right interpolation + slight vertical low-frequency smoothing
    if float(medium.sum().item()) > 0:
        coarse_medium = _rowwise_prefill(
            gt, keep_for_coarse, medium, anchor_k=anchor_k, medium_smooth=True
        )
        coarse = coarse * (1.0 - medium) + coarse_medium * medium

    # Wide gap: row-wise robust interpolation + harmonic low-frequency diffusion blend
    # Purpose:
    # 1. harmonic is stable but tends to produce gray vertical columns;
    # 2. row-wise better preserves horizontal bedding/light-dark continuity, but alone produces fake structures;
    # 3. blend both so coarse_x0 is no longer a pure gray fog band.
    if float(wide.sum().item()) > 0:
        # A. Wide gap row-wise draft: better preserves horizontal structure
        coarse_wide_row = _rowwise_prefill(
            gt,
            keep_for_coarse,
            wide,
            anchor_k=anchor_k,
            medium_smooth=False,
        )

        # B. Wide gap harmonic draft: more stable, but prone to gray columns
        coarse_wide_harmonic = _harmonic_prefill(
            gt=gt,
            keep_mask=keep_for_coarse,
            init_img=coarse_wide_row,
            fill_mask=wide,
            num_iters=iters,
            kernel_type=kernel_type,
        )

        # C. Blend ratio
        # row_ratio larger: horizontal structure more prominent, but may produce fake wedges;
        # row_ratio smaller: more stable, but gray columns more visible.
        row_ratio = float(conf.get("prefill_wide_row_ratio", 0.45))
        row_ratio = max(0.0, min(1.0, row_ratio))

        coarse_wide_mix = coarse_wide_row * row_ratio + coarse_wide_harmonic * (
            1.0 - row_ratio
        )

        coarse = coarse * (1.0 - wide) + coarse_wide_mix * wide
        # Only save debug when wide gaps exist
        coarse_wide_row_dbg = coarse_wide_row
        coarse_wide_harmonic_dbg = coarse_wide_harmonic
    # ------------------------------------------------------------
    # v3b: coarse local statistics correction.
    # Original v3 only applied to wide, but wide in _build_hole_width_masks is
    # "per-row horizontal contiguous missing width > prefill_medium_width".
    # Many visually wide/long vertical gaps, if each row's horizontal width <= medium_width,
    # are classified as medium, causing wide_sum=0 and stat correction not triggered.
    #
    # Therefore decouple stat_match region from wide, supporting:
    #   prefill_stat_apply_to: wide / medium_wide / all / width_threshold
    #   prefill_stat_min_width: select stat correction region by pixel width threshold
    # ------------------------------------------------------------
    stat_debug_maps = {}
    stat_fill = th.zeros_like(hole)
    stat_apply_to = str(conf.get("prefill_stat_apply_to", "medium_wide")).lower()
    stat_min_width = int(conf.get("prefill_stat_min_width", 0))

    if stat_apply_to in ("wide", "wide_only"):
        stat_fill = wide
    elif stat_apply_to in ("medium_wide", "medium+wide", "not_narrow"):
        stat_fill = (medium + wide).clamp(0.0, 1.0)
    elif stat_apply_to in ("all", "hole", "all_hole"):
        stat_fill = hole
    elif stat_apply_to in ("width_threshold", "threshold"):
        # width_map currently stores w / W, so multiply by W to get pixel width.
        W_img = int(gt.shape[-1])
        if stat_min_width <= 0:
            stat_min_width = int(conf.get("prefill_narrow_width", 6)) + 1
        stat_fill = ((width_map * float(max(W_img, 1))) >= float(stat_min_width)).float()
        stat_fill = stat_fill * hole
    else:
        # Invalid config falls back to old logic: process wide only.
        stat_fill = wide

    if stat_min_width > 0 and stat_apply_to not in ("width_threshold", "threshold"):
        W_img = int(gt.shape[-1])
        stat_fill = stat_fill * (
            (width_map * float(max(W_img, 1))) >= float(stat_min_width)
        ).float()
        stat_fill = stat_fill * hole

    coarse_before_stat = coarse.clone()  # original coarse before stat_match

    if bool(conf.get("prefill_stat_match_enable", False)) and float(stat_fill.sum().item()) > 0:
        coarse, stat_debug_maps = _local_stat_match_prefill(
            gt=gt,
            keep_mask=keep,
            coarse=coarse,
            fill_mask=stat_fill,
            conf=conf,
        )
        stat_debug_maps["prefill_stat_fill_mask"] = stat_fill

    coarse = gt * keep + coarse * hole
    coarse = coarse.clamp(-1.0, 1.0)

    debug_maps = {
        "coarse_x0": coarse,
        "coarse_x0_raw": coarse_before_stat,
        "hole_width_map": width_map,
        "prefill_narrow_mask": narrow,
        "prefill_medium_mask": medium,
        "prefill_wide_mask": wide,
        "prefill_hole_for_coarse": hole_for_coarse,
    }
    if stat_debug_maps:
        debug_maps.update(stat_debug_maps)

    wide_sum = float(wide.sum().detach().cpu().item())
    medium_sum = float(medium.sum().detach().cpu().item())
    narrow_sum = float(narrow.sum().detach().cpu().item())
    stat_sum = 0.0
    try:
        stat_sum = float(stat_fill.sum().detach().cpu().item())
    except Exception:
        stat_sum = 0.0
    print("[coarse_prefill] narrow_sum =", narrow_sum, "medium_sum =", medium_sum, "wide_sum =", wide_sum, "stat_sum =", stat_sum)
    if coarse_wide_row_dbg is not None:
        debug_maps["prefill_wide_row"] = coarse_wide_row_dbg

    if coarse_wide_harmonic_dbg is not None:
        debug_maps["prefill_wide_harmonic"] = coarse_wide_harmonic_dbg
    print("[coarse_debug_keys]", list(debug_maps.keys()))
    return coarse, debug_maps


def _gaussian_like_blur(x, sigma=2.0):
    """Simplified low-pass filter, approximates Gaussian with multiple avg_pool."""
    sigma = float(sigma)
    if sigma <= 0:
        return x
    y = x
    repeats = max(1, int(round(sigma)))
    for _ in range(repeats):
        y = F.avg_pool2d(y, kernel_size=5, stride=1, padding=2)
    return y


def texture_fusion_from_repaint(coarse_x0, repaint_out, keep_mask, conf, stat_fill_mask=None):
    """
    Low frequency primarily uses coarse_x0, high frequency uses repaint_out.

    v3c key modification:
    For medium/wide vertical missing bands marked by prefill_stat_fill_mask,
    allow directly bypassing RePaint/Fusion, or retaining only minimal RePaint.

    Background: self-supervised wide gap experiments show final MAE/RMSE often exceeds coarse,
    indicating current RePaint high frequency mostly adds dirty texture rather than correcting low-frequency structure.
    """
    sigma = float(conf.get("texture_fusion_sigma", 1.2))
    gain = float(conf.get("texture_fusion_gain", 1.0))

    low_repaint_ratio = float(conf.get("texture_fusion_low_repaint_ratio", 0.25))
    low_repaint_ratio = max(0.0, min(1.0, low_repaint_ratio))

    low_coarse = _gaussian_like_blur(coarse_x0, sigma=sigma)
    low_repaint = _gaussian_like_blur(repaint_out, sigma=sigma)
    high_repaint = repaint_out - low_repaint

    low_mix = low_coarse * (1.0 - low_repaint_ratio) + low_repaint * low_repaint_ratio
    refined = low_mix + gain * high_repaint

    keep = keep_mask.float()
    if keep.shape[1] == 1 and refined.shape[1] != 1:
        keep = keep.expand(-1, refined.shape[1], -1, -1)

    hole = 1.0 - keep
    out = coarse_x0 * keep + refined * hole

    # ------------------------------------------------------------
    # v3c: bypass/downweight RePaint for stat_fill_mask region.
    # This is a mechanism change, not further gain tuning.
    #
    # texture_fusion_bypass_stat_mask: true means stat region defaults to coarse_x0.
    # texture_fusion_stat_repaint_alpha: allow small retention of fusion result.
    #   0.00 = stat region uses pure coarse_x0
    #   0.15 = stat region 85% coarse + 15% fusion
    # ------------------------------------------------------------
    if bool(conf.get("texture_fusion_bypass_stat_mask", False)) and stat_fill_mask is not None:
        stat = (stat_fill_mask > 0.5).float()
        stat = stat * (1.0 - keep_mask.float())
        if stat.shape[1] == 1 and out.shape[1] != 1:
            stat = stat.expand(-1, out.shape[1], -1, -1)

        stat_alpha = float(conf.get("texture_fusion_stat_repaint_alpha", 0.0))
        stat_alpha = max(0.0, min(1.0, stat_alpha))
        stat_out = coarse_x0 * (1.0 - stat_alpha) + out * stat_alpha
        out = out * (1.0 - stat) + stat_out * stat

    return out.clamp(-1.0, 1.0)


# ===============================
# Physical Prior helpers
# ===============================
def _mask_to_hole(mask, mode="hole"):
    """
    Input: mask [B,C,H,W], range typically 0/1 or 0~1
    Output: hole_mask (missing region=1, other=0)

    mode:
      - keep: mask=1 is preserved region -> hole = 1-mask
      - hole: mask=1 is missing region -> hole = mask
      - auto: auto detect (missing portion is usually small; if mask's 1 ratio > 0.5 treat as keep)
    """
    m = (mask > 0.5).float()

    if mode == "keep":
        return 1.0 - m
    if mode == "hole":
        return m

    ones_ratio = float(m.mean().detach().cpu().item())
    return (1.0 - m) if ones_ratio > 0.5 else m


# ============================================================
# ✅ Connected component labeling (pure numpy)
# ============================================================
def _cc_label_2d(binary: np.ndarray, connectivity: int = 8):
    """
    Pure numpy BFS connected component labeling.
    binary: HxW, {0,1}
    return:
      labels: HxW int32, 0=background, 1..K=component ID
      K: number of connected components
    """
    assert binary.ndim == 2
    H, W = binary.shape
    labels = np.zeros((H, W), dtype=np.int32)
    K = 0

    if connectivity == 4:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    for y in range(H):
        for x in range(W):
            if binary[y, x] == 0 or labels[y, x] != 0:
                continue
            K += 1
            q = deque([(y, x)])
            labels[y, x] = K
            while q:
                cy, cx = q.popleft()
                for dy, dx in nbrs:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < H and 0 <= nx < W:
                        if binary[ny, nx] != 0 and labels[ny, nx] == 0:
                            labels[ny, nx] = K
                            q.append((ny, nx))
    return labels, K
def build_stripe_hole_mask(
    gt_keep_mask: th.Tensor, mode: str = "keep", stripe_thr: float = 0.95
):
    hole = _mask_to_hole(gt_keep_mask, mode=mode)
    hole = _to_1ch(hole)
    col_ratio = hole.mean(dim=2, keepdim=True)  # [B,1,1,W]
    stripe_cols = (col_ratio > float(stripe_thr)).float()
    stripe_hole = hole * stripe_cols.expand_as(hole)
    return stripe_hole, stripe_cols


# ============================================================
# Prior Diagnostics: only saves intermediate images, does not participate in sampling or loss
# ============================================================
def _prior_debug_safe_name(name):
    """Convert GT_name to a safe filename."""
    try:
        base = os.path.basename(str(name))
        base = os.path.splitext(base)[0]
    except Exception:
        base = str(name)
    base = base.replace("/", "_").replace("\\", "_").replace(":", "_")
    return base


def _prior_debug_to_u8(t: th.Tensor, mode: str = "auto"):
    """
    Convert [B,C,H,W] / [C,H,W] / [H,W] Tensor to uint8 numpy.

    mode:
    - "mask": input treated as [0,1];
    - "image": input treated as [-1,1];
    - "auto": auto detect based on min/max.
    """
    if t is None:
        return None

    x = t.detach().float().cpu()

    if x.dim() == 4:
        # [B,C,H,W]
        pass
    elif x.dim() == 3:
        # [C,H,W] -> [1,C,H,W]
        x = x.unsqueeze(0)
    elif x.dim() == 2:
        # [H,W] -> [1,1,H,W]
        x = x.unsqueeze(0).unsqueeze(0)
    else:
        return None

    # Average across channels when multi-channel, FMI is currently typically 1 channel
    if x.shape[1] > 1:
        x = x.mean(dim=1, keepdim=True)

    if mode == "mask":
        x = x.clamp(0.0, 1.0) * 255.0
    elif mode == "image":
        x = (x.clamp(-1.0, 1.0) + 1.0) * 127.5
    else:
        mn = float(x.min())
        mx = float(x.max())
        if mn >= -0.05 and mx <= 1.05:
            x = x.clamp(0.0, 1.0) * 255.0
        elif mn >= -1.05 and mx <= 1.05:
            x = (x.clamp(-1.0, 1.0) + 1.0) * 127.5
        else:
            x = (x - mn) / max(mx - mn, 1e-6)
            x = x * 255.0

    return x.squeeze(1).round().clamp(0, 255).byte().numpy()


def _prior_debug_save_batch(t, img_names, out_dir, tag, mode="auto"):
    """Save a batch of diagnostic images."""
    arr = _prior_debug_to_u8(t, mode=mode)
    if arr is None:
        return

    os.makedirs(str(out_dir), exist_ok=True)

    # Lazy import to avoid affecting existing dependencies
    from PIL import Image

    bsz = arr.shape[0]
    for i in range(bsz):
        if img_names is not None and i < len(img_names):
            stem = _prior_debug_safe_name(img_names[i])
        else:
            stem = f"sample_{i:04d}"
        path = os.path.join(out_dir, f"{stem}_{tag}.png")
        Image.fromarray(arr[i]).save(path)


def save_prefill_debug_images(debug_maps, out_dir, img_names=None, max_items=None):
    """Save coarse prefill related debug images."""
    if debug_maps is None:
        return

    os.makedirs(out_dir, exist_ok=True)

    print("[save_prefill_debug_images] out_dir =", out_dir)
    print("[save_prefill_debug_images] keys =", list(debug_maps.keys()))

    for tag, tensor in debug_maps.items():
        if tensor is None:
            continue

        if max_items is not None:
            t = tensor[:max_items]
            names = img_names[:max_items] if img_names is not None else None
        else:
            t = tensor
            names = img_names

        if "mask" in tag or "hole" in tag:
            mode = "mask"
        elif "width" in tag:
            mode = "auto"
        else:
            mode = "image"

        _prior_debug_save_batch(
            t,
            img_names=names,
            out_dir=out_dir,
            tag=tag,
            mode=mode,
        )


def _stable_int_seed(name, base_seed=20260525) -> int:
    """
    Generate a stable seed.
    Cannot use Python built-in hash() because different processes may produce different hash results.
    """
    s = f"{base_seed}:{str(name)}".encode("utf-8", errors="ignore")
    h = hashlib.sha256(s).hexdigest()
    return int(h[:8], 16) % (2**31 - 1)


def _make_stable_noise_like(x: th.Tensor, names=None, base_seed=20260525) -> th.Tensor:
    """
    Generate stable noise for each image in the batch.
    Same filename + same base_seed produces consistent results across runs.
    """
    noises = []
    bsz = x.shape[0]

    for i in range(bsz):
        if names is not None and i < len(names):
            name_i = names[i]
        else:
            name_i = f"sample_{i:04d}"

        seed_i = _stable_int_seed(name_i, base_seed=base_seed)

        # Some PyTorch versions have inconsistent CUDA Generator support, try CUDA first, fall back to CPU.
        try:
            gen = th.Generator(device=x.device)
            gen.manual_seed(seed_i)
            noise_i = th.randn(
                x[i : i + 1].shape,
                device=x.device,
                dtype=x.dtype,
                generator=gen,
            )
        except Exception:
            gen = th.Generator(device="cpu")
            gen.manual_seed(seed_i)
            noise_i = th.randn(
                x[i : i + 1].shape,
                device="cpu",
                dtype=x.dtype,
                generator=gen,
            ).to(x.device)

        noises.append(noise_i)

    return th.cat(noises, dim=0)


def _build_stripe_width_map(gt_keep_mask, mode="keep", stripe_thr=0.75):
    """
    Label each hole pixel with its horizontal missing segment width, normalized to [0,1].

    Purpose: diagnose whether wide stripes are incorrectly treated as narrow stripes.
    """
    hole = _to_1ch_robust(_mask_to_hole(gt_keep_mask, mode=mode))
    stripe_hole, _ = build_stripe_hole_mask(
        gt_keep_mask, mode=mode, stripe_thr=stripe_thr
    )

    hole_cpu = stripe_hole.detach().cpu()
    B, _, H, W = hole_cpu.shape
    width_map = th.zeros_like(hole_cpu)

    for b in range(B):
        for y in range(H):
            row = hole_cpu[b, 0, y] > 0.5
            x = 0
            while x < W:
                while x < W and not bool(row[x].item()):
                    x += 1
                if x >= W:
                    break
                s = x
                while x < W and bool(row[x].item()):
                    x += 1
                e = x
                width = float(e - s)
                width_map[b, 0, y, s:e] = width / float(max(W, 1))

    return width_map.to(gt_keep_mask.device) * hole

def main(conf: conf_mgt.Default_Conf):
    print("Start", conf["name"])

    device = th.device(conf.get("device", "cuda:0"))
    print(f"[INFO] Using device: {device}")

    model, diffusion = create_model_and_diffusion(
        **select_args(conf, model_and_diffusion_defaults().keys()), conf=conf
    )

    map_loc = device if th.cuda.is_available() else "cpu"

    model.load_state_dict(
        dist_util.load_state_dict(
            os.path.expanduser(conf.model_path), map_location=map_loc
        )
    )
    model.to(device)
    if conf.use_fp16:
        model.convert_to_fp16()
    model.eval()

    show_progress = conf.show_progress

    # ===============================
    # Read prior configs
    # ===============================

    # ============================================================
    # Prior debug output: only saves diagnostic images, does not affect sampling
    # ============================================================
    prior_debug_enable = bool(conf.get("prior_debug_enable", False))
    prior_debug_dir = str(conf.get("prior_debug_dir", "./log/FMI/prior_debug"))
    prior_debug_max_batches = int(conf.get("prior_debug_max_batches", 2))
    prior_debug_save_row_interp = bool(conf.get("prior_debug_save_row_interp", True))
    prior_debug_save_masks = bool(conf.get("prior_debug_save_masks", True))
    prior_debug_save_boundary = bool(conf.get("prior_debug_save_boundary", True))
    prior_debug_save_masked_input = bool(
        conf.get("prior_debug_save_masked_input", True)
    )

    # === Prevent "black columns": vertical constraints disabled by default for full-height stripe columns ===
    depth_exclude_stripe = bool(conf.get("depth_exclude_stripe", True))
    vertical_smooth_exclude_stripe = bool(
        conf.get("vertical_smooth_exclude_stripe", True)
    )

    # ===============================
    # Classifier (optional)
    # ===============================
    classifier = None
    classifier_guidance_enable = conf.classifier_scale > 0 and conf.classifier_path

    if classifier_guidance_enable:
        print("loading classifier...")
        classifier = create_classifier(
            **select_args(conf, classifier_defaults().keys())
        )
        classifier.load_state_dict(
            dist_util.load_state_dict(
                os.path.expanduser(conf.classifier_path), map_location=map_loc
            )
        )
        classifier.to(device)
        if conf.classifier_use_fp16:
            classifier.convert_to_fp16()
        classifier.eval()


    need_any_guidance = classifier_guidance_enable

    if need_any_guidance:

        def cond_fn(x, t, y=None, gt=None, gt_keep_mask=None, **kwargs):
            total = th.zeros_like(x)

            # 1) classifier guidance
            if classifier_guidance_enable:
                assert y is not None
                with th.enable_grad():
                    x_in = x.detach().requires_grad_(True)
                    logits = classifier(x_in, t)
                    log_probs = F.log_softmax(logits, dim=-1)
                    selected = log_probs[range(len(logits)), y.view(-1)]
                    grad_cls = (
                        th.autograd.grad(selected.sum(), x_in)[0]
                        * conf.classifier_scale
                    )
                total = total + grad_cls


            # ============================================================
            return total

    else:
        cond_fn = None

    def model_fn(x, t, y=None, gt=None, **kwargs):
        assert y is not None
        return model(x, t, y if conf.class_cond else None, gt=gt)

    print("sampling...")

    dset = "eval"
    eval_name = conf.get_default_eval_name()
    suffix = str(conf.get("eval_name_suffix", ""))
    if suffix:
        eval_name = eval_name + suffix

    # dl = conf.get_dataloader(dset=dset, dsName=eval_name)
    rank = int(conf_arg.rank)
    world_size = int(conf_arg.world_size)

    paper_conf = dict(conf["data"]["eval"]["paper_face_mask"])
    paper_conf["return_dataloader"] = True
    paper_conf["rank"] = rank
    paper_conf["world_size"] = world_size

    dl = load_data_inpa(**paper_conf)

    p7_all_rows = []
    p7_all_tex_rows = []
    p7_all_conf_rows = []
    p7_all_safety_rows = []
    p7_tex_safe_output = None
    p7_conf_gate_output = None

    for idx, batch in enumerate(iter(dl)):
        # if world_size > 1 and (idx % world_size) != rank:
        #     continue

        for k in batch.keys():
            if isinstance(batch[k], th.Tensor):
                batch[k] = batch[k].to(device)

        model_kwargs = {"gt": batch["GT"]}

        gt_keep_mask = batch.get("gt_keep_mask")
        if gt_keep_mask is not None:
            model_kwargs["gt_keep_mask"] = gt_keep_mask


        # ============================================================
        # Coarse Prefill + RePaint Texture Refiner
        # ============================================================
        coarse_prefill_enable = bool(conf.get("coarse_prefill_enable", False))
        repaint_init_from_coarse = bool(conf.get("repaint_init_from_coarse", False))
        prefill_debug_enable = bool(conf.get("prefill_debug_enable", False))
        prefill_debug_dir = str(
            conf.get("prefill_debug_dir", "./log/FMI/prefill_debug")
        )
        prefill_debug_max_batches = int(conf.get("prefill_debug_max_batches", 2))

        coarse_x0 = None
        prefill_debug_maps = None

        if coarse_prefill_enable and gt_keep_mask is not None:
            coarse_x0, prefill_debug_maps = build_coarse_prefill(
                model_kwargs["gt"],
                gt_keep_mask,
                conf,
            )
            model_kwargs["coarse_x0"] = coarse_x0

            if prefill_debug_enable and idx < prefill_debug_max_batches:
                img_names = batch.get("GT_name", None)
                debug_out_dir = os.path.join(
                    prefill_debug_dir, eval_name, f"batch_{idx:04d}"
                )
                save_prefill_debug_images(
                    prefill_debug_maps,
                    debug_out_dir,
                    img_names=img_names,
                    max_items=None,
                )

        # ============================================================
        # P1-A: Mask Routing diagnostics (pure visualization, does not modify final)
        # ============================================================
        routing_enable = bool(conf.get("routing_enable", False))
        routing_maps = None
        if routing_enable and gt_keep_mask is not None:
            routing_maps = build_mask_routing_maps(gt_keep_mask, conf)

            # Console statistics: per-image output + ratio
            B_r = gt_keep_mask.shape[0]
            hole_total = routing_maps["routing_hole_mask"]
            mask_keys = [
                "routing_narrow_mask", "routing_medium_mask",
                "routing_long_vertical_mask", "routing_wide_area_mask",
                "routing_stat_match_mask", "routing_bypass_repaint_mask",
                "routing_adaptive_fusion_mask",
            ]
            for bi in range(B_r):
                h_count = int(hole_total[bi].sum().item())
                if h_count == 0:
                    continue
                parts = []
                for key in mask_keys:
                    s = int(routing_maps[key][bi].sum().item())
                    short = key.replace("routing_", "").replace("_mask", "")
                    parts.append(f"{short}={s}({s/h_count*100:.1f}%)")
                img_label = img_names[bi] if img_names is not None and bi < len(img_names) else f"img{bi}"
                print(f"[routing] {img_label}: hole={h_count} | {' '.join(parts)}")

            # Save debug images
            if prefill_debug_enable and idx < prefill_debug_max_batches:
                routing_out_dir = os.path.join(
                    prefill_debug_dir, eval_name, f"batch_{idx:04d}"
                )
                for tag, tensor in routing_maps.items():
                    _prior_debug_save_batch(
                        tensor,
                        img_names=img_names,
                        out_dir=routing_out_dir,
                        tag=tag,
                        mode="mask",
                    )

                # Color classification map
                class_map = build_routing_class_map(routing_maps)
                _prior_debug_save_batch(
                    class_map,
                    img_names=img_names,
                    out_dir=routing_out_dir,
                    tag="routing_class_map",
                    mode="image",
                )

                # Print paths for batch_0000 and batch_0006
                if idx in (0, 6):
                    print(f"[routing] batch_{idx:04d} debug dir: {routing_out_dir}")

        # ============================================================

        batch_size = model_kwargs["gt"].shape[0]

        if conf.cond_y is not None:
            classes = th.ones(batch_size, dtype=th.long, device=device)
            model_kwargs["y"] = classes * conf.cond_y
        else:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(batch_size,), device=device
            )
            model_kwargs["y"] = classes

        sample_fn = (
            diffusion.p_sample_loop if not conf.use_ddim else diffusion.ddim_sample_loop
        )
        # ✅ Single-channel FMI: sampling shape must match the model's training input channels.
        # No longer hardcoded to 3 channels, prioritize current batch's GT channel count.

        # ============================================================
        # Prior diagnostics: save mask, stripe, boundary diagnostic images
        # Only save for the first prior_debug_max_batches batches to avoid excessive output.
        # ============================================================
        if (
            prior_debug_enable
            and gt_keep_mask is not None
            and idx < prior_debug_max_batches
        ):
            debug_out_dir = os.path.join(prior_debug_dir, eval_name, f"batch_{idx:04d}")
            img_names = batch.get("GT_name")

            if prior_debug_save_masks:
                hole_dbg = _to_1ch_robust(_mask_to_hole(gt_keep_mask, mode="keep"))
                stripe_hole_dbg, stripe_cols_dbg = build_stripe_hole_mask(
                    gt_keep_mask,
                    mode="keep",
                    stripe_thr=float(
                        conf.get(
                            "row_interp_stripe_thr", conf.get("stripe_detect_thr", 0.75)
                        )
                    ),
                )
                stripe_col_ratio_dbg = hole_dbg.mean(dim=2, keepdim=True).expand_as(
                    hole_dbg
                )
                stripe_width_dbg = _build_stripe_width_map(
                    gt_keep_mask,
                    mode="keep",
                    stripe_thr=float(
                        conf.get(
                            "row_interp_stripe_thr", conf.get("stripe_detect_thr", 0.75)
                        )
                    ),
                )

                _prior_debug_save_batch(
                    gt_keep_mask,
                    img_names=img_names,
                    out_dir=debug_out_dir,
                    tag="known_keep_mask",
                    mode="mask",
                )
                _prior_debug_save_batch(
                    hole_dbg,
                    img_names=img_names,
                    out_dir=debug_out_dir,
                    tag="hole_mask",
                    mode="mask",
                )
                _prior_debug_save_batch(
                    stripe_hole_dbg,
                    img_names=img_names,
                    out_dir=debug_out_dir,
                    tag="stripe_hole_mask",
                    mode="mask",
                )
                _prior_debug_save_batch(
                    stripe_col_ratio_dbg,
                    img_names=img_names,
                    out_dir=debug_out_dir,
                    tag="stripe_col_ratio",
                    mode="mask",
                )
                _prior_debug_save_batch(
                    stripe_width_dbg,
                    img_names=img_names,
                    out_dir=debug_out_dir,
                    tag="stripe_width_map",
                    mode="auto",
                )
                _prior_debug_save_batch(
                    stripe_width_dbg,
                    img_names=img_names,
                    out_dir=debug_out_dir,
                    tag="stripe_width_map",
                    mode="auto",
                )

            if prior_debug_save_masked_input:
                masked_input_dbg = model_kwargs["gt"] * gt_keep_mask + (
                    -1.0
                ) * th.ones_like(model_kwargs["gt"]) * (1.0 - gt_keep_mask)
                _prior_debug_save_batch(
                    masked_input_dbg,
                    img_names=img_names,
                    out_dir=debug_out_dir,
                    tag="masked_input_preview",
                    mode="image",
                )

        sample_channels = int(model_kwargs["gt"].shape[1])
        sample_shape = (batch_size, sample_channels, conf.image_size, conf.image_size)

        # ============================================================
        # Coarse init: prepare init_image / init_t / init_noise
        # ============================================================
        init_image = None
        init_t = None
        init_noise = None

        if repaint_init_from_coarse and coarse_x0 is not None:
            num_steps = int(diffusion.num_timesteps)
            refine_strength = float(conf.get("repaint_refine_strength", 0.35))
            refine_strength = max(0.01, min(0.95, refine_strength))
            init_t_int = int(round((num_steps - 1) * refine_strength))
            init_t_int = max(1, min(num_steps - 1, init_t_int))

            init_image = coarse_x0
            init_t = th.full(
                (coarse_x0.shape[0],),
                init_t_int,
                device=coarse_x0.device,
                dtype=th.long,
            )

            base_seed = int(conf.get("repaint_seed_base", 20260525))
            seed_mode = str(conf.get("repaint_seed_mode", "name_hash"))
            img_names = batch.get("GT_name", None)
            if seed_mode == "name_hash":
                init_noise = _make_stable_noise_like(
                    coarse_x0,
                    names=img_names,
                    base_seed=base_seed,
                )
            else:
                th.manual_seed(base_seed)
                init_noise = th.randn_like(coarse_x0)

            print(
                f"[coarse_init] batch={idx}, init_t={init_t_int}, refine_strength={refine_strength}"
            )


        result = sample_fn(
            model_fn,
            sample_shape,
            clip_denoised=conf.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=device,
            progress=show_progress,
            return_all=True,
            conf=conf,
            init_image=init_image,
            init_t=init_t,
            init_noise=init_noise,
        )

        # ============================================================
        # Texture Fusion + Final Hard Merge + verification
        # ============================================================
        # ============================================================
        # Texture Fusion: coarse controls low frequency, RePaint controls high frequency
        # ============================================================
        texture_fusion_enable = bool(conf.get("texture_fusion_enable", False))
        final_hard_merge = bool(conf.get("final_hard_merge", True))
        repaint_raw = None

        if texture_fusion_enable and coarse_x0 is not None and gt_keep_mask is not None:
            repaint_raw = result["sample"]

            if prefill_debug_enable and idx < prefill_debug_max_batches:
                img_names = batch.get("GT_name")
                debug_out_dir = os.path.join(
                    prefill_debug_dir,
                    eval_name,
                    f"batch_{idx:04d}",
                )

                _prior_debug_save_batch(
                    repaint_raw,
                    img_names=img_names,
                    out_dir=debug_out_dir,
                    tag="repaint_raw_output",
                    mode="image",
                )

            # v3c: if build_coarse_prefill generated prefill_stat_fill_mask,
            # pass it to texture_fusion_from_repaint to bypass or downweight
            # RePaint high-frequency noise for medium/wide vertical missing bands.
            stat_fill_mask_for_fusion = None
            if isinstance(prefill_debug_maps, dict):
                stat_fill_mask_for_fusion = prefill_debug_maps.get("prefill_stat_fill_mask", None)

            result["sample"] = texture_fusion_from_repaint(
                coarse_x0,
                repaint_raw,
                gt_keep_mask,
                conf,
                stat_fill_mask=stat_fill_mask_for_fusion,
            )

            if prefill_debug_enable and idx < prefill_debug_max_batches:
                img_names = batch.get("GT_name")
                debug_out_dir = os.path.join(
                    prefill_debug_dir,
                    eval_name,
                    f"batch_{idx:04d}",
                )

                _prior_debug_save_batch(
                    result["sample"],
                    img_names=img_names,
                    out_dir=debug_out_dir,
                    tag="texture_fusion_output",
                    mode="image",
                )

        # ============================================================
        # Final hard merge: known region must strictly equal gt before saving
        # ============================================================
        if (
            final_hard_merge
            and gt_keep_mask is not None
            and result.get("gt") is not None
        ):
            keep_for_merge = gt_keep_mask.to(result["sample"].device).float()

            if keep_for_merge.shape[1] == 1 and result["sample"].shape[1] != 1:
                keep_for_merge = keep_for_merge.expand(
                    -1,
                    result["sample"].shape[1],
                    -1,
                    -1,
                )

            result["sample"] = (
                result["sample"] * (1.0 - keep_for_merge)
                + result["gt"] * keep_for_merge
            ).clamp(-1.0, 1.0)

            known_mae = (
                (result["sample"] - result["gt"]).abs() * keep_for_merge
            ).sum() / keep_for_merge.sum().clamp_min(1e-6)

            print(
                f"[final_hard_merge] batch={idx}, "
                f"known_mae={float(known_mae.detach().cpu()):.8f}"
            )

            # P1-C-Freeze: formal freeze declaration that RePaint is not connected to final
            if idx == 0:
                print(
                    "[P1-C-Freeze] RePaint is not used in final. "
                    "Reason: P1-B found no stable positive contribution "
                    "in wide/medium/narrow regions."
                )

            if prefill_debug_enable and idx < prefill_debug_max_batches:
                img_names = batch.get("GT_name")
                debug_out_dir = os.path.join(
                    prefill_debug_dir,
                    eval_name,
                    f"batch_{idx:04d}",
                )

                # Skip final_hard_merge_output writing in P7 debug mode
                # to avoid regenerating final files
                p7_skip_final = bool(conf.get("p7_geoguide_enable", False)) and bool(conf.get("p7_debug", True))
                if not p7_skip_final:
                    _prior_debug_save_batch(
                        result["sample"],
                        img_names=img_names,
                        out_dir=debug_out_dir,
                        tag="final_hard_merge_output",
                        mode="image",
                    )

        # ============================================================
        # P7-1: Layer Base Candidate (D2-2R mechanism, debug only)
        # ============================================================
        p7_geoguide_enable = bool(conf.get("p7_geoguide_enable", False))
        p7_layer_base_enable = bool(conf.get("p7_layer_base_enable", False))
        p7_layer_base_output = None
        if (
            p7_geoguide_enable
            and p7_layer_base_enable
            and prefill_debug_enable
            and idx < prefill_debug_max_batches
            and gt_keep_mask is not None
            and result.get("gt") is not None
        ):
            p7_out_dir = os.path.join(prefill_debug_dir, eval_name, "p7_layer_base")
            p7_rows = run_p7_layer_base_diagnostics(
                baseline_output=result["sample"],
                gt=result["gt"],
                gt_keep_mask=gt_keep_mask,
                img_names=batch.get("GT_name"),
                save_dir=p7_out_dir,
                conf=conf,
            )
            p7_all_rows.extend(p7_rows)

            # Generate layer_base tensor for P7-2
            p7_layer_base_output, p7_layer_base_debug = p7_layer_base_candidate(
                result["sample"], result["gt"], gt_keep_mask, conf
            )

        # ============================================================
        # P7-2: Texture-Preserving Layer Guidance (debug only)
        # ============================================================
        p7_tex_safe_enable = bool(conf.get("p7_layer_texture_safe_enable", False))
        if (
            p7_geoguide_enable
            and p7_tex_safe_enable
            and p7_layer_base_output is not None
            and prefill_debug_enable
            and idx < prefill_debug_max_batches
            and gt_keep_mask is not None
            and result.get("gt") is not None
        ):
            p7_tex_dir = os.path.join(prefill_debug_dir, eval_name, "p7_layer_texture_safe")
            p7_tex_rows = run_p7_layer_texture_safe_diagnostics(
                baseline_output=result["sample"],
                layer_base_output=p7_layer_base_output,
                gt=result["gt"],
                gt_keep_mask=gt_keep_mask,
                img_names=batch.get("GT_name"),
                save_dir=p7_tex_dir,
                conf=conf,
                layer_base_debug=p7_layer_base_debug,
            )
            p7_all_tex_rows.extend(p7_tex_rows)

            # Generate texture_safe candidate tensor for P7-3
            p7_tex_safe_output, _ = p7_layer_texture_safe_candidate(
                result["sample"], p7_layer_base_output, result["gt"], gt_keep_mask, conf
            )

        # ============================================================
        # P7-3: Confidence Gate (debug only)
        # ============================================================
        p7_conf_gate_rows = []
        if (
            p7_geoguide_enable
            and p7_tex_safe_enable
            and p7_tex_safe_output is not None
            and p7_layer_base_debug is not None
            and prefill_debug_enable
            and idx < prefill_debug_max_batches
            and gt_keep_mask is not None
            and result.get("gt") is not None
        ):
            p7_conf_dir = os.path.join(prefill_debug_dir, eval_name, "p7_confidence_gate")
            p7_conf_gate_rows = run_p7_confidence_gate_diagnostics(
                baseline_output=result["sample"],
                texture_safe_output=p7_tex_safe_output,
                gt=result["gt"],
                gt_keep_mask=gt_keep_mask,
                img_names=batch.get("GT_name"),
                save_dir=p7_conf_dir,
                conf=conf,
                layer_base_debug=p7_layer_base_debug,
            )
            p7_all_conf_rows.extend(p7_conf_gate_rows)

            # Generate conf_gate candidate tensor for P7-4
            p7_conf_gate_output, _, _ = p7_confidence_gate_candidate(
                result["sample"], p7_tex_safe_output, result["gt"],
                gt_keep_mask, p7_layer_base_debug, conf
            )

        # ============================================================
        # P7-4: Safety Gate (debug only)
        # ============================================================
        if (
            p7_geoguide_enable
            and p7_conf_gate_output is not None
            and p7_layer_base_debug is not None
            and prefill_debug_enable
            and idx < prefill_debug_max_batches
            and gt_keep_mask is not None
            and result.get("gt") is not None
        ):
            p7_safety_dir = os.path.join(prefill_debug_dir, eval_name, "p7_safety_gate")
            p7_safety_rows, p7_group_stats = run_p7_safety_gate_diagnostics(
                baseline_output=result["sample"],
                conf_gate_output=p7_conf_gate_output,
                gt=result["gt"],
                gt_keep_mask=gt_keep_mask,
                img_names=batch.get("GT_name"),
                save_dir=p7_safety_dir,
                conf=conf,
                layer_base_debug=p7_layer_base_debug,
            )
            p7_all_safety_rows.extend(p7_safety_rows)

        # ============================================================
        # P1-B: Routing diversion diagnostic experiment (does not modify final)
        # ============================================================
        p1b_enable = bool(conf.get("p1b_enable", False))
        if (
            p1b_enable
            and prefill_debug_enable
            and idx < prefill_debug_max_batches
            and gt_keep_mask is not None
            and result.get("gt") is not None
            and routing_enable
            and routing_maps is not None
            and texture_fusion_enable
            and coarse_x0 is not None
        ):
            # Get coarse_x0_raw / coarse_x0_after_stat
            c_raw = prefill_debug_maps.get("coarse_x0_raw", coarse_x0) if isinstance(prefill_debug_maps, dict) else coarse_x0
            c_after_stat = prefill_debug_maps.get("coarse_x0_after_stat", coarse_x0) if isinstance(prefill_debug_maps, dict) else coarse_x0

            p1b_out_dir = os.path.join(prefill_debug_dir, eval_name, f"batch_{idx:04d}")
            run_p1b_diagnostics(
                baseline_current=result["sample"],
                repaint_raw=repaint_raw,
                coarse_x0_raw=c_raw,
                coarse_x0_after_stat=c_after_stat,
                routing_maps=routing_maps,
                gt_img=result["gt"],
                gt_keep_mask=gt_keep_mask,
                img_names=batch.get("GT_name"),
                save_dir=p1b_out_dir,
                conf=conf,
            )

        # ============================================================
        # P2-A: Texture/Boundary diagnostics (does not modify final)
        # ============================================================
        p2a_enable = bool(conf.get("p2a_enable", False))
        if (
            p2a_enable
            and prefill_debug_enable
            and idx < prefill_debug_max_batches
            and gt_keep_mask is not None
            and result.get("gt") is not None
        ):
            p2a_out_dir = os.path.join(prefill_debug_dir, eval_name, f"batch_{idx:04d}")
            run_p2a_diagnostics(
                baseline_current=result["sample"],
                gt_img=result["gt"],
                gt_keep_mask=gt_keep_mask,
                img_names=batch.get("GT_name"),
                save_dir=p2a_out_dir,
                conf=conf,
            )


        # ------------------------------------------------------------
        # v3: self-supervised simulated mask diagnostics with error map / metrics.
        # Only meaningful when gt is a complete reference image; do not use for evaluating final quality on real missing images.
        # ------------------------------------------------------------
        if (
            bool(conf.get("prefill_error_debug_enable", False))
            and prefill_debug_enable
            and idx < prefill_debug_max_batches
            and gt_keep_mask is not None
            and result.get("gt") is not None
        ):
            img_names = batch.get("GT_name")
            debug_out_dir = os.path.join(
                prefill_debug_dir,
                eval_name,
                f"batch_{idx:04d}",
            )
            hole_dbg = (1.0 - gt_keep_mask.to(result["sample"].device).float())
            if hole_dbg.shape[1] == 1 and result["sample"].shape[1] != 1:
                hole_dbg_c = hole_dbg.expand(-1, result["sample"].shape[1], -1, -1)
            else:
                hole_dbg_c = hole_dbg

            def _save_err_and_metrics(tag, pred):
                if pred is None:
                    return
                pred = pred.to(result["gt"].device).float()
                gt_ref = result["gt"].float()
                err = (pred - gt_ref).abs() * hole_dbg_c
                _prior_debug_save_batch(
                    err,
                    img_names=img_names,
                    out_dir=debug_out_dir,
                    tag=f"{tag}_abs_error_hole",
                    mode="auto",
                )

                denom = hole_dbg_c.sum().clamp_min(1e-6)
                mae = err.sum() / denom
                rmse = th.sqrt((((pred - gt_ref) ** 2) * hole_dbg_c).sum() / denom)

                pred_vals = pred[hole_dbg_c > 0.5]
                gt_vals = gt_ref[hole_dbg_c > 0.5]
                if pred_vals.numel() > 1 and gt_vals.numel() > 1:
                    std_ratio = pred_vals.std() / gt_vals.std().clamp_min(1e-6)
                    mean_diff = pred_vals.mean() - gt_vals.mean()
                    print(
                        f"[hole_error] batch={idx}, {tag}: "
                        f"mae={float(mae.detach().cpu()):.6f}, "
                        f"rmse={float(rmse.detach().cpu()):.6f}, "
                        f"std_ratio={float(std_ratio.detach().cpu()):.4f}, "
                        f"mean_diff={float(mean_diff.detach().cpu()):.6f}"
                    )

            _save_err_and_metrics("coarse_x0", coarse_x0)
            _save_err_and_metrics("final", result["sample"])

        srs = toU8(result["sample"])
        gts = toU8(result["gt"])

        if model_kwargs.get("gt_keep_mask") is not None:
            lrs = toU8(
                result.get("gt") * model_kwargs.get("gt_keep_mask")
                + (-1)
                * th.ones_like(result.get("gt"))
                * (1 - model_kwargs.get("gt_keep_mask"))
            )
            gt_keep_masks = toU8((model_kwargs.get("gt_keep_mask") * 2 - 1))
        else:
            lrs = None
            gt_keep_masks = None

        conf.eval_imswrite(
            srs=srs,
            gts=gts,
            lrs=lrs,
            gt_keep_masks=gt_keep_masks,
            img_names=batch["GT_name"],
            dset=dset,
            name=eval_name,
            verify_same=False,
        )


    # ============================================================
    # P7-1: Write aggregated layer_base CSV
    # ============================================================
    if p7_all_rows:
        import csv as _csv_p7
        p7_csv_dir = os.path.join(prefill_debug_dir, eval_name, "p7_layer_base")
        os.makedirs(p7_csv_dir, exist_ok=True)
        p7_csv_path = os.path.join(p7_csv_dir, "p7_metrics_layer_base.csv")
        fieldnames = list(p7_all_rows[0].keys())
        with open(p7_csv_path, "w", newline="", encoding="utf-8") as f:
            w = _csv_p7.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in p7_all_rows:
                w.writerow(row)
        print(f"[P7-1] Aggregated CSV: {p7_csv_path} ({len(p7_all_rows)} rows)")

    # ============================================================
    # P7-2: Write aggregated texture-safe CSV
    # ============================================================
    if p7_all_tex_rows:
        import csv as _csv_p7t
        p7_tex_csv_dir = os.path.join(prefill_debug_dir, eval_name, "p7_layer_texture_safe")
        os.makedirs(p7_tex_csv_dir, exist_ok=True)
        p7_tex_csv_path = os.path.join(p7_tex_csv_dir, "p7_metrics_layer_texture_safe.csv")
        fieldnames = list(p7_all_tex_rows[0].keys())
        with open(p7_tex_csv_path, "w", newline="", encoding="utf-8") as f:
            w = _csv_p7t.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in p7_all_tex_rows:
                w.writerow(row)
        print(f"[P7-2] Aggregated CSV: {p7_tex_csv_path} ({len(p7_all_tex_rows)} rows)")

    # ============================================================
    # P7-3: Write aggregated confidence gate CSV
    # ============================================================
    if p7_all_conf_rows:
        import csv as _csv_p7c
        p7_conf_csv_dir = os.path.join(prefill_debug_dir, eval_name, "p7_confidence_gate")
        os.makedirs(p7_conf_csv_dir, exist_ok=True)
        p7_conf_csv_path = os.path.join(p7_conf_csv_dir, "p7_metrics_confidence_gate.csv")
        fieldnames = list(p7_all_conf_rows[0].keys())
        with open(p7_conf_csv_path, "w", newline="", encoding="utf-8") as f:
            w = _csv_p7c.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in p7_all_conf_rows:
                w.writerow(row)
        print(f"[P7-3] Aggregated CSV: {p7_conf_csv_path} ({len(p7_all_conf_rows)} rows)")

    # ============================================================
    # P7-4: Write aggregated safety gate CSV
    # ============================================================
    if p7_all_safety_rows:
        import csv as _csv_p7s
        p7_safety_csv_dir = os.path.join(prefill_debug_dir, eval_name, "p7_safety_gate")
        os.makedirs(p7_safety_csv_dir, exist_ok=True)
        p7_safety_csv_path = os.path.join(p7_safety_csv_dir, "p7_metrics_safety_gate.csv")
        fieldnames = list(p7_all_safety_rows[0].keys())
        with open(p7_safety_csv_path, "w", newline="", encoding="utf-8") as f:
            w = _csv_p7s.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in p7_all_safety_rows:
                w.writerow(row)
        print(f"[P7-4] Aggregated CSV: {p7_safety_csv_path} ({len(p7_all_safety_rows)} rows)")

    # ============================================================
    # P7-5: Full Validation Summary
    # ============================================================
    if p7_all_conf_rows and p7_all_safety_rows:
        import csv as _csv_p7v
        p7v_dir = os.path.join(prefill_debug_dir, eval_name, "p7_full_validation")
        os.makedirs(p7v_dir, exist_ok=True)

        # Write full validation CSV (from safety gate data)
        p7v_csv_path = os.path.join(p7v_dir, "p7_metrics_full_validation.csv")
        fieldnames = list(p7_all_safety_rows[0].keys())
        with open(p7v_csv_path, "w", newline="", encoding="utf-8") as f:
            w = _csv_p7v.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in p7_all_safety_rows:
                w.writerow(row)
        print(f"[P7-5] Full validation CSV: {p7v_csv_path} ({len(p7_all_safety_rows)} rows)")

        # Global statistics
        n = len(p7_all_safety_rows)
        rd_deltas = [float(r["delta_row_delta"]) for r in p7_all_safety_rows]
        tex_deltas = [float(r["delta_tex"]) for r in p7_all_safety_rows]
        seam_deltas = [float(r["delta_seam"]) for r in p7_all_safety_rows]
        update_pxs = [int(r["update_pixel_count"]) for r in p7_all_safety_rows]
        update_ratios = [float(r["update_ratio"]) for r in p7_all_safety_rows]

        rd_pos = sum(1 for d in rd_deltas if d < -1e-6)
        rd_neg = sum(1 for d in rd_deltas if d > 1e-6)
        rd_zero = n - rd_pos - rd_neg

        # Per-group statistics
        groups = {}
        for r in p7_all_safety_rows:
            key = f"{r['trigger_type']}_{r['context_side']}"
            if key not in groups:
                groups[key] = {"count": 0, "rd": [], "tex": [], "seam": [], "upx": []}
            groups[key]["count"] += 1
            groups[key]["rd"].append(float(r["delta_row_delta"]))
            groups[key]["tex"].append(float(r["delta_tex"]))
            groups[key]["seam"].append(float(r["delta_seam"]))
            groups[key]["upx"].append(int(r["update_pixel_count"]))

        # Write summary markdown
        md_lines = []
        md_lines.append("# P7-5 Full Validation Summary\n")
        md_lines.append(f"> Date: 2026-06-02")
        md_lines.append(f"> Sample count: {n}")
        md_lines.append(f"> Coverage: full evaluation set ({n} images)\n")

        md_lines.append("## 1. Global Metrics\n")
        md_lines.append("| Metric | Mean | Median | Min | Max |")
        md_lines.append("|------|------|--------|------|------|")
        md_lines.append(f"| delta_row_delta | {np.mean(rd_deltas):.6f} | {np.median(rd_deltas):.6f} | {min(rd_deltas):.6f} | {max(rd_deltas):.6f} |")
        md_lines.append(f"| delta_tex | {np.mean(tex_deltas):.6f} | {np.median(tex_deltas):.6f} | {min(tex_deltas):.6f} | {max(tex_deltas):.6f} |")
        md_lines.append(f"| delta_seam | {np.mean(seam_deltas):.6f} | {np.median(seam_deltas):.6f} | {min(seam_deltas):.6f} | {max(seam_deltas):.6f} |")
        md_lines.append(f"| update_pixel_count | {np.mean(update_pxs):.0f} | {np.median(update_pxs):.0f} | {min(update_pxs)} | {max(update_pxs)} |")
        md_lines.append(f"| update_ratio | {np.mean(update_ratios):.4f} | {np.median(update_ratios):.4f} | {min(update_ratios):.4f} | {max(update_ratios):.4f} |")

        md_lines.append(f"\n## 2. row_delta Improvement Distribution\n")
        md_lines.append(f"| Direction | Count | Percentage |")
        md_lines.append(f"|------|------|------|")
        md_lines.append(f"| Improved (negative) | {rd_pos} | {rd_pos/n*100:.1f}% |")
        md_lines.append(f"| Unchanged | {rd_zero} | {rd_zero/n*100:.1f}% |")
        md_lines.append(f"| Degraded (positive) | {rd_neg} | {rd_neg/n*100:.1f}% |")

        md_lines.append(f"\n## 3. Group Statistics\n")
        md_lines.append("| Group | count | avg Δrd | avg Δtex | avg Δseam | avg update_px |")
        md_lines.append("|----|-------|---------|----------|-----------|---------------|")
        for key in sorted(groups.keys()):
            g = groups[key]
            md_lines.append(f"| {key} | {g['count']} | {np.mean(g['rd']):.6f} | {np.mean(g['tex']):.6f} | {np.mean(g['seam']):.6f} | {np.mean(g['upx']):.0f} |")

        md_lines.append(f"\n## 4. Risk Checks\n")
        all_known_ok = all(r["known_mae_ok"] == "1" for r in p7_all_safety_rows)
        all_nan_ok = all(r["nan_ok"] == "1" for r in p7_all_safety_rows)
        all_tex_ok = all(r["tex_ok"] == "1" for r in p7_all_safety_rows)
        all_seam_ok = all(r["seam_ok"] == "1" for r in p7_all_safety_rows)
        none_update_ok = all(int(r["update_pixel_count"]) == 0 for r in p7_all_safety_rows if r["trigger_type"] == "none")

        md_lines.append(f"| Check Item | Result |")
        md_lines.append(f"|--------|------|")
        md_lines.append(f"| known_mae all zero | {'✅' if all_known_ok else '❌'} |")
        md_lines.append(f"| No NaN | {'✅' if all_nan_ok else '❌'} |")
        md_lines.append(f"| tex_ratio no systematic decline | {'✅' if all_tex_ok else '❌'} |")
        md_lines.append(f"| seam_grad no systematic degradation | {'✅' if all_seam_ok else '❌'} |")
        md_lines.append(f"| none trigger update count is 0 | {'✅' if none_update_ok else '❌'} |")
        md_lines.append(f"| row_delta improvement not from single sample | {'✅' if rd_pos > 2 else '❌'} ({rd_pos}/{n} samples) |")

        # Conclusion
        md_lines.append(f"\n## 5. Conclusion\n")
        if all_known_ok and all_nan_ok and all_tex_ok and all_seam_ok and rd_pos > 2:
            conclusion = "pass_to_p7_6_proposal"
            md_lines.append("**pass_to_p7_6_proposal** — All safety checks passed, row_delta improvement not from single sample.")
        elif all_known_ok and all_nan_ok:
            conclusion = "hold_needs_adjustment"
            md_lines.append("**hold_needs_adjustment** — Safety checks passed but metrics need adjustment.")
        else:
            conclusion = "freeze_not_safe"
            md_lines.append("**freeze_not_safe** — Safety checks failed.")

        md_lines.append(f"\nConclusion: `{conclusion}`")

        md_path = os.path.join(p7v_dir, "p7_full_validation_summary.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))
        print(f"[P7-5] Summary: {md_path}")
        print(f"[P7-5] Conclusion: {conclusion}")

    print("sampling complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf_path", type=str, required=False, default=None)

    # Multi-GPU sharding parameters
    parser.add_argument(
        "--device", type=str, default=None, help="e.g. cuda:0 / cuda:1 / cpu"
    )
    parser.add_argument("--rank", type=int, default=0, help="Current process rank, starting from 0")
    parser.add_argument("--world_size", type=int, default=1, help="Total number of processes (GPUs)")
    parser.add_argument(
        "--out_suffix", type=str, default="", help="Output directory suffix, e.g. r0of2 / r1of2"
    )

    args = vars(parser.parse_args())

    conf_arg = conf_mgt.conf_base.Default_Conf()
    conf_arg.update(yamlread(args.get("conf_path")))

    if args.get("device") is not None:
        conf_arg["device"] = args["device"]

    conf_arg["rank"] = int(args.get("rank", 0))
    conf_arg["world_size"] = int(args.get("world_size", 1))

    suffix = args.get("out_suffix", "")
    if suffix:
        eval_sets = conf_arg.get("data", {}).get("eval", {})
        for _, ds_conf in eval_sets.items():
            paths = ds_conf.get("paths", {})
            for k, v in paths.items():
                paths[k] = f"{v}_{suffix}"

    main(conf_arg)
