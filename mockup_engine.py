"""
Mockup Engine
=============
Reads a compact "mockup definition" (a definition.json plus a handful of
small extracted PNGs) and drops a seamless pattern into it - perspective
warp, real-world scale, garment-shape masking, and shadow/highlight
blending, all automatically.

A mockup definition is produced ONCE per Creatsy/PSD mockup file (an
offline, one-time extraction step - see EXTRACTING_NEW_MOCKUPS.md), so the
big 500MB+ PSD never has to be touched again. From then on this module only
ever reads small PNGs.

Folder layout expected for each mockup:
    mockups/<mockup_id>/
        definition.json
        base_background.png
        garment_base.png
        mask_*.png
        <overlay/shadow layer files referenced in definition.json>

Usage:
    from mockup_engine import render_mockup
    result = render_mockup("mockups/blouse_front_3", pattern_image, working_width=3000)
    result.save("output.png")
"""
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def _find_coeffs(dst_pts, src_pts):
    """Solves for the 8 perspective-transform coefficients PIL needs to map
    dst_pts (where we want the corners to end up) back to src_pts (the
    corners of our source image) - i.e. the inverse mapping PIL's
    Image.transform(..., PERSPECTIVE, ...) expects."""
    matrix = []
    for (x, y), (X, Y) in zip(dst_pts, src_pts):
        matrix.append([x, y, 1, 0, 0, 0, -X * x, -X * y])
        matrix.append([0, 0, 0, x, y, 1, -Y * x, -Y * y])
    A = np.array(matrix, dtype=float)
    B = np.array(src_pts, dtype=float).reshape(8)
    return np.linalg.solve(A, B)


def _dist(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _blend(base_arr, top_img, mode):
    """Applies one Photoshop-style blend mode. base_arr is float32 RGB in
    0-1, top_img is the overlay layer (with its own alpha as opacity)."""
    top = np.array(top_img, dtype=np.float32) / 255.0
    a = top[..., 3:4]
    b = base_arr[..., :3]
    t = top[..., :3]
    if mode == "multiply":
        out = b * t
    elif mode == "screen":
        out = 1 - (1 - b) * (1 - t)
    elif mode == "linear_burn":
        out = np.clip(b + t - 1, 0, 1)
    elif mode == "color_burn":
        out = 1 - np.clip((1 - b) / np.clip(t, 1e-4, 1), 0, 1)
    else:
        raise ValueError(f"Unknown blend mode: {mode}")
    result = base_arr.copy()
    result[..., :3] = b * (1 - a) + out * a
    return result


def render_mockup(mockup_dir, pattern_tile: Image.Image, working_width: int = 3000) -> Image.Image:
    """Composites `pattern_tile` (one seamless repeat unit, RGBA, any size -
    exported straight from Minimal Pattern Studio) into the mockup described
    by mockup_dir/definition.json, at correct real-world scale, and returns
    the finished RGB image at `working_width` pixels wide.
    """
    mockup_dir = Path(mockup_dir)
    definition = json.loads((mockup_dir / "definition.json").read_text())

    canvas_w, canvas_h = definition["canvas_size"]
    dpi = definition["dpi"]
    scale = working_width / canvas_w
    W, H = working_width, int(canvas_h * scale)

    # Her pattern tile is assumed to be exported at the same DPI convention
    # as Minimal Pattern Studio (300 DPI), so its own pixel size directly
    # tells us its real-world size - no separate "cm" input needed.
    tile_px = pattern_tile.width
    cm_per_inch = 2.54

    design_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))

    for region in definition["regions"]:
        quad_full = region["quad"]
        top_edge = _dist(quad_full[0], quad_full[1])
        left_edge = _dist(quad_full[0], quad_full[3])
        local_w = max(1, int(round(top_edge)))
        local_h = max(1, int(round(left_edge)))

        # Tile the pattern at its own true pixel size (1:1, no stretching)
        # to cover the region's real-world size before warping - this is
        # what keeps motif size consistent and realistic across regions.
        tiled_source = Image.new("RGBA", (local_w, local_h))
        for i in range(0, local_w, tile_px):
            for j in range(0, local_h, tile_px):
                tiled_source.paste(pattern_tile, (i, j))

        quad = [(x * scale, y * scale) for x, y in quad_full]
        xs = [p[0] for p in quad]
        ys = [p[1] for p in quad]
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        out_w, out_h = max(1, int(maxx - minx)), max(1, int(maxy - miny))
        local_quad = [(x - minx, y - miny) for x, y in quad]
        src_quad = [(0, 0), (local_w, 0), (local_w, local_h), (0, local_h)]
        coeffs = _find_coeffs(local_quad, src_quad)
        warped = tiled_source.transform((out_w, out_h), Image.PERSPECTIVE, coeffs, resample=Image.BICUBIC)

        placed = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        placed.alpha_composite(warped, (int(minx), int(miny)))

        mask_img = Image.open(mockup_dir / region["mask"]).convert("L")
        mleft, mtop = region["mask_offset"]
        mask_full = Image.new("L", (canvas_w, canvas_h), 0)
        mask_full.paste(mask_img, (mleft, mtop))
        mask_full = mask_full.resize((W, H), Image.LANCZOS)

        r, g, b, a = placed.split()
        a = Image.fromarray(
            (np.array(a, dtype=np.float32) * (np.array(mask_full, dtype=np.float32) / 255.0)).astype("uint8")
        )
        placed = Image.merge("RGBA", (r, g, b, a))
        design_layer.alpha_composite(placed)

    def load_base(name):
        return Image.open(mockup_dir / name).convert("RGBA").resize((W, H), Image.LANCZOS)

    comp = None
    for base_name in definition["base_layers"]:
        layer = load_base(base_name)
        comp = layer if comp is None else Image.alpha_composite(comp, layer)
    comp = Image.alpha_composite(comp, design_layer)
    arr = np.array(comp, dtype=np.float32) / 255.0

    for overlay in definition["overlay_layers"]:
        top_img = load_base(overlay["file"])
        arr = _blend(arr, top_img, overlay["mode"])

    final = Image.fromarray((arr * 255).astype("uint8"), "RGBA").convert("RGB")
    return final
