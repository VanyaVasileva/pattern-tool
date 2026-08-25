from pathlib import Path

TARGETS = [Path("pattern_tool.py"), Path("pattern_tool_customer.py")]

REPLACEMENTS = [
    ('    bg.paste(canvas, (0, 0), canvas)\n', '    bg.alpha_composite(canvas)\n', 'template preview compositing'),
    ('    base.paste(img, (0, 0), img)\n', '    base.alpha_composite(img.convert("RGBA"))\n', 'background flattening'),
    ('            repeated.paste(tile, (i * tile_target_px, j * tile_target_px), tile)\n', '            repeated.alpha_composite(tile, (i * tile_target_px, j * tile_target_px))\n', '4000 repeat compositing'),
    ('            export_canvas.paste(canvas, (0, 0), canvas)\n', '            export_canvas.alpha_composite(canvas)\n', 'tile background compositing'),
    ('        preview_bg.paste(canvas, (0, 0), canvas)\n', '        preview_bg.alpha_composite(canvas)\n', 'builder preview compositing'),
    ('            export_canvas.save(\n                tiff_buf,\n                format="TIFF",\n', '            export_canvas.convert("RGB").save(\n                tiff_buf,\n                format="TIFF",\n', 'opaque RGB TIFF export'),
]

OLD_PNG_HELPER = '''def pil_image_to_png_bytes(img: Image.Image, dpi: int = 300) -> bytes:\n    buf = io.BytesIO()\n    img.save(buf, format="PNG", dpi=(dpi, dpi), icc_profile=SRGB_ICC_PROFILE)\n    return buf.getvalue()\n'''

NEW_PNG_HELPER = '''def pil_image_to_png_bytes(img: Image.Image, dpi: int = 300) -> bytes:\n    buf = io.BytesIO()\n    save_img = img\n    # Opaque listing/repeat previews do not need an alpha channel. Saving them\n    # as true RGB avoids Apple Quick Look/iCloud preview artifacts, while\n    # genuinely transparent PNGs keep their alpha exactly as before.\n    if img.mode == "RGBA" and img.getchannel("A").getextrema() == (255, 255):\n        save_img = img.convert("RGB")\n    save_img.save(buf, format="PNG", dpi=(dpi, dpi), icc_profile=SRGB_ICC_PROFILE)\n    return buf.getvalue()\n'''


def patch(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    original = text

    for old, new, label in REPLACEMENTS:
        count = text.count(old)
        if count < 1:
            raise RuntimeError(f"{path}: expected at least 1 occurrence for {label}, found {count}")
        text = text.replace(old, new)
        print(f"{path}: replaced {count} occurrence(s) for {label}")

    helper_count = text.count(OLD_PNG_HELPER)
    if helper_count < 1:
        raise RuntimeError(f"{path}: expected at least 1 PNG helper, found {helper_count}")
    text = text.replace(OLD_PNG_HELPER, NEW_PNG_HELPER)
    print(f"{path}: replaced {helper_count} PNG helper(s)")

    forbidden = [
        'bg.paste(canvas, (0, 0), canvas)',
        'base.paste(img, (0, 0), img)',
        'repeated.paste(tile, (i * tile_target_px, j * tile_target_px), tile)',
        'export_canvas.paste(canvas, (0, 0), canvas)',
        'preview_bg.paste(canvas, (0, 0), canvas)',
    ]
    for snippet in forbidden:
        if snippet in text:
            raise RuntimeError(f"{path}: forbidden alpha-double-mask call remains: {snippet}")

    if text == original:
        raise RuntimeError(f"{path}: no changes made")
    path.write_text(text, encoding="utf-8")
    print(f"Patched {path}")


for target in TARGETS:
    patch(target)
