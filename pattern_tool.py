"""
SeamlessChaos — Pattern Tool
--------------------------------------------------------
Workflow:
1) Template auswählen (Vorschau zeigt Slot-Anordnung)
2) Motive hochladen (einzeln oder als Sheet, wird automatisch getrennt)
3) App platziert automatisch nach Template-Vorgabe (Position, Größe, Rotation fest)
4) Optional: Zuordnung tauschen (welches Motiv in welchem Slot)
5) Download als scharfes, transparentes PNG

Run locally:
    pip install streamlit pillow numpy scipy
    streamlit run pattern_tool.py
"""

import streamlit as st
from PIL import Image, ImageDraw, ImageCms
import numpy as np
from scipy import ndimage
import io

st.set_page_config(page_title="SeamlessChaos", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #FAFAFA; color: #2D2D2D; }
.stButton>button, .stDownloadButton>button {
    background-color: #A2B5A2; color: white; border: none;
    border-radius: 8px; padding: 0.5em 1.2em;
}
.template-card {
    border: 1px solid #E4E1D8; border-radius: 12px; padding: 10px;
    background: white;
}
</style>
""", unsafe_allow_html=True)

CANVAS_SIZE = 2000
DPI = 300

# ---------------------------------------------------------
# Template-Bibliothek (alle kalibrierten Templates)
# ---------------------------------------------------------
TEMPLATES = {
    "Crab": {
        "main": [
            {"id": "A", "x": 340,  "y": 389,  "width": 480, "rotation": -48},
            {"id": "B", "x": 1026, "y": 998,  "width": 460, "rotation": -6},
            {"id": "C", "x": 1299, "y": 29,   "width": 420, "rotation": 150},
            {"id": "D", "x": 33,   "y": 1447, "width": 460, "rotation": 32},
        ],
        "secondary": [],
    },
    "Vehicles": {
        "main": [
            {"id": "1", "x": 1253, "y": 195,  "width": 375, "rotation": 0},
            {"id": "2", "x": 510,  "y": 497,  "width": 470, "rotation": 0},
            {"id": "3", "x": 1710, "y": 714,  "width": 417, "rotation": 0},
            {"id": "4", "x": 995,  "y": 1001, "width": 625, "rotation": 0},
            {"id": "5", "x": 237,  "y": 1235, "width": 444, "rotation": 0},
            {"id": "6", "x": 1488, "y": 1421, "width": 300, "rotation": 0},
            {"id": "7", "x": 740,  "y": 1696, "width": 492, "rotation": 0},
            {"id": "8", "x": 39,   "y": 1977, "width": 420, "rotation": 0},
        ],
        "secondary": [],
    },
    "Donkey & Pear": {
        "main": [
            {"id": "A", "x": 1097, "y": 432,  "width": 345, "rotation": 0},
            {"id": "B", "x": 343,  "y": 1025, "width": 345, "rotation": 0},
            {"id": "C", "x": 1358, "y": 1419, "width": 345, "rotation": 0},
            {"id": "D", "x": 64,   "y": 1981, "width": 345, "rotation": 0},
        ],
        "secondary": [
            {"id": "leaf1", "x": 511,  "y": 312,  "width": 195, "rotation": 179.7},
            {"id": "leaf2", "x": 1774, "y": 595,  "width": 195, "rotation": 57.5},
            {"id": "leaf3", "x": 1029, "y": 923,  "width": 195, "rotation": -174.7},
            {"id": "leaf4", "x": 1924, "y": 1376, "width": 195, "rotation": 106.7},
            {"id": "leaf5", "x": 681,  "y": 1579, "width": 195, "rotation": 122.5},
            {"id": "leaf6", "x": 1321, "y": 1909, "width": 195, "rotation": 60.8},
        ],
    },
    "Woodland": {
        "main": [
            {"id": "deer",     "x": 855,  "y": 166,  "width": 320, "rotation": 0},
            {"id": "bird",     "x": 1917, "y": 85,   "width": 320, "rotation": 0},
            {"id": "pinecone", "x": 0,    "y": 811,  "width": 320, "rotation": 0},
            {"id": "squirrel", "x": 1052, "y": 924,  "width": 320, "rotation": 0},
            {"id": "mushroom", "x": 376,  "y": 1439, "width": 320, "rotation": 0},
            {"id": "owl",      "x": 1358, "y": 1568, "width": 320, "rotation": 0},
        ],
        "secondary": [],
    },
    "Baseball": {
        "main": [
            {"id": "pitcher",  "x": 244,  "y": 334,  "width": 345, "rotation": 0},
            {"id": "bat_ball", "x": 975,  "y": 855,  "width": 345, "rotation": 0},
            {"id": "batter",   "x": 1716, "y": 1076, "width": 345, "rotation": 0},
            {"id": "jersey",   "x": 399,  "y": 1511, "width": 345, "rotation": 0},
            {"id": "glove",    "x": 1266, "y": 1879, "width": 345, "rotation": 0},
        ],
        "secondary": [],
    },
}


# ---------------------------------------------------------
# Bildverarbeitung
# ---------------------------------------------------------
def normalize_to_srgb(image: Image.Image) -> Image.Image:
    icc = image.info.get("icc_profile")
    if not icc:
        return image
    try:
        input_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        srgb_profile = ImageCms.createProfile("sRGB")
        mode = "RGBA" if image.mode == "RGBA" else "RGB"
        return ImageCms.profileToProfile(image, input_profile, srgb_profile, outputMode=mode)
    except Exception:
        return image


def detect_motifs(image: Image.Image, dilation_iterations: int = 10, min_size: int = 300):
    arr = np.array(image.convert("RGBA"))
    alpha = arr[:, :, 3]
    rgb = arr[:, :, :3]
    is_white = (rgb[:, :, 0] > 245) & (rgb[:, :, 1] > 245) & (rgb[:, :, 2] > 245)
    content = (alpha > 10) & (~is_white)

    dilated = ndimage.binary_dilation(content, iterations=dilation_iterations)
    labeled, n = ndimage.label(dilated, structure=np.ones((3, 3)))
    sizes = ndimage.sum(content, labeled, range(1, n + 1))
    objs = ndimage.find_objects(labeled)

    crops = []
    for i in range(n):
        if sizes[i] < min_size:
            continue
        sl = objs[i]
        padding = 10
        y0 = max(sl[0].start - padding, 0)
        y1 = min(sl[0].stop + padding, image.height)
        x0 = max(sl[1].start - padding, 0)
        x1 = min(sl[1].stop + padding, image.width)
        crops.append(image.crop((x0, y0, x1, y1)))
    # groesste zuerst (fuer sinnvolle Haupt/Neben-Zuordnung)
    crops.sort(key=lambda im: im.width * im.height, reverse=True)
    return crops


def crop_to_content(img: Image.Image, padding: int = 4) -> Image.Image:
    arr = np.array(img)
    alpha = arr[:, :, 3]
    ys, xs = np.where(alpha > 10)
    if len(xs) == 0:
        return img
    x0, x1 = max(xs.min() - padding, 0), min(xs.max() + padding, img.width)
    y0, y1 = max(ys.min() - padding, 0), min(ys.max() + padding, img.height)
    return img.crop((x0, y0, x1, y1))


def place_motif_on_canvas(canvas, motif, slot, canvas_size=CANVAS_SIZE):
    target_dim = slot["width"]
    longer_side = max(motif.width, motif.height)
    scale = target_dim / longer_side
    target_w = int(motif.width * scale)
    target_h = int(motif.height * scale)
    resized = motif.resize((target_w, target_h), Image.Resampling.LANCZOS)
    rotated = resized.rotate(slot["rotation"], expand=True, resample=Image.Resampling.BICUBIC)
    rw, rh = rotated.size
    cx, cy = slot["x"], slot["y"]
    paste_x = cx - rw // 2
    paste_y = cy - rh // 2
    canvas.paste(rotated, (paste_x, paste_y), rotated)
    offsets = []
    if paste_x < 0: offsets.append((canvas_size, 0))
    if paste_x + rw > canvas_size: offsets.append((-canvas_size, 0))
    if paste_y < 0: offsets.append((0, canvas_size))
    if paste_y + rh > canvas_size: offsets.append((0, -canvas_size))
    if paste_x < 0 and paste_y < 0: offsets.append((canvas_size, canvas_size))
    if paste_x + rw > canvas_size and paste_y < 0: offsets.append((-canvas_size, canvas_size))
    if paste_x < 0 and paste_y + rh > canvas_size: offsets.append((canvas_size, -canvas_size))
    if paste_x + rw > canvas_size and paste_y + rh > canvas_size: offsets.append((-canvas_size, -canvas_size))
    for dx, dy in offsets:
        canvas.paste(rotated, (paste_x + dx, paste_y + dy), rotated)
    return canvas


def render_template_schematic(template, size=260):
    """Einfache Vorschau-Grafik: zeigt Slot-Positionen/Groessen als Kreise."""
    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    scale = size / CANVAS_SIZE
    for slot in template["main"] + template["secondary"]:
        x = slot["x"] * scale
        y = slot["y"] * scale
        r = (slot["width"] * scale) / 2
        draw.ellipse([x - r, y - r, x + r, y + r], outline=(162, 181, 162), width=2)
    return img


# ---------------------------------------------------------
# UI
# ---------------------------------------------------------
st.title("🌿 SeamlessChaos — Pattern Tool")
st.caption("Template wählen, Motive hochladen, fertiges Seamless Pattern in Sekunden")

# --- Schritt 1: Template auswählen ---
st.subheader("1. Template wählen")
template_names = list(TEMPLATES.keys())
cols = st.columns(len(template_names))
if "selected_template" not in st.session_state:
    st.session_state.selected_template = template_names[0]

for i, name in enumerate(template_names):
    with cols[i]:
        tpl = TEMPLATES[name]
        n_main = len(tpl["main"])
        n_sec = len(tpl["secondary"])
        st.image(render_template_schematic(tpl), use_container_width=True)
        label = f"{n_main} Motive" + (f" + {n_sec} klein" if n_sec else "")
        st.markdown(f"**{name}**  \n{label}")
        if st.button("Wählen", key=f"select_{name}"):
            st.session_state.selected_template = name

selected = st.session_state.selected_template
st.success(f"Gewähltes Template: **{selected}**")
template = TEMPLATES[selected]
n_main = len(template["main"])
n_sec = len(template["secondary"])
n_total = n_main + n_sec

canvas_choice = st.radio(
    "Leinwandgröße",
    options=["2000×2000 px (Motive ~3cm)", "3000×3000 px (Motive ~5cm)"],
    horizontal=True,
)
scale_factor = 1.0 if "2000" in canvas_choice else 1.5
canvas_size = int(CANVAS_SIZE * scale_factor)

st.divider()

# --- Schritt 2: Motive hochladen ---
st.subheader("2. Motive hochladen")
st.markdown(f"Dieses Template braucht **{n_total} Motiv(e)** ({n_main} Haupt" +
            (f" + {n_sec} Neben" if n_sec else "") +
            "). Du kannst auch nur 1 Motiv hochladen — es wird dann auf alle Plätze wiederholt.")

uploaded_files = st.file_uploader(
    "Motiv(e) hochladen (PNG, transparent, auch mehrere gleichzeitig oder ein Sheet)",
    type=["png"], accept_multiple_files=True,
)

all_motifs = []
if uploaded_files:
    for f in uploaded_files:
        img = Image.open(f).convert("RGBA")
        img = normalize_to_srgb(img)
        found = detect_motifs(img)
        all_motifs.extend(found if found else [img])

if all_motifs:
    st.success(f"{len(all_motifs)} Motiv(e) erkannt.")
    thumb_cols = st.columns(min(6, len(all_motifs)))
    for i, m in enumerate(all_motifs):
        with thumb_cols[i % len(thumb_cols)]:
            st.image(crop_to_content(m), caption=f"Motiv {i+1}", use_container_width=True)

    st.divider()
    st.subheader("3. Zuordnung (automatisch, tauschbar)")

    # Automatische Erstverteilung
    slot_list = template["main"] + template["secondary"]
    assignment = {}
    if len(all_motifs) == 1:
        for slot in slot_list:
            assignment[slot["id"]] = 0
    else:
        for i, slot in enumerate(slot_list):
            assignment[slot["id"]] = i % len(all_motifs)

    # Manuelle Zuordnung ermöglichen (Tausch-Ersatz ohne Drag&Drop)
    cols2 = st.columns(3)
    for i, slot in enumerate(slot_list):
        with cols2[i % 3]:
            options = list(range(len(all_motifs)))
            assignment[slot["id"]] = st.selectbox(
                f"Slot {slot['id']}", options,
                index=assignment[slot["id"]],
                format_func=lambda x: f"Motiv {x+1}",
                key=f"assign_{slot['id']}",
            )

    st.divider()
    st.subheader("4. Ergebnis")

    canvas = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    for slot in slot_list:
        scaled_slot = {
            "x": int(slot["x"] * scale_factor),
            "y": int(slot["y"] * scale_factor),
            "width": int(slot["width"] * scale_factor),
            "rotation": slot["rotation"],
        }
        motif = crop_to_content(all_motifs[assignment[slot["id"]]])
        canvas = place_motif_on_canvas(canvas, motif, scaled_slot, canvas_size)

    preview_bg = Image.new("RGBA", canvas.size, (250, 250, 250, 255))
    preview_bg.paste(canvas, (0, 0), canvas)
    st.image(preview_bg, caption="Vorschau (mit hellem Hintergrund zur Ansicht)", use_container_width=True)

    buf = io.BytesIO()
    canvas.save(buf, format="PNG", dpi=(DPI, DPI))
    st.download_button(
        "📥 Pattern als transparentes PNG herunterladen",
        data=buf.getvalue(),
        file_name=f"seamlesschaos_{selected.lower().replace(' ', '_')}.png",
        mime="image/png",
    )
else:
    st.info("Noch keine Motive hochgeladen.")
