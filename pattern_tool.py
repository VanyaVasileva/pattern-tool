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
            {"id": "A", "x": 340,  "y": 389,  "width": 455, "rotation": -48},
            {"id": "B", "x": 1026, "y": 998,  "width": 455, "rotation": -6},
            {"id": "C", "x": 1299, "y": 29,   "width": 455, "rotation": 150},
            {"id": "D", "x": 33,   "y": 1447, "width": 455, "rotation": 32},
        ],
        "secondary": [],
    },
    "Vehicles": {
        "main": [
            {"id": "1", "x": 1253, "y": 195,  "width": 443, "rotation": 0},
            {"id": "2", "x": 510,  "y": 497,  "width": 443, "rotation": 0},
            {"id": "3", "x": 1710, "y": 714,  "width": 443, "rotation": 0},
            {"id": "4", "x": 995,  "y": 1001, "width": 443, "rotation": 0},
            {"id": "5", "x": 237,  "y": 1235, "width": 443, "rotation": 0},
            {"id": "6", "x": 1488, "y": 1421, "width": 443, "rotation": 0},
            {"id": "7", "x": 740,  "y": 1696, "width": 443, "rotation": 0},
            {"id": "8", "x": 39,   "y": 1977, "width": 443, "rotation": 0},
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


def render_template_schematic(template, sample_motif, size=260, canvas_size=CANVAS_SIZE):
    """Echte Vorschau: platziert ein Beispiel-Motiv auf alle Slots, damit man
    das Template-Layout tatsächlich sieht, nicht nur leere Kreise."""
    canvas = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    for slot in template["main"] + template["secondary"]:
        canvas = place_motif_on_canvas(canvas, sample_motif, slot, canvas_size)
    bg = Image.new("RGBA", canvas.size, (250, 250, 250, 255))
    bg.paste(canvas, (0, 0), canvas)
    return bg.resize((size, size), Image.Resampling.LANCZOS)


def make_sample_motif(size=300):
    """Erzeugt ein einfaches, generisches Platzhalter-Motiv (Blatt-Form) fuer die Vorschau."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse([size*0.2, size*0.1, size*0.8, size*0.9], fill=(162, 181, 162, 255))
    draw.ellipse([size*0.35, size*0.3, size*0.65, size*0.6], fill=(250, 250, 250, 255))
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
        sample = make_sample_motif()
        st.image(render_template_schematic(tpl, sample), use_container_width=True)
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

no_split = st.checkbox(
    "Bilder NICHT automatisch aufteilen (jede Datei = 1 Motiv, "
    "auch wenn es aus mehreren nicht verbundenen Teilen besteht)"
)

def load_motifs(files, no_split):
    motifs = []
    if not files:
        return motifs
    for f in files:
        img = Image.open(f).convert("RGBA")
        img = normalize_to_srgb(img)
        if no_split:
            motifs.append(img)
        else:
            found = detect_motifs(img)
            motifs.extend(found if found else [img])
    return motifs

main_motifs = []
secondary_motifs = []

if n_sec > 0:
    st.markdown(f"Dieses Template hat **{n_main} große Haupt-Plätze** und **{n_sec} kleine Neben-Plätze**. "
                "Lad für jede Kategorie separat hoch, damit die Zuordnung stimmt.")
    col_main, col_sec = st.columns(2)
    with col_main:
        st.markdown(f"**Hauptmotive** ({n_main} benötigt)")
        main_files = st.file_uploader("Hauptmotive hochladen", type=["png"],
                                       accept_multiple_files=True, key="main_upload",
                                       label_visibility="collapsed")
        main_motifs = load_motifs(main_files, no_split)
    with col_sec:
        st.markdown(f"**Nebenmotive** ({n_sec} benötigt)")
        sec_files = st.file_uploader("Nebenmotive hochladen", type=["png"],
                                      accept_multiple_files=True, key="sec_upload",
                                      label_visibility="collapsed")
        secondary_motifs = load_motifs(sec_files, no_split)
    all_motifs = main_motifs + secondary_motifs
else:
    st.markdown(f"Dieses Template braucht **{n_total} Motiv(e)**. "
                "Du kannst auch nur 1 Motiv hochladen — es wird dann auf alle Plätze wiederholt.")
    uploaded_files = st.file_uploader(
        "Motiv(e) hochladen (PNG, transparent, auch mehrere gleichzeitig oder ein Sheet)",
        type=["png"], accept_multiple_files=True,
    )
    all_motifs = load_motifs(uploaded_files, no_split)
    main_motifs = all_motifs

if all_motifs:
    st.success(f"{len(all_motifs)} Motiv(e) erkannt.")
    thumb_cols = st.columns(min(6, len(all_motifs)))
    for i, m in enumerate(all_motifs):
        with thumb_cols[i % len(thumb_cols)]:
            label = f"Motiv {i+1}" if not n_sec else (
                f"Haupt {i+1}" if i < len(main_motifs) else f"Neben {i - len(main_motifs) + 1}"
            )
            st.image(crop_to_content(m), caption=label, use_container_width=True)

    st.divider()
    st.subheader("3. Zuordnung (automatisch, tauschbar)")

    slot_list = template["main"] + template["secondary"]
    main_pool = main_motifs if main_motifs else all_motifs
    sec_pool = secondary_motifs if secondary_motifs else main_pool

    # ("main"/"secondary", index) pro Slot
    assignment = {}
    for i, slot in enumerate(template["main"]):
        assignment[slot["id"]] = ("main", i % len(main_pool))
    for i, slot in enumerate(template["secondary"]):
        assignment[slot["id"]] = ("secondary", i % len(sec_pool))

    def resolve_motif(assign_tuple):
        pool_name, idx = assign_tuple
        pool = main_pool if pool_name == "main" else sec_pool
        return pool[idx % len(pool)]

    # Manuelle Zuordnung ermoeglichen (Tausch-Ersatz ohne Drag&Drop)
    cols2 = st.columns(3)
    for i, slot in enumerate(slot_list):
        with cols2[i % 3]:
            is_secondary = slot in template["secondary"]
            pool = sec_pool if is_secondary else main_pool
            pool_name = "secondary" if is_secondary else "main"
            options = list(range(len(pool)))
            current_idx = assignment[slot["id"]][1]
            chosen = st.selectbox(
                f"Slot {slot['id']}", options,
                index=min(current_idx, len(options) - 1),
                format_func=lambda x, pn=pool_name: f"{'Neben' if pn=='secondary' else 'Haupt'} {x+1}",
                key=f"assign_{slot['id']}",
            )
            assignment[slot["id"]] = (pool_name, chosen)

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
        motif = crop_to_content(resolve_motif(assignment[slot["id"]]))
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
