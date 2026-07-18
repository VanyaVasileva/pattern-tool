import io
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from mockup_engine import render_mockup

st.set_page_config(page_title="Mockup Studio", page_icon="🧺", layout="centered")
st.title("Mockup Studio")
st.caption("Drop a finished pattern into a mockup template - perspective, scale, and shadows handled automatically.")

MOCKUPS_DIR = Path(__file__).parent / "mockups"
_bg_color_picker = components.declare_component("bg_color_picker", path=str(Path(__file__).parent))


def list_mockups():
    if not MOCKUPS_DIR.exists():
        return {}
    found = {}
    for folder in sorted(MOCKUPS_DIR.iterdir()):
        definition = folder / "definition.json"
        if definition.exists():
            import json
            name = json.loads(definition.read_text()).get("name", folder.name)
            found[name] = folder
    return found


mockups = list_mockups()

if not mockups:
    st.warning("No mockup templates found yet. Add a folder under `mockups/` with a `definition.json`.")
    st.stop()

selected_name = st.selectbox("Mockup template", list(mockups.keys()))
selected_dir = mockups[selected_name]

uploaded = st.file_uploader(
    "Upload your finished pattern (one seamless tile, transparent PNG - straight from Minimal Pattern Studio)",
    type=["png"],
)

st.markdown("**Background color**")
bg_transparent = st.checkbox(
    "No background color (design stays transparent over the garment's own fabric color)",
    value=False, key="mockup_bg_transparent",
)
if not bg_transparent:
    if "mockup_bg_color" not in st.session_state:
        st.session_state.mockup_bg_color = "#D8C4B6"
    picked_hex = _bg_color_picker(default=st.session_state.mockup_bg_color, key="mockup_bg_picker")
    if picked_hex:
        st.session_state.mockup_bg_color = picked_hex

if uploaded:
    pattern = Image.open(uploaded).convert("RGBA")
    st.caption(f"Pattern tile: {pattern.width}×{pattern.height}px")

    if not bg_transparent:
        hex_clean = st.session_state.mockup_bg_color.strip().lstrip("#")
        try:
            bg_rgb = tuple(int(hex_clean[i:i+2], 16) for i in (0, 2, 4))
        except Exception:
            bg_rgb = (216, 196, 182)
        flattened = Image.new("RGBA", pattern.size, (*bg_rgb, 255))
        flattened.alpha_composite(pattern)
        pattern = flattened

    with st.spinner("Placing your pattern into the mockup..."):
        result = render_mockup(selected_dir, pattern, working_width=3000)

    st.image(result, caption=selected_name, use_container_width=True)

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    st.download_button(
        "📥 Download mockup PNG",
        data=buf.getvalue(),
        file_name=f"mockup_{selected_dir.name}.png",
        mime="image/png",
    )
else:
    st.info("Upload a pattern to see it on the mockup.")

