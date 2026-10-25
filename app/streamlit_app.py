from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import models, transforms


st.set_page_config(
    page_title="NeuroSight | Brain MRI",
    page_icon="🧠",
    layout="wide",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "efficientnet_b0_best.pth"

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
DISPLAY_NAMES = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No tumor",
    "pituitary": "Pituitary tumor",
}
CHART_COLORS = ["#38BDF8", "#A78BFA", "#34D399", "#FBBF24"]

IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;700&display=swap');

    :root {
        --bg-0: #050b16;
        --bg-1: #0a1324;
        --card: rgba(17, 27, 48, 0.62);
        --card-border: rgba(148, 163, 184, 0.16);
        --text: #e6eefb;
        --muted: #94a6c0;
        --cyan: #38bdf8;
        --violet: #a78bfa;
        --green: #34d399;
        --amber: #fbbf24;
    }

    html, body, [class*="css"], .stApp {
        font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
    }

    /* ---------- App background ---------- */
    .stApp {
        background:
            radial-gradient(900px 500px at 8% -5%, rgba(56, 189, 248, 0.18), transparent 60%),
            radial-gradient(700px 500px at 100% 10%, rgba(167, 139, 250, 0.16), transparent 60%),
            radial-gradient(800px 600px at 50% 110%, rgba(52, 211, 153, 0.10), transparent 60%),
            linear-gradient(160deg, var(--bg-0) 0%, var(--bg-1) 60%, #0d1527 100%);
        background-attachment: fixed;
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    footer, #MainMenu {
        visibility: hidden;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, h4 {
        color: #f8fafc !important;
        font-family: 'Space Grotesk', 'Inter', sans-serif !important;
        letter-spacing: -0.01em;
    }

    p, li, label, span {
        color: inherit;
    }

    hr {
        border: 0 !important;
        height: 1px !important;
        background: linear-gradient(90deg, transparent, rgba(148, 163, 184, 0.35), transparent) !important;
        margin: 2rem 0 !important;
    }

    /* ---------- Hero ---------- */
    .hero {
        position: relative;
        padding: 34px 36px 30px 36px;
        border-radius: 24px;
        background:
            linear-gradient(135deg, rgba(56, 189, 248, 0.10), rgba(167, 139, 250, 0.10) 55%, rgba(52, 211, 153, 0.08)),
            rgba(12, 20, 38, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.18);
        box-shadow: 0 20px 60px -20px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.05);
        overflow: hidden;
        margin-bottom: 1.1rem;
    }

    .hero::after {
        content: "";
        position: absolute;
        right: -80px;
        top: -80px;
        width: 280px;
        height: 280px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(56, 189, 248, 0.28), transparent 65%);
        pointer-events: none;
    }

    .eyebrow {
        display: inline-block;
        color: #7dd3fc;
        text-transform: uppercase;
        letter-spacing: 0.18em;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 6px 12px;
        border-radius: 999px;
        background: rgba(56, 189, 248, 0.10);
        border: 1px solid rgba(56, 189, 248, 0.28);
    }

    .hero-title {
        font-family: 'Space Grotesk', 'Inter', sans-serif;
        font-size: clamp(2.4rem, 5vw, 3.6rem);
        font-weight: 700;
        line-height: 1.05;
        margin: 16px 0 8px 0;
        background: linear-gradient(90deg, #f8fafc 0%, #7dd3fc 45%, #a78bfa 100%);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .subtitle {
        color: var(--muted);
        font-size: 1.08rem;
        margin: 0;
    }

    /* ---------- Warning ---------- */
    [data-testid="stAlert"] {
        background: linear-gradient(90deg, rgba(251, 191, 36, 0.12), rgba(251, 191, 36, 0.05));
        border: 1px solid rgba(251, 191, 36, 0.32);
        border-left: 4px solid var(--amber);
        border-radius: 14px;
        color: #fde9b0;
    }

    [data-testid="stAlert"] * {
        color: #fde9b0 !important;
    }

    /* ---------- Section headers ---------- */
    .section-title {
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 0.4rem 0 0.9rem 0;
        font-family: 'Space Grotesk', 'Inter', sans-serif;
        font-size: 1.45rem;
        font-weight: 700;
        color: #f8fafc;
    }

    .section-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 32px;
        height: 32px;
        border-radius: 10px;
        font-size: 0.95rem;
        font-weight: 800;
        color: #06111f;
        background: linear-gradient(135deg, var(--cyan), var(--green));
        box-shadow: 0 6px 18px -4px rgba(56, 189, 248, 0.55);
    }

    .viz-title {
        font-family: 'Space Grotesk', 'Inter', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        color: #f1f5f9;
        margin: 0 0 0.6rem 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .viz-title::before {
        content: "";
        width: 4px;
        height: 18px;
        border-radius: 4px;
        background: linear-gradient(180deg, var(--cyan), var(--violet));
    }

    /* ---------- File uploader ---------- */
    [data-testid="stFileUploader"] {
        background: var(--card);
        border: 1px solid var(--card-border);
        border-radius: 20px;
        padding: 14px;
        backdrop-filter: blur(10px);
        box-shadow: 0 14px 40px -18px rgba(0, 0, 0, 0.6);
    }

    [data-testid="stFileUploaderDropzone"] {
        background: rgba(8, 15, 30, 0.55);
        border: 1.5px dashed rgba(56, 189, 248, 0.45);
        border-radius: 16px;
        transition: border-color 0.2s ease, background 0.2s ease;
    }

    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: var(--cyan);
        background: rgba(56, 189, 248, 0.07);
    }

    [data-testid="stFileUploaderDropzone"] button {
        border-radius: 10px;
        border: 1px solid rgba(56, 189, 248, 0.4);
        background: rgba(56, 189, 248, 0.12);
        color: #bae6fd;
        font-weight: 600;
    }

    /* ---------- Buttons ---------- */
    div.stButton > button {
        width: 100%;
        min-height: 52px;
        border: 0;
        border-radius: 14px;
        color: #04101e;
        font-weight: 800;
        font-size: 1rem;
        letter-spacing: 0.01em;
        background: linear-gradient(90deg, #38bdf8 0%, #34d399 100%);
        box-shadow: 0 12px 30px -10px rgba(56, 189, 248, 0.6);
        transition: transform 0.18s ease, filter 0.18s ease, box-shadow 0.18s ease;
    }

    div.stButton > button:hover {
        transform: translateY(-2px);
        filter: brightness(1.08);
        box-shadow: 0 16px 36px -10px rgba(52, 211, 153, 0.65);
        color: #04101e;
    }

    div.stButton > button:active {
        transform: translateY(0);
    }

    div.stButton > button p {
        color: #04101e !important;
        font-weight: 800;
    }

    /* ---------- Metrics ---------- */
    [data-testid="stMetric"] {
        position: relative;
        background: var(--card);
        border: 1px solid var(--card-border);
        padding: 20px 20px 18px 20px;
        border-radius: 18px;
        backdrop-filter: blur(10px);
        box-shadow: 0 14px 40px -18px rgba(0, 0, 0, 0.6);
        overflow: hidden;
    }

    [data-testid="stMetric"]::before {
        content: "";
        position: absolute;
        left: 0;
        top: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--cyan), var(--violet), var(--green));
    }

    [data-testid="stMetricLabel"] p {
        color: var(--muted) !important;
        font-size: 0.78rem !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }

    [data-testid="stMetricValue"] {
        color: #f8fafc;
        font-family: 'Space Grotesk', 'Inter', sans-serif;
        font-weight: 700;
    }

    /* ---------- Images ---------- */
    [data-testid="stImage"] img {
        border-radius: 18px;
        border: 1px solid rgba(148, 163, 184, 0.2);
        box-shadow: 0 18px 50px -20px rgba(0, 0, 0, 0.8);
    }

    [data-testid="stImageCaption"],
    [data-testid="stCaptionContainer"] {
        color: var(--muted) !important;
    }

    /* ---------- Info panel ---------- */
    .panel {
        position: relative;
        background:
            linear-gradient(160deg, rgba(56, 189, 248, 0.08), rgba(167, 139, 250, 0.06)),
            var(--card);
        border: 1px solid var(--card-border);
        border-radius: 20px;
        padding: 24px;
        min-height: 150px;
        backdrop-filter: blur(10px);
        box-shadow: 0 14px 40px -18px rgba(0, 0, 0, 0.6);
    }

    .panel h3 {
        margin: 12px 0 10px 0;
        font-size: 1.3rem;
    }

    .panel .eyebrow {
        font-size: 0.68rem;
        padding: 5px 10px;
    }

    .small-note {
        color: var(--muted);
        font-size: 0.92rem;
        line-height: 1.6;
    }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {
        background:
            linear-gradient(180deg, rgba(12, 20, 38, 0.97), rgba(7, 13, 26, 0.99));
        border-right: 1px solid rgba(148, 163, 184, 0.14);
    }

    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        font-size: 1.25rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(148, 163, 184, 0.16);
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        padding: 8px 12px;
        margin-bottom: 6px;
        border-radius: 10px;
        background: rgba(56, 189, 248, 0.06);
        border: 1px solid rgba(148, 163, 184, 0.10);
        font-size: 0.9rem;
    }

    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
        background: none;
        border: 0;
        padding: 0;
    }

    /* ---------- Spinner ---------- */
    [data-testid="stSpinner"] {
        color: #7dd3fc;
    }

    /* ---------- Footer note ---------- */
    .footer-note {
        text-align: center;
        color: var(--muted);
        font-size: 0.85rem;
        padding: 14px 18px;
        border-radius: 14px;
        background: rgba(15, 23, 42, 0.5);
        border: 1px solid rgba(148, 163, 184, 0.12);
    }

    /* ---------- Scrollbar ---------- */
    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-thumb {
        background: rgba(148, 163, 184, 0.28);
        border-radius: 10px;
    }
    ::-webkit-scrollbar-track { background: transparent; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    """Load the saved EfficientNet model once and reuse it."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.efficientnet_b0(weights=None)
    input_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(input_features, len(CLASS_NAMES))

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, device


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Apply the same resize and normalization used during training."""
    preprocessing = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
        ]
    )
    return preprocessing(image).unsqueeze(0)


def analyze_image(image: Image.Image):
    """Return model scores and a Grad-CAM overlay for one image."""
    model, device = load_model()
    input_tensor = preprocess_image(image).to(device)

    with torch.inference_mode():
        logits = model(input_tensor)
        scores = torch.softmax(logits, dim=1)[0].cpu().numpy()

    predicted_index = int(np.argmax(scores))

    # Grad-CAM needs gradients, so generate it outside inference mode.
    target_layers = [model.features[-1]]
    targets = [ClassifierOutputTarget(predicted_index)]

    with torch.enable_grad():
        cam = GradCAM(model=model, target_layers=target_layers)
        heatmap = cam(
            input_tensor=input_tensor,
            targets=targets,
        )[0]

    display_image = image.resize((224, 224))
    image_array = np.asarray(display_image, dtype=np.float32) / 255.0
    overlay = show_cam_on_image(image_array, heatmap, use_rgb=True)

    return predicted_index, scores, overlay


def section_title(number: int, text: str) -> None:
    st.markdown(
        f'<div class="section-title"><span class="section-badge">{number}</span>'
        f"<span>{text}</span></div>",
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <div class="hero">
        <span class="eyebrow">Educational AI project</span>
        <div class="hero-title">NeuroSight</div>
        <p class="subtitle">Brain MRI image classification with visual explanation</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.warning(
    "Educational demonstration only. This model is not a medical device and "
    "must not be used for diagnosis or treatment decisions."
)

with st.sidebar:
    st.header("About this demo")
    st.write("**Model:** EfficientNetB0")
    st.write("**Classes:** 4 brain MRI categories")
    st.write("**Input size:** 224 × 224 pixels")
    st.write("**Test accuracy:** 84.8% on this project’s test split")
    st.caption(
        "Dataset-specific result only. It does not indicate clinical accuracy. "
        "Possible near-duplicate candidates in the dataset have not been "
        "manually confirmed."
    )
    st.divider()
    st.caption(
        "Grad-CAM highlights image regions associated with a prediction. "
        "It does not prove that the prediction is correct."
    )

st.write("")
upload_column, info_column = st.columns([1.25, 0.75], gap="large")

with upload_column:
    section_title(1, "Upload an MRI image")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"],
        help="Supported formats: JPG, PNG, BMP, TIFF, and WEBP.",
    )

    if uploaded_file is not None:
        try:
            uploaded_image = Image.open(uploaded_file).convert("RGB")
            st.image(
                uploaded_image,
                caption="Image ready for analysis",
                width="stretch",
            )

            file_identifier = f"{uploaded_file.name}-{uploaded_file.size}"

            if st.session_state.get("file_identifier") != file_identifier:
                st.session_state["file_identifier"] = file_identifier
                st.session_state.pop("analysis_result", None)

            if st.button("Analyze MRI image", type="primary"):
                with st.spinner("Analyzing image and creating explanation..."):
                    prediction, scores, overlay = analyze_image(uploaded_image)

                st.session_state["analysis_result"] = {
                    "prediction": prediction,
                    "scores": scores,
                    "overlay": overlay,
                }

        except Exception as error:
            st.error(f"Could not open this image: {error}")

with info_column:
    st.markdown(
        """
        <div class="panel">
            <div class="eyebrow">How it works</div>
            <h3>From image to model output</h3>
            <p class="small-note">
                Upload a brain MRI image. The classifier returns scores for
                four learned categories. Grad-CAM creates a visual overlay
                showing image regions associated with the selected prediction.
            </p>
            <p class="small-note">
                Model scores are not calibrated medical probabilities.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

result = st.session_state.get("analysis_result")

if result is not None:
    predicted_index = result["prediction"]
    scores = result["scores"]
    overlay = result["overlay"]
    predicted_class = CLASS_NAMES[predicted_index]

    st.divider()
    section_title(2, "Analysis results")

    metric_columns = st.columns(3)
    metric_columns[0].metric(
        "Predicted category",
        DISPLAY_NAMES[predicted_class],
    )
    metric_columns[1].metric(
        "Top model score",
        f"{scores[predicted_index]:.1%}",
    )
    metric_columns[2].metric(
        "Model input",
        "224 × 224",
    )

    st.caption(
        "The top model score is a relative model output, not a measure of "
        "medical certainty."
    )

    st.write("")
    visualization_column, chart_column = st.columns([1.1, 0.9], gap="large")

    with visualization_column:
        st.markdown('<div class="viz-title">Grad-CAM visualization</div>', unsafe_allow_html=True)
        st.image(
            overlay,
            caption=(
                "Highlighted regions influenced the model output; "
                "they do not confirm medical correctness."
            ),
            width="stretch",
        )

    with chart_column:
        st.markdown('<div class="viz-title">Scores across categories</div>', unsafe_allow_html=True)

        labels = [DISPLAY_NAMES[name] for name in CLASS_NAMES]

        figure, axis = plt.subplots(figsize=(6, 4), facecolor="none")
        axis.set_facecolor("none")

        wedges, _ = axis.pie(
            scores,
            colors=CHART_COLORS,
            startangle=90,
            counterclock=False,
            wedgeprops={
                "width": 0.32,
                "edgecolor": "#0a1324",
                "linewidth": 4,
            },
        )

        # Subtle inner ring for depth.
        axis.pie(
            [1],
            colors=["#ffffff"],
            radius=0.64,
            wedgeprops={"width": 0.012, "alpha": 0.10},
        )

        axis.text(
            0,
            0.05,
            f"{scores[predicted_index]:.0%}",
            ha="center",
            va="center",
            fontsize=26,
            fontweight="bold",
            color="#f8fafc",
        )
        axis.text(
            0,
            -0.2,
            "TOP SCORE",
            ha="center",
            va="center",
            fontsize=8,
            color="#94a6c0",
        )

        legend = axis.legend(
            wedges,
            [
                f"{label} — {score:.1%}"
                for label, score in zip(labels, scores)
            ],
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            frameon=False,
            labelcolor="#e6eefb",
            fontsize=10,
            handlelength=1.1,
            handleheight=1.1,
            labelspacing=1.1,
        )

        for text in legend.get_texts():
            text.set_color("#e6eefb")

        axis.set_aspect("equal")
        figure.tight_layout()
        st.pyplot(figure, width="stretch")
        plt.close(figure)

        st.caption(
            "The chart shows how the model distributes its scores among "
            "the four categories; these are not medical probabilities."
        )

st.divider()
st.markdown(
    '<div class="footer-note">NeuroSight is an educational deep-learning demonstration. '
    "It does not replace a qualified healthcare professional.</div>",
    unsafe_allow_html=True,
)