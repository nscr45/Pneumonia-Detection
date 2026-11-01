import io
import html
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
import pydicom
from PIL import Image, ImageOps

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass


# -------------------------------------------------------------------
# Page configuration and styling
# -------------------------------------------------------------------
st.set_page_config(
    page_title="PneumoAI | Chest X-ray Analysis",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    :root {
        --bg: #0b1220;
        --panel: #111c2e;
        --panel2: #152238;
        --line: rgba(148, 163, 184, .18);
        --muted: #9aabc2;
        --cyan: #42d6c5;
        --blue: #69a8ff;
    }
    .stApp {
        background:
          radial-gradient(circle at 12% 0%, rgba(35, 117, 145, .17), transparent 28rem),
          radial-gradient(circle at 90% 8%, rgba(67, 98, 180, .13), transparent 26rem),
          var(--bg);
        color: #edf4ff;
        font-family: 'DM Sans', sans-serif;
    }
    h1, h2, h3, h4 {
        font-family: 'Manrope', sans-serif !important;
        color: #f4f8ff !important;
    }
    [data-testid="stSidebar"] {
        background: #0d1727;
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] * { color: #dce8f7; }
    .hero {
        padding: 1.45rem 1.7rem;
        border: 1px solid var(--line);
        border-radius: 22px;
        background: linear-gradient(115deg, rgba(22, 43, 65, .96), rgba(15, 30, 49, .88));
        margin-bottom: 1.2rem;
    }
    .eyebrow {
        color: var(--cyan);
        font-size: .76rem;
        font-weight: 700;
        letter-spacing: .16em;
        text-transform: uppercase;
    }
    .hero-title {
        font-family: 'Manrope', sans-serif;
        font-size: clamp(1.8rem, 3vw, 2.7rem);
        font-weight: 800;
        line-height: 1.13;
        margin: .45rem 0;
        color: #f7fbff;
    }
    .subtle { color: var(--muted); }
    .panel {
        background: linear-gradient(145deg, rgba(21, 34, 56, .96), rgba(15, 27, 45, .96));
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 1.15rem 1.25rem;
        margin-bottom: 1rem;
    }
    .metric-label {
        color: var(--muted);
        font-size: .78rem;
        text-transform: uppercase;
        letter-spacing: .09em;
        font-weight: 700;
    }
    .metric-value {
        color: #f5f9ff;
        font-family: 'Manrope', sans-serif;
        font-size: 1.7rem;
        font-weight: 800;
        margin-top: .2rem;
    }
    .stButton > button {
        border: 0;
        border-radius: 12px;
        min-height: 3rem;
        color: #061a20;
        font-weight: 800;
        background: linear-gradient(90deg, #42d6c5, #76e2cf);
        width: 100%;
    }
    .stButton > button:hover {
        color: #061a20;
        border: 0;
        filter: brightness(1.06);
    }
    [data-testid="stFileUploader"] {
        background: rgba(18, 34, 54, .72);
        border: 1px dashed rgba(66, 214, 197, .5);
        border-radius: 16px;
        padding: .8rem;
    }
    [data-testid="stAlert"] { border-radius: 12px; }
    footer { visibility: hidden; }

    /* Responsive layout: prevent horizontal overflow and clipped panels */
    html, body, [data-testid="stAppViewContainer"], .stApp {
        width: 100%;
        max-width: 100%;
        overflow-x: hidden !important;
    }
    [data-testid="stMainBlockContainer"], .block-container {
        box-sizing: border-box !important;
        width: 100% !important;
        max-width: 1200px !important;
        padding-left: clamp(1rem, 3vw, 2rem) !important;
        padding-right: clamp(1rem, 3vw, 2rem) !important;
    }
    .panel, .hero, [data-testid="stImage"] {
        max-width: 100%;
        min-width: 0;
        box-sizing: border-box;
    }
    [data-testid="stImage"] img {
        max-width: 100%;
        height: auto;
        object-fit: contain;
    }
    .stMarkdown, .stCaption, p {
        overflow-wrap: anywhere;
    }
    @media (max-width: 768px) {
        .hero { padding: 1rem; }
        .metric-value { font-size: 1.35rem; }
        .panel { padding: 1rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------
# Model loading
# -------------------------------------------------------------------
MODEL_CANDIDATES = [
    Path("best_cnn_model.h5"),
    Path("models/best_cnn_model.h5"),
    Path("models/pneumonia_cnn_phase1.h5"),
    Path("model.h5"),
]


@st.cache_resource(show_spinner="Loading the trained pneumonia model...")
def load_prediction_model():
    for model_path in MODEL_CANDIDATES:
        if model_path.is_file():
            model = tf.keras.models.load_model(str(model_path), compile=False)
            return model, str(model_path)

    searched = ", ".join(str(path) for path in MODEL_CANDIDATES)
    raise FileNotFoundError(f"Model file not found. Checked: {searched}")


def get_model_dimensions(model):
    shape = model.input_shape
    if isinstance(shape, list):
        shape = shape[0]

    if len(shape) != 4:
        raise ValueError(f"Expected a 4D image model input, received: {shape}")

    height = int(shape[1]) if shape[1] is not None else 128
    width = int(shape[2]) if shape[2] is not None else 128
    channels = int(shape[3]) if shape[3] is not None else 1

    if channels not in (1, 3):
        raise ValueError(f"Unsupported model channel count: {channels}")

    return height, width, channels


# -------------------------------------------------------------------
# File decoding and preprocessing
# -------------------------------------------------------------------
IMAGE_TYPES = [
    "jpg", "jpeg", "jpe", "png", "webp", "bmp",
    "tif", "tiff", "gif", "heic", "heif",
    "dcm", "dicom", "pdf",
]

MAX_FILES_PER_BATCH = 10


def scale_to_unit_range(array):
    array = np.asarray(array, dtype=np.float32)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)

    low = float(np.min(array))
    high = float(np.max(array))

    if high <= low:
        return np.zeros_like(array, dtype=np.float32)

    return (array - low) / (high - low)


def decode_upload(uploaded_file, pdf_page_index=0):
    """Decode one uploaded image, DICOM file, or selected PDF page."""
    file_bytes = uploaded_file.getvalue()
    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix in (".dcm", ".dicom"):
        dataset = pydicom.dcmread(io.BytesIO(file_bytes))
        pixels = dataset.pixel_array

        # Handle multi-frame grayscale DICOM and color DICOM.
        if pixels.ndim == 3 and pixels.shape[-1] not in (3, 4):
            pixels = pixels[0]

        pixels_scaled = (scale_to_unit_range(pixels) * 255).astype(np.uint8)
        image = Image.fromarray(pixels_scaled)

        if image.mode not in ("L", "RGB"):
            image = image.convert("RGB")

        image = ImageOps.grayscale(image)

        if getattr(dataset, "PhotometricInterpretation", "") == "MONOCHROME1":
            image = ImageOps.invert(image)

        return image, "DICOM image"

    if suffix == ".pdf":
        if fitz is None:
            raise RuntimeError(
                "PDF support is missing. Add pymupdf to requirements.txt."
            )

        document = fitz.open(stream=file_bytes, filetype="pdf")

        if document.page_count == 0:
            raise ValueError("The PDF contains no pages.")

        page_index = max(0, min(pdf_page_index, document.page_count - 1))
        page = document.load_page(page_index)
        pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)

        image = Image.frombytes(
            "RGB",
            [pixmap.width, pixmap.height],
            pixmap.samples,
        )

        return image, f"PDF page {page_index + 1} of {document.page_count}"

    image = Image.open(io.BytesIO(file_bytes))
    image = ImageOps.exif_transpose(image)

    if getattr(image, "is_animated", False):
        image.seek(0)

    return image.convert("RGB"), f"{suffix.lstrip('.').upper()} image"


def get_pdf_page_count(uploaded_file):
    if fitz is None:
        raise RuntimeError("PDF support is missing. Install pymupdf.")

    document = fitz.open(
        stream=uploaded_file.getvalue(),
        filetype="pdf",
    )
    return document.page_count


def preprocess_for_model(image, model):
    """Resize to the loaded model's input shape and normalize to [0, 1]."""
    height, width, channels = get_model_dimensions(model)

    grayscale = ImageOps.grayscale(image)
    grayscale = grayscale.resize(
        (width, height),
        Image.Resampling.LANCZOS,
    )

    array = np.asarray(grayscale, dtype=np.float32) / 255.0

    if channels == 3:
        array = np.stack([array, array, array], axis=-1)
    else:
        array = np.expand_dims(array, axis=-1)

    return np.expand_dims(array.astype(np.float32), axis=0)


def predict_probability(model, batch):
    output = np.asarray(model.predict(batch, verbose=0)).reshape(-1)

    if output.size != 1:
        raise ValueError(
            f"Expected one binary prediction, but model returned {output.size} values."
        )

    probability = float(output[0])

    if not np.isfinite(probability) or probability < 0 or probability > 1:
        raise ValueError(
            f"Model output {probability} is not a probability between 0 and 1."
        )

    return probability


# -------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🫁 PneumoAI")
    st.caption("CHEST IMAGING • AI RESEARCH")
    st.divider()

    st.markdown("### Workspace")
    st.markdown("**Multi-image analysis**")
    st.markdown(
        "<span style='color:#9aabc2;font-size:.85rem'>"
        "Upload chest X-rays, review each file, and run batch inference."
        "</span>",
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("### Supported files")
    st.markdown(
        "- DICOM (.dcm, .dicom)\n"
        "- Images (JPG, JPEG, JPE, PNG, WEBP, BMP, TIFF, GIF)\n"
        "- HEIC / HEIF\n"
        "- PDF documents"
    )

    st.divider()
    st.caption("Educational research prototype · Not for clinical diagnosis")


# -------------------------------------------------------------------
# Main interface
# -------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">AI-assisted chest imaging workspace</div>
      <div class="hero-title">Pneumonia Detection</div>
      <div class="subtle">
        Upload one or multiple chest X-rays in supported formats and run
        inference using your trained convolutional neural network.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# Load model and show status
try:
    model, model_path = load_prediction_model()
    model_height, model_width, model_channels = get_model_dimensions(model)

    st.markdown(
        f"""
        <div class="panel">
          <span class="metric-label">Model status</span>
          <div style="display:flex;align-items:center;gap:.55rem;margin-top:.35rem">
            <span style="height:9px;width:9px;border-radius:50%;
            background:#42d6c5;display:inline-block"></span>
            <strong style="color:#eaf7f5">Model loaded and ready</strong>
          </div>
          <div class="subtle" style="font-size:.82rem;margin-top:.35rem">
            {html.escape(model_path)} · Input {model_height} × {model_width} × {model_channels}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
except Exception as error:
    model = None
    st.error(f"Model could not be loaded: {error}")
    st.info(
        "Check that best_cnn_model.h5 is committed to the root of your Space "
        "and that its TensorFlow/Keras version is compatible."
    )


# Multi-file uploader
st.markdown("### 01 · Upload medical images")
uploaded_files = st.file_uploader(
    "Choose one or more chest X-rays or PDF files",
    type=IMAGE_TYPES,
    accept_multiple_files=True,
    help=(
        "Supported: JPG, JPEG, JPE, PNG, WEBP, BMP, TIFF, GIF, "
        "HEIC, HEIF, DICOM and PDF."
    ),
)


# -------------------------------------------------------------------
# Multi-file preview and batch analysis
# -------------------------------------------------------------------
if not uploaded_files:
    st.markdown(
        """
        <div class="panel" style="text-align:center;padding:2.3rem 1rem">
          <div style="font-size:2.5rem">☁️</div>
          <h3 style="margin:.5rem 0">Your imaging workspace is ready</h3>
          <div class="subtle">
            Upload one or multiple files above to begin.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

else:
    if len(uploaded_files) > MAX_FILES_PER_BATCH:
        st.error(
            f"Please upload no more than {MAX_FILES_PER_BATCH} files per batch. "
            f"You currently selected {len(uploaded_files)}."
        )
        st.stop()

    st.success(f"{len(uploaded_files)} file(s) uploaded.")

    # Store decoded images and any per-file errors.
    decoded_files = []

    st.markdown("### 02 · Review uploaded files")

    for index, uploaded_file in enumerate(uploaded_files):
        suffix = Path(uploaded_file.name).suffix.lower()
        selected_pdf_page = 1

        try:
            if suffix == ".pdf":
                page_count = get_pdf_page_count(uploaded_file)

                if page_count > 1:
                    selected_pdf_page = st.selectbox(
                        f"Select page for {uploaded_file.name}",
                        options=list(range(1, page_count + 1)),
                        key=f"pdf_page_{index}_{uploaded_file.name}",
                    )

            image, description = decode_upload(
                uploaded_file,
                pdf_page_index=selected_pdf_page - 1,
            )

            decoded_files.append(
                {
                    "file": uploaded_file,
                    "image": image,
                    "description": description,
                }
            )

            with st.expander(
                f"{index + 1}. {uploaded_file.name}",
                expanded=(index == 0),
            ):
                st.image(
                    image,
                    caption=description,
                    use_container_width=True,
                )
                st.caption(
                    f"Dimensions: {image.width} × {image.height} px"
                )
                st.caption(
                    f"File size: {uploaded_file.size / (1024 * 1024):.2f} MB"
                )

        except Exception as error:
            st.error(
                f"Could not read {uploaded_file.name}: {error}"
            )

    st.markdown("### 03 · Batch AI analysis")

    if not decoded_files:
        st.warning("None of the uploaded files could be decoded.")

    elif model is None:
        st.warning("The trained model is unavailable. Resolve model loading first.")

    elif st.button(
        "Analyze all valid images",
        type="primary",
        use_container_width=True,
    ):
        results = []
        progress = st.progress(0)
        status = st.empty()

        for index, item in enumerate(decoded_files):
            uploaded_file = item["file"]
            image = item["image"]

            status.write(
                f"Analyzing {index + 1} of {len(decoded_files)}: "
                f"{uploaded_file.name}"
            )

            try:
                processed_image = preprocess_for_model(image, model)
                probability = predict_probability(model, processed_image)

                threshold = 0.45
                prediction = (
                    "Pneumonia"
                    if probability >= threshold
                    else "No Pneumonia"
                )

                results.append(
                    {
                        "filename": uploaded_file.name,
                        "image": image.copy(),
                        "description": item["description"],
                        "prediction": prediction,
                        "probability": probability,
                        "error": None,
                    }
                )

            except Exception as error:
                results.append(
                    {
                        "filename": uploaded_file.name,
                        "prediction": None,
                        "probability": None,
                        "error": str(error),
                    }
                )

            progress.progress((index + 1) / len(decoded_files))

        status.success("Batch analysis finished.")

        st.markdown("### Analysis results")

        successful = 0

        for result in results:
            st.markdown(
                f"**{html.escape(result['filename'])}**"
            )

            # Keep the original image and its result together in one row.
            image_col, prediction_col = st.columns(
                [1.15, 1],
                gap="large",
                vertical_alignment="center",
            )

            with image_col:
                if result.get("image") is not None:
                    st.image(
                        result["image"],
                        caption=result.get("description", "Uploaded image"),
                        use_container_width=True,
                    )
                else:
                    st.info("Image preview unavailable.")

            with prediction_col:
                if result["error"] is not None:
                    st.error(f"Analysis failed: {result['error']}")
                else:
                    successful += 1

                    if result["prediction"] == "Pneumonia":
                        st.error("Model classification: Pneumonia")
                    else:
                        st.success("Model classification: No Pneumonia")

                    st.metric(
                        "Pneumonia model output",
                        f"{result['probability']:.2%}",
                    )
                    st.progress(result["probability"])
                    st.caption(
                        f"Other-class complement: "
                        f"{1.0 - result['probability']:.2%}"
                    )

            st.divider()

        st.caption(
            f"Successfully analyzed {successful} of {len(results)} valid image(s)."
        )

        st.warning(
            "Educational research prototype only. These model outputs are not "
            "medical diagnoses and must not be used to guide treatment. "
            "Consult a qualified radiologist or clinician."
        )


# -------------------------------------------------------------------
# Footer - keep outside the upload if/else block
# -------------------------------------------------------------------
st.markdown(
    """
    <div style="text-align:center;color:#8192aa;font-size:.78rem;padding:1.5rem 0 0">
      PNEUMOAI · EDUCATIONAL RESEARCH PROTOTYPE · NOT A CLINICAL DEVICE
    </div>
    """,
    unsafe_allow_html=True,
)
