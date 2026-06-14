import os
import sys
import tempfile

import cv2
import numpy as np
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from models.predict import predict
from evaluation.psnr import compute_psnr
from evaluation.ssim import compute_ssim

st.set_page_config(
    page_title="Cloud Removal — ISRO BAH 2026",
    page_icon="◉",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
    #root > div:first-child {
        padding-top: 1rem;
    }
    .stApp {
        background: #0E1117;
    }
    .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        font-family: 'Inter', -apple-system, sans-serif;
        font-weight: 300;
        font-size: 2.2rem;
        letter-spacing: -0.02em;
        color: #F3F4F6;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-family: 'Inter', -apple-system, sans-serif;
        font-size: 0.85rem;
        color: #9CA3AF;
        margin-bottom: 0;
        letter-spacing: 0.02em;
    }
    .divider {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, #2D3139 0%, transparent 100%);
        margin: 1.5rem 0;
    }
    .upload-card {
        border: 1.5px dashed #2D3139;
        border-radius: 12px;
        padding: 2.5rem 1rem;
        text-align: center;
        background: #1C1F26;
        transition: border-color 0.2s;
        margin-bottom: 1.5rem;
    }
    .upload-card:hover {
        border-color: #7C3AED;
    }
    .upload-card .icon {
        font-size: 2rem;
        color: #7C3AED;
        margin-bottom: 0.5rem;
    }
    .upload-card .text {
        color: #9CA3AF;
        font-size: 0.9rem;
    }
    .upload-card .hint {
        color: #4B5563;
        font-size: 0.75rem;
        margin-top: 0.3rem;
    }
    div[data-testid="stFileUploader"] {
        margin-bottom: 0;
    }
    div[data-testid="stFileUploader"] section {
        padding: 0;
        border: none;
        background: transparent;
    }
    div[data-testid="stFileUploader"] section > div:first-child {
        display: none;
    }
    div[data-testid="stFileUploader"] section > div:last-child {
        border: none;
        padding: 0;
    }
    div[data-testid="stFileUploader"] button {
        background: #7C3AED;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-size: 0.85rem;
        font-weight: 500;
    }
    div[data-testid="stFileUploader"] button:hover {
        background: #6D28D9;
    }
    .image-card {
        background: #1C1F26;
        border: 1px solid #2D3139;
        border-radius: 12px;
        padding: 0.75rem;
        overflow: hidden;
    }
    .image-card img {
        border-radius: 8px;
        width: 100%;
        height: auto;
    }
    .image-label {
        text-align: center;
        font-size: 0.75rem;
        color: #9CA3AF;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.5rem;
        margin-bottom: 0;
    }
    .image-label.accent {
        color: #7C3AED;
    }
    .metric-card {
        background: #1C1F26;
        border: 1px solid #2D3139;
        border-radius: 12px;
        padding: 1.25rem 1rem;
        text-align: center;
    }
    .metric-card .value {
        font-size: 1.75rem;
        font-weight: 600;
        color: #10B981;
        line-height: 1.2;
    }
    .metric-card .label {
        font-size: 0.7rem;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.25rem;
    }
    div[data-testid="stDownloadButton"] button {
        background: #7C3AED;
        color: white;
        border: none;
        border-radius: 12px;
        padding: 1.25rem 1rem;
        width: 100%;
        height: 100%;
        font-size: 0.85rem;
        font-weight: 500;
        letter-spacing: 0.02em;
        transition: background 0.2s;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
    }
    div[data-testid="stDownloadButton"] button:hover {
        background: #6D28D9;
    }
    .stSpinner > div {
        border-color: #7C3AED !important;
    }
    .empty-state {
        display: flex;
        align-items: center;
        justify-content: center;
        min-height: 220px;
        color: #4B5563;
        font-size: 0.85rem;
    }
    footer {
        margin-top: 3rem;
        text-align: center;
    }
    footer hr {
        border: none;
        height: 1px;
        background: #1F2937;
        margin-bottom: 1rem;
    }
    footer p {
        font-size: 0.7rem;
        color: #4B5563;
        letter-spacing: 0.03em;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown("<h1>Cloud Removal</h1>", unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Satellite Image Reconstruction &mdash; ISRO BAH 2026</p>',
    unsafe_allow_html=True,
)
st.markdown('<hr class="divider" />', unsafe_allow_html=True)

checkpoint_path = os.path.join("models", "checkpoints", "unet_final.pth")
model_available = os.path.exists(checkpoint_path)

st.markdown(
    """
<div class="upload-card">
    <div class="icon">☁</div>
    <div class="text">Upload a cloudy satellite image</div>
    <div class="hint">PNG &middot; JPG &middot; TIFF</div>
</div>
""",
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "", type=["png", "jpg", "jpeg", "tif"], label_visibility="collapsed"
)

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    input_img = cv2.imread(temp_path)
    if input_img is None:
        st.error("Could not read the uploaded file. Please try a different image.")
        os.unlink(temp_path)
        st.stop()

    input_img_rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

    cols = st.columns(2, gap="small")

    with cols[0]:
        st.markdown(
            '<div class="image-card">',
            unsafe_allow_html=True,
        )
        st.image(input_img_rgb, use_container_width=True)
        st.markdown(
            '<p class="image-label">Cloudy Input</p>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with cols[1]:
        st.markdown(
            '<div class="image-card">',
            unsafe_allow_html=True,
        )
        if model_available:
            with st.spinner(""):
                output_bgr = predict(temp_path, checkpoint_path=checkpoint_path)
            output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
            st.image(output_rgb, use_container_width=True)
            st.markdown(
                '<p class="image-label accent">Reconstructed Output</p>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="empty-state">No trained model found</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<p class="image-label">Reconstructed Output</p>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    if model_available:
        st.markdown('<hr class="divider" />', unsafe_allow_html=True)

        input_gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        output_gray = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2GRAY)
        input_resized = cv2.resize(input_gray, (256, 256))
        output_resized = cv2.resize(output_gray, (256, 256))

        psnr_val = compute_psnr(input_resized, output_resized)
        ssim_val = compute_ssim(input_resized, output_resized, channel_axis=None)

        mcols = st.columns(3, gap="small")

        with mcols[0]:
            st.markdown(
                f'<div class="metric-card"><div class="value">{psnr_val:.2f}</div><div class="label">PSNR (dB)</div></div>',
                unsafe_allow_html=True,
            )

        with mcols[1]:
            st.markdown(
                f'<div class="metric-card"><div class="value">{ssim_val:.4f}</div><div class="label">SSIM</div></div>',
                unsafe_allow_html=True,
            )

        with mcols[2]:
            _, encoded_img = cv2.imencode(".png", output_bgr)
            st.download_button(
                label="⬇ Download Reconstructed",
                data=encoded_img.tobytes(),
                file_name="reconstructed.png",
                mime="image/png",
            )

    os.unlink(temp_path)

st.markdown(
    """
<footer>
    <hr />
    <p>PyTorch &middot; U-Net &middot; Streamlit &nbsp;&nbsp;|&nbsp;&nbsp; v1.0</p>
</footer>
""",
    unsafe_allow_html=True,
)
