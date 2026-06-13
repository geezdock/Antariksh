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


st.set_page_config(page_title="Cloud Removal — ISRO BAH 2026", layout="wide")
st.title("Satellite Image Cloud Removal")
st.markdown("Upload a cloudy satellite image to reconstruct a cloud-free version using a U-Net.")

uploaded_file = st.file_uploader("Choose a cloudy satellite image", type=["png", "jpg", "jpeg", "tif"])

checkpoint_path = os.path.join("models", "checkpoints", "unet_final.pth")
model_available = os.path.exists(checkpoint_path)

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    input_img = cv2.imread(temp_path)
    input_img_rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Cloudy Image")
        st.image(input_img_rgb, use_container_width=True)

    if model_available:
        with st.spinner("Running inference..."):
            output_bgr = predict(temp_path, checkpoint_path=checkpoint_path)
        output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)

        with col2:
            st.subheader("Reconstructed Image")
            st.image(output_rgb, use_container_width=True)

        st.subheader("Evaluation Metrics")
        input_gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        output_gray = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2GRAY)

        min_dim = min(input_gray.shape[0], input_gray.shape[1])
        win_size = min_dim if min_dim % 2 == 1 else min_dim - 1
        win_size = max(3, win_size)

        input_resized = cv2.resize(input_gray, (256, 256))
        output_resized = cv2.resize(output_gray, (256, 256))

        psnr_val = compute_psnr(input_resized, output_resized)
        ssim_val = compute_ssim(input_resized, output_resized, channel_axis=None)

        m1, m2, m3 = st.columns(3)
        m1.metric("PSNR", f"{psnr_val:.2f} dB")
        m2.metric("SSIM", f"{ssim_val:.4f}")

        _, encoded_img = cv2.imencode(".png", output_bgr)
        st.download_button(
            label="Download Reconstructed Image",
            data=encoded_img.tobytes(),
            file_name="reconstructed.png",
            mime="image/png",
        )
    else:
        with col2:
            st.subheader("Reconstructed Image")
            st.info("No trained model found. Train the model first using `python models/train.py`.")
            st.info("For quick testing, run the synthetic data generation script first.")

    os.unlink(temp_path)
else:
    st.info("Awaiting image upload...")
