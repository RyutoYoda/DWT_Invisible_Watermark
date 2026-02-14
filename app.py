import streamlit as st
import numpy as np
import pywt
from PIL import Image
import io

st.title("DWT Invisible Watermark Demo")

uploaded_file = st.file_uploader("画像をアップロード", type=["jpg", "jpeg", "png"])
watermark_text = st.text_input("ウォーターマーク文字列", "RyutoYoda")

def embed_watermark(image, text):
    img = image.convert("RGB")
    img_array = np.array(img, dtype=np.float64)

    # 32文字固定
    text = text[:32].ljust(32)
    bits = [int(b) for b in ''.join(format(ord(c), '08b') for c in text)]

    blue = img_array[:, :, 2]

    # Haar wavelet
    LL, (LH, HL, HH) = pywt.dwt2(blue, 'haar')

    flat_LL = LL.flatten()
    delta = 50.0
    repeat = 2

    for i, bit in enumerate(bits):
        for r in range(repeat):
            idx = (i * repeat + r) * 2
            if idx < len(flat_LL):
                val = flat_LL[idx]
                quantized = delta * np.round(val / delta)
                if bit == 1:
                    flat_LL[idx] = quantized + delta / 4
                else:
                    flat_LL[idx] = quantized - delta / 4

    LL_wm = flat_LL.reshape(LL.shape)

    blue_wm = pywt.idwt2((LL_wm, (LH, HL, HH)), 'haar')
    blue_wm = blue_wm[:blue.shape[0], :blue.shape[1]]
    blue_wm = np.clip(blue_wm, 0, 255)

    img_array[:, :, 2] = blue_wm
    result = Image.fromarray(img_array.astype(np.uint8))

    return result


if uploaded_file and watermark_text:
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Image", use_column_width=True)

    watermarked = embed_watermark(image, watermark_text)
    st.image(watermarked, caption="Watermarked Image (Invisible)", use_column_width=True)

    buf = io.BytesIO()
    watermarked.save(buf, format="JPEG", quality=95)
    byte_im = buf.getvalue()

    st.download_button(
        label="ウォーターマーク付き画像をダウンロード",
        data=byte_im,
        file_name="watermarked.jpg",
        mime="image/jpeg"
    )
