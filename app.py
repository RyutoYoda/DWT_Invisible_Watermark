import streamlit as st
import numpy as np
import pywt
from PIL import Image
import io
import uuid
import duckdb
import hashlib
from datetime import datetime

# -----------------------
# DB初期化
# -----------------------
conn = duckdb.connect("watermark.db")

conn.execute("""
CREATE TABLE IF NOT EXISTS images (
    uuid TEXT PRIMARY KEY,
    owner TEXT,
    created_at TIMESTAMP,
    original_hash TEXT
)
""")

# -----------------------
# ハッシュ
# -----------------------
def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()

# -----------------------
# DWT埋め込み
# -----------------------
def embed_watermark(image, text):
    img = image.convert("RGB")
    img_array = np.array(img, dtype=np.float64)

    text = text[:32].ljust(32)
    bits = [int(b) for b in ''.join(format(ord(c), '08b') for c in text)]

    blue = img_array[:, :, 2]
    LL, (LH, HL, HH) = pywt.dwt2(blue, 'haar')

    flat_LL = LL.flatten()
    delta = 80.0
    repeat = 3

    for i, bit in enumerate(bits):
        for r in range(repeat):
            idx = (i * repeat + r) * 2
            if idx < len(flat_LL):
                val = flat_LL[idx]
                quantized = delta * np.round(val / delta)
                if bit == 1:
                    flat_LL[idx] = quantized + delta / 3
                else:
                    flat_LL[idx] = quantized - delta / 3

    LL_wm = flat_LL.reshape(LL.shape)

    blue_wm = pywt.idwt2((LL_wm, (LH, HL, HH)), 'haar')
    blue_wm = blue_wm[:blue.shape[0], :blue.shape[1]]
    blue_wm = np.clip(blue_wm, 0, 255)

    img_array[:, :, 2] = blue_wm
    return Image.fromarray(img_array.astype(np.uint8))

# -----------------------
# DWT抽出
# -----------------------
def extract_watermark(image):
    img = image.convert("RGB")
    img_array = np.array(img, dtype=np.float64)

    blue = img_array[:, :, 2]
    LL, _ = pywt.dwt2(blue, 'haar')
    flat_LL = LL.flatten()

    delta = 80.0
    repeat = 3
    total_bits = 32 * 8
    bits = []

    for i in range(total_bits):
        votes = []
        for r in range(repeat):
            idx = (i * repeat + r) * 2
            if idx < len(flat_LL):
                val = flat_LL[idx]
                quantized = delta * np.round(val / delta)
                diff = val - quantized
                votes.append(1 if diff > 0 else 0)
        if votes:
            bits.append(round(sum(votes) / len(votes)))

    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        if len(byte) == 8:
            chars.append(chr(int("".join(map(str, byte)), 2)))

    return "".join(chars).strip()

# -----------------------
# UI
# -----------------------
st.title("DWT Invisible Watermark App")

mode = st.radio("モード選択", ["埋め込み", "照会"])

# -----------------------
# 埋め込み
# -----------------------
if mode == "埋め込み":
    uploaded_file = st.file_uploader("画像アップロード", type=["png", "jpg", "jpeg"])
    owner = st.text_input("所有者名", "owner_name")
    watermark_text = st.text_input("ウォーターマーク文字列", "埋め込みたい文字列")

    if uploaded_file and owner:
        image_bytes = uploaded_file.read()
        original_hash = sha256_bytes(image_bytes)
        image = Image.open(io.BytesIO(image_bytes))

        # 安定用：UUIDは32文字hex
        new_uuid = uuid.uuid4().hex

        # 埋め込み文字列はUUID（追跡用）
        final_text = new_uuid

        watermarked = embed_watermark(image, final_text)

        conn.execute("""
        INSERT INTO images VALUES (?, ?, ?, ?)
        """, (new_uuid, owner, datetime.now(), original_hash))

        st.success(f"発行UUID: {new_uuid}")

        st.image(watermarked, caption="ウォーターマーク付き画像")

        buf = io.BytesIO()
        watermarked.save(buf, format="PNG")  # PNG保存で劣化防止

        st.download_button(
            label="ダウンロード",
            data=buf.getvalue(),
            file_name="watermarked.png",
            mime="image/png"
        )

# -----------------------
# 照会
# -----------------------
if mode == "照会":
    uploaded_file = st.file_uploader("照会したい画像をアップロード", type=["png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        extracted = extract_watermark(image)

        st.write("抽出UUID:", extracted)

        result = conn.execute("""
        SELECT * FROM images WHERE uuid = ?
        """, (extracted,)).fetchall()

        if result:
            st.success("登録済み画像です")
            st.write(result)
        else:
            st.error("登録情報が見つかりません")
