import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm

# -------------------------------
# 1️⃣ PAGE CONFIGURATION
# -------------------------------
st.set_page_config(page_title="Real vs Fake Image Classifier", layout="wide", page_icon="🧠")

# -------------------------------
# 2️⃣ CUSTOM CSS
# -------------------------------
st.markdown("""
<style>
    .stApp {
        background-color: #f5f7fa;
        font-family: 'Segoe UI', sans-serif;
    }
    h1 {
        text-align: center;
        color: #1a5276;
        font-weight: 700;
    }
    h2, h3 {
        color: #1a5276;
        text-align: center;
    }
    .description {
        text-align: center;
        color: #4a4a4a;
        font-size: 16px;
        margin-bottom: 20px;
    }
    .upload-box {
        border: 2px dashed #2874a6;
        border-radius: 15px;
        padding: 25px;
        background-color: #ffffff;
        text-align: center;
    }
    .result-card {
        background-color: #eaf2f8;
        border-radius: 12px;
        padding: 25px;
        margin-top: 15px;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
    }
    .footer {
        text-align: center;
        color: gray;
        font-size: 13px;
        margin-top: 40px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# 3️⃣ HEADER SECTION
# -------------------------------
st.markdown("<h1>🧠 Real vs Fake Image Classification</h1>", unsafe_allow_html=True)
st.markdown("<p class='description'>A hybrid CNN–Vision Transformer (ViT) based model for detecting the authenticity of images across multiple object categories.</p>", unsafe_allow_html=True)
st.markdown("---")

# -------------------------------
# 4️⃣ MODEL DEFINITION & LOADING
# -------------------------------
@st.cache_resource
def load_model():
    class FusionModel(nn.Module):
        def __init__(self, num_classes=2):
            super().__init__()
            # CNN: EfficientNet-B3
            self.cnn = timm.create_model('efficientnet_b3', pretrained=True)
            for name, param in self.cnn.named_parameters():
                if "blocks.6" not in name and "blocks.7" not in name and "classifier" not in name:
                    param.requires_grad = False
            self.cnn_out_dim = self.cnn.classifier.in_features
            self.cnn.classifier = nn.Identity()

            # ViT: DeiT-Tiny
            self.vit = timm.create_model('deit_tiny_patch16_224', pretrained=True)
            for name, param in self.vit.named_parameters():
                if "blocks.0" in name or "blocks.1" in name:
                    param.requires_grad = False
            self.vit.head = nn.Identity()
            self.vit_embed_dim = self.vit.embed_dim

            # Fusion classifier
            self.classifier = nn.Sequential(
                nn.Linear(self.cnn_out_dim + self.vit_embed_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )

        def forward(self, x):
            cnn_feat = self.cnn(x)
            vit_feat = self.vit(x)
            fused = torch.cat([cnn_feat, vit_feat], dim=1)
            return self.classifier(fused)

    model = FusionModel()
    state_dict = torch.load("feature_fusion_model.pth", map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model

with st.spinner("🔄 Loading model... Please wait"):
    model = load_model()
st.success("✅ Model loaded successfully!")

# -------------------------------
# 5️⃣ IMAGE TRANSFORMS
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# -------------------------------
# 6️⃣ IMAGE UPLOAD + PREDICTION SIDE-BY-SIDE
# -------------------------------
# -------------------------------
# 6️⃣ IMAGE UPLOAD + SIDE-BY-SIDE PREDICTION
# -------------------------------
st.markdown("### 📤 Upload an Image to check its Authenticity")
uploaded_file = st.file_uploader(
    "Drag & drop an image here, or click to browse.",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    # Preprocess
    img_tensor = transform(image).unsqueeze(0)

    # Prediction
    with st.spinner("🧠 Analyzing image..."):
        with torch.no_grad():
            output = model(img_tensor)
            probs = F.softmax(output, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_class].item() * 100

    label = "✅ Real Image" if pred_class == 1 else "❌ Fake Image"
    color = "#2ecc71" if pred_class == 1 else "#e74c3c"

    # Two columns layout: Image (left) | Prediction (right)
    left_col, right_col = st.columns([1, 1], gap="large")

    with left_col:
        st.markdown("<h4 style='text-align:center;'>📸 Uploaded Image</h4>", unsafe_allow_html=True)
        st.image(image, use_container_width=True)

    with right_col:
        st.markdown(f"""
        <div style="
            background-color:#eaf2f8;
            border-radius:12px;
            padding:35px;
            text-align:center;
            box-shadow:0 4px 8px rgba(0,0,0,0.05);
        ">
            <h2 style='color:{color};'>{label}</h2>
            <h4>Confidence: {confidence:.2f}%</h4>
        </div>
        """, unsafe_allow_html=True)

        st.progress(int(confidence))

        if confidence < 70:
            st.warning("⚠️ Low confidence — the image may be ambiguous.")
        else:
            st.info("✅ The model is confident about this prediction.")

else:
    st.info("👆 Please upload an image to begin classification.")


# -------------------------------
# 8️⃣ OPTIONAL GALLERY
# -------------------------------
st.markdown("---")
st.markdown("### 🖼️ Try with Sample Images")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.image("https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png", caption="Real Example", use_container_width=True)
with col2:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a8/Fake_miniature_of_train.jpg/640px-Fake_miniature_of_train.jpg", caption="Fake Example", use_container_width=True)
with col3:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/12/Red_apple.jpg/640px-Red_apple.jpg", caption="Real Example", use_container_width=True)
with col4:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/ae/Airplane_model_toy.jpg/640px-Airplane_model_toy.jpg", caption="Fake Example", use_container_width=True)

# -------------------------------
# 9️⃣ FOOTER
# -------------------------------
st.markdown("<div class='footer'>Developed as part of the Real vs Fake Image Classification Research Project © 2025</div>", unsafe_allow_html=True)
