# =============================================================================
# SCRIPT: app.py (FINAL, COMPLETE, AND CORRECTED VERSION)
# =============================================================================
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageDraw
import io
import cv2
import tempfile
from collections import Counter
from ultralytics import YOLO
import numpy as np

# =============================================================================
# 0. BREED INFORMATION DATABASE (Comprehensive Version with Underscores)
# =============================================================================
breed_info_db = {
    "Alambadi": { "milk_yield": "Low (Draught Breed)", "weight_range": "Female: ~300 kg", "info": "A strong draught breed from Tamil Nadu.", "avg_length_cm": 125 },
    "Amritmahal": { "milk_yield": "Low (Draught Breed)", "weight_range": "Female: ~320 kg", "info": "A famous draught breed from Karnataka.", "avg_length_cm": 130 },
    "Ayrshire": { "milk_yield": "6000-7500 kg/lactation (Exotic)", "weight_range": "Female: ~600 kg", "info": "An exotic dairy breed from Scotland.", "avg_length_cm": 150 },
    "Bargur": { "milk_yield": "Low (Draught Breed)", "weight_range": "Female: ~280 kg", "info": "A fierce draught breed from the Bargur hills of Tamil Nadu.", "avg_length_cm": 120 },
    "Brown_Swiss": { "milk_yield": "5000-6000 kg/lactation (Exotic)", "weight_range": "Female: 600-650 kg", "info": "A hardy exotic dairy breed from Switzerland.", "avg_length_cm": 160 },
    "Dangi": { "milk_yield": "Low (Draught Breed)", "weight_range": "Female: ~300 kg", "info": "A draught breed from Gujarat, suited for heavy rainfall areas.", "avg_length_cm": 125 },
    "Deoni": { "milk_yield": "900-1200 kg/lactation", "weight_range": "Female: ~400 kg", "info": "A dual-purpose breed from Maharashtra.", "avg_length_cm": 135 },
    "Gir": { "milk_yield": "1500-2200 kg/lactation", "weight_range": "Female: 380-450 kg", "info": "Originating from Gujarat, known for high milk quality.", "avg_length_cm": 135 },
    "Guernsey": { "milk_yield": "4500-5500 kg/lactation (Exotic)", "weight_range": "Female: ~500 kg", "info": "Exotic breed from the Isle of Guernsey, famous for its golden-colored milk.", "avg_length_cm": 140 },
    "Hallikar": { "milk_yield": "Low (Draught Breed)", "weight_range": "Female: ~250 kg", "info": "A prominent draught breed from Karnataka.", "avg_length_cm": 130 },
    "Hariana": { "milk_yield": "1000-1500 kg/lactation", "weight_range": "Female: 310-350 kg", "info": "A popular dual-purpose breed from Haryana.", "avg_length_cm": 135 },
    "Holstein_Friesian": { "milk_yield": "6000-8000 kg/lactation (Exotic)", "weight_range": "Female: 600-700 kg", "info": "The highest milk-producing dairy animal in the world.", "avg_length_cm": 170 },
    "Jersey": { "milk_yield": "4000-5000 kg/lactation (Exotic)", "weight_range": "Female: 400-450 kg", "info": "Exotic breed known for high butterfat content in milk.", "avg_length_cm": 130 },
    "Kangayam": { "milk_yield": "Low (Draught Breed)", "weight_range": "Female: ~340 kg", "info": "A strong draught breed from Tamil Nadu, used in Jallikattu.", "avg_length_cm": 140 },
    "Kankrej": { "milk_yield": "1300-1800 kg/lactation", "weight_range": "Female: 320-370 kg", "info": "A dual-purpose breed from the Rann of Kutch.", "avg_length_cm": 140 },
    "Kasargod": { "milk_yield": "Low (Dwarf Breed)", "weight_range": "Female: ~150 kg", "info": "A dwarf cattle breed from Kerala.", "avg_length_cm": 90 },
    "Kenkatha": { "milk_yield": "Low (Draught Breed)", "weight_range": "Female: ~200 kg", "info": "A small draught breed from Uttar Pradesh.", "avg_length_cm": 115 },
    "Kherigarh": { "milk_yield": "Low (Draught Breed)", "weight_range": "Female: ~300 kg", "info": "A draught breed from Uttar Pradesh, known for its activeness.", "avg_length_cm": 130 },
    "Khillari": { "milk_yield": "Low (Draught Breed)", "weight_range": "Female: ~380 kg", "info": "A draught breed from Maharashtra, known for its speed.", "avg_length_cm": 140 },
    "Krishna_Valley": { "milk_yield": "Low (Draught Breed)", "weight_range": "Female: ~420 kg", "info": "A large draught breed from Karnataka.", "avg_length_cm": 150 },
    "Malnad_gidda": { "milk_yield": "Low, data variable (Dwarf breed)", "weight_range": "Female: ~100 kg", "info": "A dwarf breed from the hilly regions of Karnataka, well-adapted to heavy rainfall.", "avg_length_cm": 85 },
    "Nagori": { "milk_yield": "Low (Draught Breed)", "weight_range": "Female: ~300 kg", "info": "A fine trotting draught breed from Rajasthan.", "avg_length_cm": 130 },
    "Nimari": { "milk_yield": "Low (Draught Breed)", "weight_range": "Female: ~300 kg", "info": "A draught breed from Madhya Pradesh.", "avg_length_cm": 130 },
    "Ongole": { "milk_yield": "800-1200 kg/lactation", "weight_range": "Female: 430-480 kg", "info": "A large draught breed from Andhra Pradesh, known for its strength.", "avg_length_cm": 150 },
    "Pulikulam": { "milk_yield": "Low (Draught Breed)", "weight_range": "Female: ~250 kg", "info": "A small draught breed from Tamil Nadu, also used in Jallikattu.", "avg_length_cm": 115 },
    "Rathi": { "milk_yield": "1500-2000 kg/lactation", "weight_range": "Female: ~290 kg", "info": "A milch breed from the arid regions of Rajasthan.", "avg_length_cm": 125 },
    "Red_Dane": { "milk_yield": "6000-7000 kg/lactation (Exotic)", "weight_range": "Female: ~600 kg", "info": "An exotic dairy breed from Denmark.", "avg_length_cm": 160 },
    "Red_Sindhi": { "milk_yield": "1500-2200 kg/lactation", "weight_range": "Female: 300-350 kg", "info": "A popular heat-tolerant dairy breed.", "avg_length_cm": 120 },
    "Sahiwal": { "milk_yield": "1800-2500 kg/lactation", "weight_range": "Female: 350-450 kg", "info": "One of the best Zebu dairy breeds from the Punjab region.", "avg_length_cm": 135 },
    "Tharparkar": { "milk_yield": "1600-2200 kg/lactation", "weight_range": "Female: 380-420 kg", "info": "A hardy dual-purpose breed from the Thar Desert.", "avg_length_cm": 135 },
    "Umblachery": { "milk_yield": "Low (Draught Breed)", "weight_range": "Female: ~280 kg", "info": "A draught breed from Tamil Nadu, suited for marshy rice fields.", "avg_length_cm": 110 },
    "Vechur": { "milk_yield": "Low (Dwarf Breed)", "weight_range": "Female: ~130 kg", "info": "The smallest cattle breed in the world, from Kerala.", "avg_length_cm": 90 },
    "Banni": { "milk_yield": "1800-2500 kg/lactation", "weight_range": "Female: ~350 kg", "info": "A hardy buffalo breed from the Kutch region of Gujarat.", "avg_length_cm": 130 },
    "Bhadawari": { "milk_yield": "900-1200 kg/lactation", "weight_range": "Female: ~375 kg", "info": "A buffalo breed from Uttar Pradesh, known for high milk fat content.", "avg_length_cm": 125 },
    "Jaffrabadi": { "milk_yield": "1800-2500 kg/lactation", "weight_range": "Female: 400-500 kg", "info": "A very heavy buffalo breed from the Gir forests of Gujarat.", "avg_length_cm": 140 },
    "Mehsana": { "milk_yield": "1800-2200 kg/lactation", "weight_range": "Female: 400-450 kg", "info": "A dairy buffalo from Gujarat, a Murrah/Surti crossbreed.", "avg_length_cm": 135 },
    "Murrah": { "milk_yield": "1800-2600 kg/lactation", "weight_range": "Female: 450-550 kg", "info": "A world-renowned buffalo breed from Haryana.", "avg_length_cm": 145 },
    "Nagpuri": { "milk_yield": "1000-1200 kg/lactation", "weight_range": "Female: ~350 kg", "info": "A dual-purpose buffalo from Maharashtra.", "avg_length_cm": 130 },
    "Nili_Ravi": { "milk_yield": "1800-2500 kg/lactation", "weight_range": "Female: 450-550 kg", "info": "A dairy buffalo from Punjab, known for its wall eyes.", "avg_length_cm": 140 },
    "Surti": { "milk_yield": "1500-1700 kg/lactation", "weight_range": "Female: 380-420 kg", "info": "A dairy buffalo from Gujarat with sickle-shaped horns.", "avg_length_cm": 130 },
    "Toda": { "milk_yield": "Low, ~500 kg/lactation", "weight_range": "Female: ~320 kg", "info": "A semi-wild buffalo from the Nilgiri Hills.", "avg_length_cm": 120 }
}

# =============================================================================
# 1. MODEL DEFINITIONS
# =============================================================================
class MultiOutputModel(nn.Module):
    def __init__(self, num_breeds, num_types):
        super().__init__()
        self.base_model = models.mobilenet_v2(weights='IMAGENET1K_V1')
        for param in self.base_model.features[:-4].parameters(): param.requires_grad = False
        num_ftrs = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(num_ftrs, 512), nn.ReLU())
        self.breed_head = nn.Linear(512, num_breeds)
        self.type_head = nn.Linear(512, num_types)
    def forward(self, x):
        x = self.base_model(x)
        return self.breed_head(x), self.type_head(x)

def create_health_model(num_classes=2):
    model = models.mobilenet_v2(weights='IMAGENET1K_V1')
    for param in model.parameters(): param.requires_grad = False
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    return model

# =============================================================================
# 2. LOAD ALL TRAINED MODELS
# =============================================================================
@st.cache_resource
def load_models():
    device = torch.device("cpu")
    breed_model_data = torch.load("bovine_classifier_model.pth", map_location=device)
    num_breeds = len(breed_model_data["breed_to_idx"])
    num_types = len(breed_model_data["type_to_idx"])
    breed_model = MultiOutputModel(num_breeds=num_breeds, num_types=num_types)
    breed_model.load_state_dict(breed_model_data["model_state"])
    breed_model.eval()
    idx_to_breed = {v: k for k, v in breed_model_data["breed_to_idx"].items()}
    idx_to_type = {v: k for k, v in breed_model_data["type_to_idx"].items()}

    health_model_data = torch.load("health_classifier_model.pth", map_location=device)
    health_model = create_health_model(len(health_model_data["class_names"]))
    health_model.load_state_dict(health_model_data["model_state"])
    health_model.eval()
    
    segmentation_model = YOLO("best.pt")
    return breed_model, idx_to_breed, idx_to_type, health_model, health_model_data["class_names"], segmentation_model

breed_model, idx_to_breed, idx_to_type, health_model, health_class_names, segmentation_model = load_models()

# =============================================================================
# 3. DEFINE THE PREDICTION & MEASUREMENT FUNCTION
# =============================================================================
def analyze_image(image):
    # --- Classification ---
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        breed_logits, type_logits = breed_model(image_tensor)
        p_type = idx_to_type[type_logits.argmax(1).item()]
        p_breed = idx_to_breed[breed_logits.argmax(1).item()]
        
        health_logits = health_model(image_tensor)
        p_health = health_class_names[health_logits.argmax(1).item()]

    # --- Segmentation and Measurement ---
    results = segmentation_model(image)
    length_cm, height_cm, p_weight_est = 0, 0, 0
    debug_image = image.copy()
    draw = ImageDraw.Draw(debug_image)

    if results[0].masks is not None and len(results[0].masks) > 0:
        mask = results[0].masks[0].data[0].cpu().numpy()
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, w, h = cv2.boundingRect(contours[0])
            draw.rectangle([x, y, x + w, y + h], outline="red", width=5)
            
            pixel_to_cm = 0.22 # Default fallback value
            if p_breed in breed_info_db and 'avg_length_cm' in breed_info_db[p_breed] and w > 0:
                known_length = breed_info_db[p_breed]['avg_length_cm']
                pixel_to_cm = known_length / w
            
            length_cm = w * pixel_to_cm
            height_cm = h * pixel_to_cm
            
            p_weight_est = ((height_cm**2) * length_cm) / 10840 if length_cm > 0 else 0

    return p_type, p_breed, p_health, length_cm, height_cm, p_weight_est, debug_image

# =============================================================================
# 4. CREATE THE STREAMLIT WEB APP INTERFACE
# =============================================================================
st.set_page_config(page_title="Bovine Analysis Tool", layout="wide")
st.title("🐄 Advanced Bovine Analysis Tool 🐃")

tab1, tab2 = st.tabs(["🖼️ Image Analysis", "🎬 Video Analysis"])

with tab1:
    st.header("Analyze Still Images")
    uploaded_files = st.file_uploader("Upload images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="image_uploader")

    if uploaded_files:
        if st.button('Analyze Images', key="image_button"):
            for uploaded_file in uploaded_files:
                image = Image.open(uploaded_file).convert("RGB")
                
                with st.spinner(f'Analyzing {uploaded_file.name}...'):
                    p_type, p_breed, p_health, p_len_cm, p_hgt_cm, p_wgt, debug_img = analyze_image(image)
                    
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.image(image, caption=f'Uploaded: {uploaded_file.name}', use_column_width=True)
                        st.image(debug_img, caption=f'Model Detection Outline', use_column_width=True)
                    with col2:
                        st.success(f"AI Analysis for {uploaded_file.name}")
                        st.metric("Animal Type", p_type)
                        st.metric("Predicted Breed", p_breed)
                        st.metric("Visual Health Status", p_health)
                        
                        st.divider()
                        st.info(f"**Automated Measurements & Estimations**", icon="📏")
                        st.metric("Approx. Body Length", f"{p_len_cm:.2f} cm")
                        st.metric("Approx. Body Height", f"{p_hgt_cm:.2f} cm")
                        st.metric("Estimated Live Weight", f"~{p_wgt:.2f} kg")
                        st.caption("Weight is dynamically estimated based on the predicted breed. Requires a clear, side-view image.")

                        st.divider()
                        st.info(f"**Breed Information Database**", icon="ℹ️")
                        if p_breed in breed_info_db:
                            info = breed_info_db[p_breed]
                            st.write(f"**Typical Milk Yield:** {info['milk_yield']}")
                            st.write(f"**Typical Weight Range:** {info['weight_range']}")
                            st.caption(info['info'])
                        else:
                            st.warning("Breed info not in database.")
                st.divider()

with tab2:
    st.header("Analyze a Video File")
    st.info("Video analysis can be enabled in a future version.")