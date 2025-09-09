
# =============================================================================
# SCRIPT: app.py (FINAL WEBSITE - Frontend + Backend + Full DB Integrated)
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
# 0. BREED INFORMATION DATABASE (Comprehensive Version with Underscores and avg_length_cm)
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

# Preload all models once to avoid re-loading in analyze_image
BREED_MODEL, IDX_TO_BREED, IDX_TO_TYPE, HEALTH_MODEL, HEALTH_CLASS_NAMES, SEGMENTATION_MODEL = load_models()

# =============================================================================
# 3. DEFINE THE PREDICTION & MEASUREMENT FUNCTION
# =============================================================================
def analyze_image(image):
    breed_model = BREED_MODEL
    idx_to_breed = IDX_TO_BREED
    idx_to_type = IDX_TO_TYPE
    health_model = HEALTH_MODEL
    health_class_names = HEALTH_CLASS_NAMES
    segmentation_model = SEGMENTATION_MODEL

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
            
            pixel_to_cm = 0.22
            if p_breed in breed_info_db and 'avg_length_cm' in breed_info_db[p_breed] and w > 0:
                known_length = breed_info_db[p_breed]['avg_length_cm']
                pixel_to_cm = known_length / w
            
            length_cm = w * pixel_to_cm
            height_cm = h * pixel_to_cm
            p_weight_est = ((height_cm**2) * length_cm) / 10840 if length_cm > 0 else 0

    return p_type, p_breed, p_health, length_cm, height_cm, p_weight_est, debug_image

# =============================================================================
# 4. PAGE CONFIGURATION AND STYLING
# =============================================================================
st.set_page_config(page_title="Cow Breed Detection AI", page_icon="🐄", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
/* Global resets and helpers */
* { box-sizing: border-box; -webkit-text-fill-color: inherit; }
body {
  margin: 0;
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, Apple Color Emoji, Segoe UI Emoji, Segoe UI Symbol;
  color: #2d3748;
  background: #f7fafc;
}
hr { border: none; height: 1px; background: #e2e8f0; }

/* Header (from Streamlit CSS) */
.main-header {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  padding: 4rem 2rem;
  border-radius: 20px;
  margin-bottom: 2rem;
  position: relative;
  overflow: hidden;
  color: white;
}
.main-header::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0; bottom: 0;
  background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.05'%3E%3Ccircle cx='30' cy='30' r='2'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
  animation: float 20s ease-in-out infinite;
}
@keyframes float {
  0%, 100% { transform: translateY(0px); }
  50% { transform: translateY(-10px); }
}
.hero-title {
  font-size: 3.5rem;
  font-weight: 800;
  margin-bottom: 1rem;
  background: linear-gradient(45deg, #fff, #e0e7ff);
  background-clip: text;
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  text-align: left;
  line-height: 1.2;
}
.hero-subtitle {
  font-size: 1.2rem;
  opacity: 0.9;
  margin-bottom: 2rem;
  max-width: 600px;
  line-height: 1.6;
}
.cta-button {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 1rem 2rem;
  border: none;
  border-radius: 50px;
  font-size: 1.1rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
  text-decoration: none;
  display: inline-block;
  margin-right: 1rem;
  margin-top: 1rem;
}
.cta-button:hover {
  transform: translateY(-2px);
  box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
}
.secondary-button {
  background: transparent;
  color: white;
  border: 2px solid white;
  padding: 1rem 2rem;
  border-radius: 50px;
  font-size: 1.1rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
  text-decoration: none;
  display: inline-block;
  margin-top: 1rem;
}
.secondary-button:hover { background: white; color: #667eea; }

/* Section titles */
.section-title {
  font-size: 2.5rem;
  font-weight: 700;
  text-align: center;
  margin: 3rem 0 2rem 0;
  color: #2d3748;
  position: relative;
}
.section-title::after {
  content: '';
  position: absolute;
  bottom: -10px; left: 50%;
  transform: translateX(-50%);
  width: 80px; height: 4px;
  background: linear-gradient(90deg, #667eea, #764ba2);
  border-radius: 2px;
}

/* Stats section */
.stats-container {
  background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%);
  padding: 3rem 2rem;
  border-radius: 20px;
  color: white;
  margin: 3rem 0;
}
.stats-grid {
  display: grid;
  gap: 1rem;
  grid-template-columns: repeat(4, 1fr);
}
.stat-item { text-align: center; padding: 1rem; }
.stat-number {
  font-size: 2.5rem;
  font-weight: 800;
  color: #63b3ed;
  display: block;
  margin-bottom: 0.5rem;
}
.stat-label {
  font-size: 0.9rem;
  opacity: 0.8;
  text-transform: uppercase;
  letter-spacing: 1px;
}

/* Features */
.features-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1.25rem;
}
.feature-card {
  background: white;
  padding: 2rem;
  border-radius: 15px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.1);
  border: 1px solid #f0f0f0;
  transition: all 0.3s ease;
  height: 100%;
  position: relative;
  overflow: hidden;
}
.feature-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0;
  width: 100%; height: 4px;
  background: linear-gradient(90deg, #667eea, #764ba2);
}
.feature-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 20px 40px rgba(0,0,0,0.15);
}
.feature-icon { font-size: 3rem; margin-bottom: 1rem; display: block; }
.feature-title { font-size: 1.3rem; font-weight: 600; margin-bottom: 1rem; color: #2d3748; }
.feature-description { color: #718096; line-height: 1.6; font-size: 0.95rem; }

/* Tech stack */
.tech-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1.25rem;
  align-items: start;
}
.tech-badge {
  display: inline-block;
  background: linear-gradient(135deg, #667eea, #764ba2);
  color: white;
  padding: 0.5rem 1rem;
  border-radius: 25px;
  font-size: 0.8rem;
  font-weight: 500;
  margin: 0.25rem;
}
.tech-badges { margin-top: 0.5rem; }

/* Upload area (Demo) */
.upload-area {
  border: 2px dashed #cbd5e0;
  border-radius: 15px;
  padding: 3rem 2rem;
  text-align: center;
  background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
  transition: all 0.3s ease;
  margin: 2rem 0;
}
.upload-area:hover {
  border-color: #667eea;
  background: linear-gradient(135deg, #edf2f7 0%, #e2e8f0 100%);
}

/* Result layout (Demo) */
.result-card {
  background: white;
  padding: 2rem;
  border-radius: 20px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.1);
  margin: 2rem 0;
}
.result-title { color: #2d3748; margin-bottom: 1.5rem; text-align: center; }
.result-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1.25rem;
}
.result-col { display: flex; flex-direction: column; gap: 1rem; }
.result-image {
  width: 100%;
  height: auto;
  border-radius: 12px;
  border: 1px solid #edf2f7;
}

/* Panels and pills */
.panel {
  color: white;
  padding: 1.25rem 1.5rem;
  border-radius: 15px;
  margin: 1rem 0 1.25rem 0;
}
.panel-green { background: linear-gradient(135deg, #48bb78 0%, #38a169 100%); }
.panel-purple { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
.panel-orange { background: linear-gradient(135deg, #ed8936 0%, #dd6b20 100%); }
.panel-hero { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }

.pill {
  background: #f8fafc;
  padding: 1rem;
  border-radius: 10px;
  margin-bottom: 1rem;
  border-left: 4px solid #667eea;
}
.pill-purple { border-left-color: #764ba2; }
.pill-orange { border-left-color: #ed8936; }
.pill-row { display: flex; align-items: center; gap: 1rem; }
.pill-icon { font-size: 1.5rem; }
.pill-label { margin: 0; font-weight: 600; color: #2d3748; }
.pill-value { margin: 0; color: #4a5568; font-size: 1.1rem; }

/* About */
.card {
  background: white;
  padding: 3rem 2rem;
  border-radius: 20px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.1);
  margin-bottom: 2rem;
}
.features-auto-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 1.25rem;
}
.feature-line {
  background: #f8fafc;
  padding: 1.5rem;
  border-radius: 10px;
  border-left: 4px solid;
}
.feature-blue { border-left-color: #667eea; }
.feature-green { border-left-color: #48bb78; }
.feature-orange { border-left-color: #ed8936; }
.feature-purple { border-left-color: #9f7aea; }

/* Contact */
.contact-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1.25rem;
}
.contact-row {
  display: flex;
  align-items: center;
  margin-bottom: 1rem;
  padding: 1rem;
  background: #f8fafc;
  border-radius: 10px;
}
.contact-icon { font-size: 1.5rem; margin-right: 1rem; }
.contact-form {
  display: grid;
  gap: 0.75rem;
}
.contact-form input,
.contact-form textarea {
  width: 100%;
  padding: 0.875rem 1rem;
  border-radius: 10px;
  border: 1px solid #e2e8f0;
  outline: none;
  font-size: 0.95rem;
  background: #ffffff;
}
.contact-form input:focus,
.contact-form textarea:focus {
  border-color: #cbd5e0;
}

/* Footer */
.footer-wrap {
  text-align: center;
  padding: 2rem 0;
  color: #718096;
  background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
  border-radius: 15px;
  margin: 3rem 0 1rem 0;
}
.footer-badge {
  background: linear-gradient(135deg, #667eea, #764ba2);
  color: white;
  padding: 0.5rem 1rem;
  border-radius: 20px;
  font-size: 0.9rem;
}

/* Dividers */
.section-divider {
  border: none;
  height: 2px;
  background: linear-gradient(90deg, #667eea, #764ba2);
  margin: 3rem 0;
}

/* Responsive */
@media (max-width: 1024px) {
  .stats-grid { grid-template-columns: repeat(2, 1fr); }
  .features-grid { grid-template-columns: 1fr 1fr; }
  .tech-grid { grid-template-columns: 1fr; }
  .result-grid { grid-template-columns: 1fr; }
  .contact-grid { grid-template-columns: 1fr; }
}
@media (max-width: 768px) {
  .hero-title { font-size: 2.5rem; }
  .main-header { padding: 2rem 1rem; text-align: center; }
  .feature-card { margin-bottom: 1rem; }
}

/* Visibility fixes for Key Features text */
.feature-card,
.features-auto-grid .feature-line {
  background: #ffffff !important;
  color: #2d3748 !important;
  -webkit-text-fill-color: initial !important;
}
.feature-card h3.feature-title { color: #2d3748 !important; }
.feature-card p.feature-description { color: #4a5568 !important; }
.features-auto-grid .feature-line h4 { color: #2d3748 !important; }
.features-auto-grid .feature-line p { color: #4a5568 !important; }
.feature-card, .features-auto-grid .feature-line { opacity: 1 !important; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 5. NAVIGATION AND PAGE ROUTING
# =============================================================================
if 'page' not in st.session_state:
    st.session_state.page = "home"

def set_page(page_name):
    st.session_state.page = page_name

col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
with col1:
    st.markdown("<h1>🐄 BovineAI Analytics</h1>", unsafe_allow_html=True)
with col2:
    st.button("Home", on_click=set_page, args=("home",), use_container_width=True)
with col3:
    st.button("About", on_click=set_page, args=("about",), use_container_width=True)
with col4:
    st.button("Demo", on_click=set_page, args=("demo",), use_container_width=True)
with col5:
    st.button("Contact", on_click=set_page, args=("contact",), use_container_width=True)

st.markdown("---")

# =============================================================================
# 6. PAGE CONTENT
# =============================================================================
if st.session_state.page == "home":
    st.markdown("""
    <div class="main-header">
      <div style="position: relative; z-index: 1;">
        <h1 class="hero-title">World's Leading Bovine Detection AI</h1>
        <p class="hero-subtitle">
          Advanced machine learning technology to accurately identify and classify cow breeds with 
          state-of-the-art computer vision. Get instant results with confidence scores, detailed 
          breed information, and automated measurements for comprehensive livestock management.
        </p>
        <a href="#demo" class="cta-button">Get Started</a>
        <a href="#demo" class="secondary-button">Try Demo</a>
      </div>
    </div>

    <div class="stats-container">
      <div style="text-align: center; margin-bottom: 2rem;">
        <h2 style="margin: 0; font-size: 2rem;">Our Performance</h2>
        <p style="opacity: 0.8; margin-top: 0.5rem;">Trusted by farmers and livestock professionals worldwide</p>
      </div>
      <div class="stats-grid">
        <div class="stat-item">
          <span class="stat-number">92%</span>
          <span class="stat-label">Accuracy Rate</span>
        </div>
        <div class="stat-item">
          <span class="stat-number">40+</span>
          <span class="stat-label">Breeds Supported</span>
        </div>
        <div class="stat-item">
          <span class="stat-number">1k+</span>
          <span class="stat-label">Images Analyzed</span>
        </div>
        <div class="stat-item">
          <span class="stat-number">24/7</span>
          <span class="stat-label">Availability</span>
        </div>
      </div>
    </div>

    <h2 class="section-title">Our Featured Solutions</h2>
    <div class="features-grid">
      <div class="feature-card">
        <span class="feature-icon">🧠</span>
        <h3 class="feature-title">AI-Powered Detection</h3>
        <p class="feature-description">Advanced deep learning models trained on thousands of bovine images for accurate breed identification and health assessment</p>
      </div>
      <div class="feature-card">
        <span class="feature-icon">⚡</span>
        <h3 class="feature-title">Instant Results</h3>
        <p class="feature-description">Get breed predictions, health status, and automated measurements in seconds with detailed confidence scores</p>
      </div>
      <div class="feature-card">
        <span class="feature-icon">📱</span>
        <h3 class="feature-title">Comprehensive Analysis</h3>
        <p class="feature-description">Complete livestock analytics including breed classification, health monitoring, and biometric measurements</p>
      </div>
    </div>

    <h2 class="section-title">Technology Stack</h2>
    <div class="tech-grid">
      <div>
        <h3>Machine Learning & AI</h3>
        <div class="tech-badges">
          <span class="tech-badge">PyTorch</span>
          <span class="tech-badge">YOLOv8</span>
          <span class="tech-badge">MobileNetV2</span>
          <span class="tech-badge">OpenCV</span>
          <span class="tech-badge">Computer Vision</span>
          <span class="tech-badge">Deep Learning</span>
        </div>
      </div>
      <div>
        <h3>Deployment & Infrastructure</h3>
        <div class="tech-badges">
          <span class="tech-badge">Streamlit</span>
          <span class="tech-badge">Python</span>
          <span class="tech-badge">Image Processing</span>
          <span class="tech-badge">Real-time Analysis</span>
          <span class="tech-badge">Cloud Computing</span>
          <span class="tech-badge">API Integration</span>
        </div>
      </div>
    </div>

    <a id="demo"></a>
    """, unsafe_allow_html=True)

elif st.session_state.page == "demo":
    st.markdown('<h1 class="section-title">Try Our Advanced Bovine Analysis Tool</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div class="upload-area">
      <h3 style="margin-bottom: 1rem; color: #4a5568;">Upload Bovine Images</h3>
      <p style="color: #718096; margin-bottom: 2rem;">Drag and drop your images here or click to browse. Supports multiple images for batch analysis.</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader("Upload images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="image_uploader")

    if uploaded_files:
        if st.button('Analyze Images', key="image_button", use_container_width=True):
            for uploaded_file in uploaded_files:
                image = Image.open(uploaded_file).convert("RGB")
                
                with st.spinner(f'Analyzing {uploaded_file.name}...'):
                    p_type, p_breed, p_health, p_len_cm, p_hgt_cm, p_wgt, debug_img = analyze_image(image)
                    
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.image(image, caption=f'Uploaded: {uploaded_file.name}', use_column_width=True)
                        st.image(debug_img, caption=f'Model Detection Outline', use_column_width=True)
                    with col2:
                        st.success(f"AI Analysis for {uploaded_file.name}")
                        st.metric("Animal Type", p_type)
                        st.metric("Predicted Breed", p_breed)
                        st.metric("Visual Health Status", p_health)
                        
                        st.divider()
                        st.info("Automated Measurements & Estimations", icon="📏")
                        st.metric("Approx. Body Length", f"{p_len_cm:.2f} cm")
                        st.metric("Approx. Body Height", f"{p_hgt_cm:.2f} cm")
                        st.metric("Estimated Live Weight", f"~{p_wgt:.2f} kg")
                        
                        st.divider()
                        st.info("Breed Information Database", icon="ℹ️")
                        if p_breed in breed_info_db:
                            info = breed_info_db[p_breed]
                            st.write(f"Typical Milk Yield: {info['milk_yield']}")
                            st.write(f"Typical Weight Range: {info['weight_range']}")
                            st.caption(info['info'])
                        else:
                            st.warning("Breed info not in database.")
                st.divider()

elif st.session_state.page == "about":
    st.markdown('<h1 class="section-title">About BovineAI Analytics</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
      <h3 style="color: #2d3748; margin-bottom: 2rem;">Our Mission</h3>
      <p style="color: #4a5568; line-height: 1.8; font-size: 1.1rem; margin-bottom: 2rem;">
        We're revolutionizing livestock management through advanced AI technology. Our comprehensive bovine analysis system 
        combines breed detection, health monitoring, and automated biometric measurements to help farmers, veterinarians, 
        and livestock professionals make informed decisions about their cattle and buffalo.
      </p>

      <h3 style="color: #2d3748; margin-bottom: 2rem;">Key Features</h3>
      <div class="features-auto-grid">
        <div class="feature-line feature-blue">
          <h4>🧬 Breed Classification</h4>
          <p>Accurate identification of 40+ Indian and exotic bovine breeds using deep learning models.</p>
        </div>
        <div class="feature-line feature-green">
          <h4>🏥 Health Assessment</h4>
          <p>Visual health status detection to identify potential health issues early.</p>
        </div>
        <div class="feature-line feature-orange">
          <h4>📏 Automated Measurements</h4>
          <p>Non-invasive biometric measurements including length, height, and weight estimation.</p>
        </div>
        <div class="feature-line feature-purple">
          <h4>📊 Comprehensive Database</h4>
          <p>Extensive breed information including milk yields, weight ranges, and characteristics.</p>
        </div>
      </div>

      <h3 style="color: #2d3748; margin: 3rem 0 2rem 0;">Technical Approach</h3>
      <div class="panel panel-hero">
        <p style="margin: 0; line-height: 1.6;">
          Our system employs state-of-the-art computer vision techniques including MobileNetV2 for classification tasks, 
          YOLOv8 for object detection and segmentation, and custom neural networks for multi-output predictions. 
          The models are trained on extensive datasets of Indian and exotic bovine breeds, ensuring high accuracy 
          across diverse conditions and environments.
        </p>
      </div>
    </div>
    """, unsafe_allow_html=True)

elif st.session_state.page == "contact":
    st.markdown('<h1 class="section-title">Contact Us</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div class="contact-grid">
      <div class="card">
        <h3 style="color: #2d3748; margin-bottom: 2rem;">Get In Touch</h3>
        <p style="color: #4a5568; line-height: 1.6; margin-bottom: 2rem;">
          Have questions about our bovine detection AI? Want to integrate our technology 
          into your livestock management system? We'd love to hear from you!
        </p>

        <div style="margin: 2rem 0;">
          <div class="contact-row">
            <span class="contact-icon">📧</span>
            <div>
              <strong style="color: #2d3748;">Email:</strong><br />
              <span style="color: #4a5568;">bovineai@analytics.com</span>
            </div>
          </div>
          <div class="contact-row">
            <span class="contact-icon">📱</span>
            <div>
              <strong style="color: #2d3748;">Phone:</strong><br />
              <span style="color: #4a5568;">+91 (555) 123-4567</span>
            </div>
          </div>
          <div class="contact-row">
            <span class="contact-icon">📍</span>
            <div>
              <strong style="color: #2d3748;">Address:</strong><br />
              <span style="color: #4a5568;">Agricultural AI Research Center<br />Innovation Hub, Tech City</span>
            </div>
          </div>
        </div>
      </div>

      <div class="card">
        <h3 style="color: #2d3748; margin-bottom: 2rem;">Send us a message</h3>
        <form class="contact-form">
          <input type="text" placeholder="Your full name" />
          <input type="email" placeholder="Your email address" />
          <input type="text" placeholder="Subject" />
          <textarea placeholder="Tell us how we can help you..." rows="6"></textarea>
          <button type="submit" class="cta-button">Send Message</button>
        </form>
      </div>
    </div>

    <hr />
    <div class="footer-wrap">
      <div style="margin-bottom: 1rem;">
        <h3 style="color: #2d3748; margin-bottom: 1rem;">🐄 BovineAI Analytics</h3>
        <p style="margin: 0;">© 2024 BovineAI Analytics. All rights reserved.</p>
        <p style="margin: 0.5rem 0 0 0;">Powered by Advanced Machine Learning & Computer Vision</p>
      </div>
      <div style="margin-top: 1rem;">
        <span class="footer-badge">
          🚀 Making livestock management smarter, one prediction at a time
        </span>
      </div>
    </div>
    """, unsafe_allow_html=True)
