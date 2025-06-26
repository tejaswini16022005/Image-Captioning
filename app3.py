import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, GitProcessor, GitForCausalLM
import torch
import random

# Set Streamlit page config
st.set_page_config(page_title="Lens to Language", layout="centered")

# Custom styles
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Raleway:wght@300&display=swap');
    .fun-title {
        font-size: 2.6em;
        font-weight: 300;
        color: #900b39;
        text-align: center;
        font-family: 'Raleway', sans-serif;
        margin-bottom: 0.5em;
    }
    .subtext {
        text-align: center;
        font-size: 1.1em;
        color: #555;
        margin-bottom: 30px;
    }
    .caption-fun-box {
        background-color: #f7c6d7;
        padding: 1.5rem;
        border-radius: 15px;
        font-size: 1.3em;
        font-weight: 500;
        border-left: 6px solid #7f364f;
        margin-top: 20px;
    }
    .footer {
        margin-top: 3rem;
        text-align: center;
        font-size: 0.9em;
        color: #aaa;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="fun-title">Lens to Language</div>', unsafe_allow_html=True)
st.markdown('<div class="subtext">Upload a pic and let our caption wizards do their magic! 🧙‍♂️✨</div>', unsafe_allow_html=True)

# Model selection
model_choice = st.selectbox(
    "🔍 Choose your captioning wizard:",
    ("BLIP (Salesforce)", "GIT (Microsoft)", "Compare Both")
)

# Load models
device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    return processor, model

@st.cache_resource
def load_git():
    processor = GitProcessor.from_pretrained("microsoft/git-base")
    model = GitForCausalLM.from_pretrained("microsoft/git-base").to(device)
    return processor, model

# Upload image
uploaded_file = st.file_uploader("📸 Drop your masterpiece here", type=["jpg", "jpeg", "png"])

# Caption generation
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption=random.choice(["🎨 Art in motion", "👀 What do we have here?", "Visual vibes"]), use_container_width=True)

    if model_choice in ["BLIP (Salesforce)", "Compare Both"]:
        with st.spinner("🔮 BLIP is thinking..."):
            blip_processor, blip_model = load_blip()
            inputs = blip_processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                output = blip_model.generate(**inputs)
            blip_caption = blip_processor.decode(output[0], skip_special_tokens=True)
            st.markdown(f'<div class="caption-fun-box">🧠 <strong>BLIP says:</strong><br>{blip_caption.capitalize()}</div>', unsafe_allow_html=True)

    if model_choice in ["GIT (Microsoft)", "Compare Both"]:
        with st.spinner("🔍 GIT is analyzing..."):
            git_processor, git_model = load_git()
            pixel_values = git_processor(images=image, return_tensors="pt").pixel_values.to(device)
            with torch.no_grad():
                generated_ids = git_model.generate(pixel_values=pixel_values, max_length=50)
            git_caption = git_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            st.markdown(f'<div class="caption-fun-box">🤖 <strong>GIT says:</strong><br>{git_caption.capitalize()}</div>', unsafe_allow_html=True)

# Feedback
if model_choice == "Compare Both" and uploaded_file:
    st.subheader("🗳️ Which caption did you prefer?")
    feedback = st.radio("Your pick:", ["BLIP", "GIT", "Both were great!", "Neither impressed me"])
    if feedback:
        st.success(f"Thanks for your feedback! You chose: **{feedback}**")

# Footer
st.markdown("""
<div class="footer">
    Built with <span style='color:#f7c6d7;'> ♥ </span> by your friendly neighborhood AI
</div>
""", unsafe_allow_html=True)
