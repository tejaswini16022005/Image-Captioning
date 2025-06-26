# Image-Captioning
A fun and interactive Streamlit app that turns your images into intelligent captions using cutting-edge AI models like BLIP (by Salesforce) and GIT (by Microsoft). This project showcases multimodal AI capabilities by combining computer vision and natural language generation.

---

## 🚀 Features

- 🧠 **BLIP Model** – Generates human-like captions using image-to-text generation.
- 🤖 **GIT Model** – Microsoft's generative image-to-text transformer.
- 🔁 **Comparison Mode** – Compare outputs from both models and choose your favorite.
- 🎨 **Stylish UI** – Built with Streamlit, includes playful prompts and feedback options.
- 📂 **Supports JPG, PNG, JPEG** – Just upload and caption!

---

## 🛠️ Technologies Used

- **Python**
- **Streamlit** – For building the interactive web UI
- **Transformers (Hugging Face)** – For model loading and inference
  - [`Salesforce/blip-image-captioning-base`](https://huggingface.co/Salesforce/blip-image-captioning-base)
  - [`microsoft/git-base`](https://huggingface.co/microsoft/git-base)
- **PyTorch** – Backend framework for model execution
- **Pillow (PIL)** – Image preprocessing

---

## 📷 How It Works

1. Upload an image (JPG, JPEG, PNG).
2. Select your preferred AI model: BLIP, GIT, or compare both.
3. The selected model processes the image and generates a natural language caption.
4. Optionally vote on which caption you liked more when in comparison mode.

---

## 💻 Setup Instructions

### 🔧 Install Dependencies

Make sure you have Python 3.8+ and pip installed.

```bash
pip install streamlit torch torchvision transformers pillow

