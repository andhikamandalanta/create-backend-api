from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import openai

app = FastAPI()

# Izinkan akses dari frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Sesuaikan jika dihosting
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup BLIP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)

# OpenAI API Key
openai.api_key = "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"

# Endpoint: Generate caption dengan BLIP
@app.post("/generate-caption")
async def generate_caption(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return {"caption": caption}

# Endpoint: Refinement dengan GPT-3.5
@app.post("/refine-caption")
async def refine_caption(caption: str = Form(...), style: str = Form(...)):
    prompt_styles = {
        "formal": "Refine the following image caption to match a formal writing style. Keep the key details and meaning intact while making the sentence more professional:",
        "informal": "Rewrite the following image caption in a relaxed and conversational style. Maintain the key details while making it sound more friendly:",
        "social": "Rewrite the following image caption in a casual and trendy social media style. Use Gen Z slang, humor, emojis, and viral expressions while keeping the message clear:",
        "ecommerce": "Refine the following image caption to match the persuasive and engaging style of e-commerce. Ensure it highlights product benefits, creates a sense of urgency, and uses informal but professional language suitable for live-selling platforms. Do NOT introduce new elements:"
    }

    prompt = f"{prompt_styles.get(style, '')}\n\n{caption}"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{ "role": "user", "content": prompt }]
    )

    refined = response["choices"][0]["message"]["content"]
    return {"refined_caption": refined}

# Endpoint: Translate ke Bahasa Indonesia
@app.post("/translate-caption")
async def translate_caption(caption: str = Form(...)):
    prompt = (
        "Translate the following English caption into natural, fluent, and contextually appropriate Indonesian. "
        "Preserve the original meaning, tone, and nuance. This is a caption for an image and may reflect informal, formal, or marketing styles:\n\n"
        f"{caption}"
    )

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{ "role": "user", "content": prompt }]
    )

    translated = response["choices"][0]["message"]["content"]
    return {"translated_caption": translated}
