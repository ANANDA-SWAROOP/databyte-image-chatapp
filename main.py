#this is the final code in anand.py


from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    BlipForQuestionAnswering
)
from diffusers import StableDiffusionPipeline
import io
import base64
import httpx
import asyncio

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load local image captioning and VQA models
caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

vqa_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
vqa_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device)

# Hugging Face API configuration for chat (using model from the second code)
HUGGINGFACE_API_TOKEN = ""
CHAT_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
HUGGINGFACE_MISTRAL_API_URL = f"https://api-inference.huggingface.co/models/{CHAT_MODEL}"

# Load Stable Diffusion for image generation
if device == "cuda":
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    )
else:
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe = pipe.to(device)

async def get_mistral_response(prompt: str, max_length: int = 512) -> str:
    """
    Asynchronously query the chat model via the Hugging Face API.
    The payload now includes max_new_tokens and return_full_text settings,
    and the returned text is split on "</s>" if present.
    """
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_length,
            "temperature": 0.7,
            "do_sample": True,
            "return_full_text": False
        }
    }
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(HUGGINGFACE_MISTRAL_API_URL, headers=headers, json=payload)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Error in Hugging Face API request")
        data = response.json()
        if isinstance(data, list) and data and "generated_text" in data[0]:
            text = data[0]["generated_text"]
            text = text.split("</s>")[0].strip()
            return text
        else:
            raise HTTPException(status_code=500, detail="Unexpected response format from Hugging Face API")
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    BlipForQuestionAnswering
)
from diffusers import StableDiffusionPipeline
import io
import base64
import httpx
import asyncio

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load local image captioning and VQA models
caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

vqa_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
vqa_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device)

# Hugging Face API configuration for chat (using model from the second code)
HUGGINGFACE_API_TOKEN = ""
CHAT_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
HUGGINGFACE_MISTRAL_API_URL = f"https://api-inference.huggingface.co/models/{CHAT_MODEL}"

# Load Stable Diffusion for image generation
if device == "cuda":
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    )
else:
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe = pipe.to(device)

async def get_mistral_response(prompt: str, max_length: int = 512) -> str:
    """
    Asynchronously query the chat model via the Hugging Face API.
    The payload now includes max_new_tokens and return_full_text settings,
    and the returned text is split on "</s>" if present.
    """
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_length,
            "temperature": 0.3,
            "do_sample": True,
            "return_full_text": False
        }
    }
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(HUGGINGFACE_MISTRAL_API_URL, headers=headers, json=payload)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Error in Hugging Face API request")
        data = response.json()
        if isinstance(data, list) and data and "generated_text" in data[0]:
            text = data[0]["generated_text"]
            text = text.split("</s>")[0].strip()
            return text
        else:
            raise HTTPException(status_code=500, detail="Unexpected response format from Hugging Face API")

async def process_text(prompt: str):
    """
    Process a text prompt:
      - If the prompt starts with "generate image:" the Stable Diffusion pipeline is used.
      - Otherwise, the prompt is wrapped in the expected formatting and sent to the chat model.
    """
    try:
        if prompt.lower().startswith("generate image:"):
            image_prompt = prompt[len("generate image:"):].strip()
            image = pipe(image_prompt).images[0]
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            encoded_image = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
            return {"text": f"Generated image: {image_prompt}", "image": encoded_image}
        else:
            formatted_prompt = f"<s>[INST] {prompt} [/INST]"
            mistral_answer = await get_mistral_response(formatted_prompt)
            return {"text": mistral_answer, "image": None}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def process_image(image: Image.Image, prompt: str = None):
    """
    Process an image:
      - Generate an image caption.
      - If a text prompt is provided, perform Visual Q&A.
      - Pass both the VQA answer (if available) and the caption to the Mistral API for detailed analysis.
      - Return the results in a clearly formatted, numbered list with:
            1. Visual Q&A Answer (if prompt exists)
            2. Caption
            3. Detailed Analysis (Mistral explanation)
    """
    try:
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Generate image caption
        caption_inputs = caption_processor(image, return_tensors="pt")
        caption_inputs = {k: v.to(device) for k, v in caption_inputs.items()}
        caption_outputs = caption_model.generate(**caption_inputs)
        caption = caption_processor.decode(caption_outputs[0], skip_special_tokens=True)
        
        # Generate Visual Q&A answer if a prompt is provided
        vqa_answer = None
        if prompt:
            vqa_inputs = vqa_processor(image, prompt, return_tensors="pt")
            vqa_inputs = {k: v.to(device) for k, v in vqa_inputs.items()}
            vqa_outputs = vqa_model.generate(**vqa_inputs)
            vqa_answer = vqa_processor.decode(vqa_outputs[0], skip_special_tokens=True)
        
        # Build a combined input for Mistral using both VQA answer (if available) and the caption
        if vqa_answer:
            combined_input = f"Visual Q&A Answer: '{vqa_answer}'. Caption: '{caption}'."
        else:
            combined_input = f"Caption: '{caption}'."
        
        # Generate detailed analysis using the Mistral API
        mistral_prompt = (
            f"<s>[INST] Analyze the following combined image description and provide a detailed analysis. "
            f"Discuss the usage, environment, type of material, applications and impact in its industry in clear points: "
            f"{combined_input} [/INST]"
        )
        additional_info = await get_mistral_response(mistral_prompt)
        
        # Build the output with headings and numbered points in the desired order
        result_lines = []
        result_lines.append("**Image Analysis Results:**\n")
        if vqa_answer:
            result_lines.append("1. **Visual Q&A Answer:**")
            result_lines.append(f"   - {vqa_answer}\n")
            result_lines.append("2. **Caption:**")
            result_lines.append(f"   - {caption}\n")
            result_lines.append("3. **Detailed Analysis:**")
            result_lines.append(f"   - {additional_info}\n")
        else:
            result_lines.append("1. **Caption:**")
            result_lines.append(f"   - {caption}\n")
            result_lines.append("2. **Detailed Analysis:**")
            result_lines.append(f"   - {additional_info}\n")
        
        combined_text = "\n".join(result_lines)
        return {"text": combined_text, "image": None}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
async def chat(
    prompt: str = Form(None),
    images: list[UploadFile] = File(None),
):
    """
    Main chat endpoint.
      - For text-only requests:
           * "generate image:" triggers image generation.
           * Otherwise, the text is processed via the chat model.
      - For image-based requests:
           * The image is captioned and analyzed.
           * If a prompt is provided, Visual Q&A is also performed.
           * All results are combined and sent back in a single response.
    """
    try:
        if not images:
            if not prompt:
                raise HTTPException(status_code=400, detail="Text prompt required")
            return await process_text(prompt)
        
        results = []
        for image in images:
            image_data = await image.read()
            try:
                image_pil = Image.open(io.BytesIO(image_data))
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid image file")
            result = await process_image(image_pil, prompt)
            results.append(result)
        
        combined_text = " ".join([res["text"] for res in results])
        return {"text": combined_text, "image": None}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Vision ChatBot API is running!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


async def process_image(image: Image.Image, prompt: str = None):
    """
    Process an image:
      - Always generate an image caption.
      - Use the caption to query the Mistral API for a detailed analysis.
      - If a text prompt is provided, also perform Visual Q&A using BLIP-VQA.
      - Return the combined results to the frontend.
    """
    try:
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Generate image caption
        caption_inputs = caption_processor(image, return_tensors="pt")
        caption_inputs = {k: v.to(device) for k, v in caption_inputs.items()}
        caption_outputs = caption_model.generate(**caption_inputs)
        caption = caption_processor.decode(caption_outputs[0], skip_special_tokens=True)
        
        # Generate detailed analysis based on the caption using the Mistral API
        mistral_prompt = (
            f"<s>[INST] Analyze the following image caption and provide a detailed analysis: '{caption}'. "
            "Discuss the scene, colors, atmosphere, and possible interpretations in a concise manner. [/INST]"
        )
        additional_info = await get_mistral_response(mistral_prompt)
        
        # Combine caption and analysis
        combined_text = f"**Caption**: {caption}\n\n**Detailed Analysis**: {additional_info}"
        
        # If a prompt is provided, perform Visual Q&A and append its answer
        if prompt:
            vqa_inputs = vqa_processor(image, prompt, return_tensors="pt")
            vqa_inputs = {k: v.to(device) for k, v in vqa_inputs.items()}
            vqa_outputs = vqa_model.generate(**vqa_inputs)
            vqa_answer = vqa_processor.decode(vqa_outputs[0], skip_special_tokens=True)
            combined_text += f"\n\n**Visual Q&A Answer**: {vqa_answer}"
        
        return {"text": combined_text, "image": None}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat(
    prompt: str = Form(None),
    images: list[UploadFile] = File(None),
):
    """
    Main chat endpoint.
      - For text-only requests:
           * "generate image:" triggers image generation.
           * Otherwise, the text is processed via the chat model.
      - For image-based requests:
           * The image is captioned and analyzed.
           * If a prompt is provided, Visual Q&A is also performed.
           * All results are combined and sent back in a single response.
    """
    try:
        if not images:
            if not prompt:
                raise HTTPException(status_code=400, detail="Text prompt required")
            return await process_text(prompt)
        
        results = []
        for image in images:
            image_data = await image.read()
            try:
                image_pil = Image.open(io.BytesIO(image_data))
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid image file")
            result = await process_image(image_pil, prompt)
            results.append(result)
        
        combined_text = " ".join([res["text"] for res in results])
        return {"text": combined_text, "image": None}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Vision ChatBot API is running!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
