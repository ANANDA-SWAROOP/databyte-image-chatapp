# databyte-image-chatapp
a modular architecture with 3 ai models communicating with each other

# Vision ChatBot API

## Overview
Vision ChatBot API is a FastAPI-based application that integrates multiple AI models to process text and images. It can generate image captions, answer visual questions about images, and provide detailed analyses using Mistral AI. The API also supports text-based chat interactions and Stable Diffusion-based image generation.

## Features
- **Image Captioning:** Uses BLIP to generate captions for uploaded images.
- **Visual Question Answering (VQA):** Answers questions related to uploaded images.
- **Text-Based Chat:** Utilizes Mistral AI for intelligent responses.
- **Image Generation:** Uses Stable Diffusion to generate images from text prompts.
- **CORS Support:** Allows cross-origin requests for easy frontend integration.

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- pip
- CUDA (for GPU acceleration, optional)

### Clone the Repository
```sh
 git clone <repository-url>
 cd <repository-folder>
```

### Install Dependencies
```sh
pip install -r requirements.txt
```

## Usage

### Running the API
To start the FastAPI server, run:
```sh
uvicorn anand:app --host 0.0.0.0 --port 8000
```

The API will be accessible at `http://localhost:8000`.

## API Endpoints

### Root Endpoint
- **GET /**
  - Returns a message indicating that the API is running.

### Chat Endpoint
- **POST /api/chat**
  - Accepts a text prompt or an image (or both) and returns:
    - A text response from Mistral AI.
    - A caption or analysis for an image.

#### Request Parameters:
- **prompt**: Text input (optional, required for text-based chat).
- **images**: Uploaded image files (optional, required for image processing).

#### Response Format:
```json
{
  "text": "Generated response or analysis",
  "image": "Base64-encoded image (if applicable)"
}
```

## Environment Variables
To use the Mistral API, set up the following environment variable:
```sh
export HUGGINGFACE_API_TOKEN='your-api-key'
```

## Dependencies
- `fastapi`
- `torch`
- `transformers`
- `diffusers`
- `PIL`
- `httpx`
- `uvicorn`

## Deployment
To deploy the API using Docker:
1. Create a `Dockerfile`:
```dockerfile
FROM python:3.9
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
CMD ["uvicorn", "anand:app", "--host", "0.0.0.0", "--port", "8000"]
```
2. Build and run the container:
```sh
docker build -t vision-chatbot .
docker run -p 8000:8000 vision-chatbot
```

## License
This project is licensed under the MIT License.

## Author
Anand

