import torch
import io
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from torchvision.transforms import ToPILImage

from helper_lib.model import Generator
from helper_lib.utils import get_device

app = FastAPI(title="MNIST GAN Image Generator API")

Z_DIM = 100

DEVICE = get_device()

MODEL_PATH = "gan_generator.pth"

model = Generator(z_dim=Z_DIM)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

model.to(DEVICE)
model.eval()

@app.get("/generate", responses={200: {"content": {"image/png": {}}}})
async def generate_image():
    """
    Generates a new handwritten digit image using the GAN.
    """
    try:
        # Create a random noise vector on the correct device
        noise = torch.randn(1, Z_DIM, device=DEVICE)

        # Generate the fake image (no gradients needed)
        with torch.no_grad():
            generated_tensor = model(noise)

        # Post-process the tensor: the tensor is [1, 1, 28, 28] with values from -1 to 1.
        # Un-normalize it from [-1, 1] back to [0, 1]
        generated_tensor = (generated_tensor * 0.5) + 0.5

        # Remove the batch dimension (dim 0)
        generated_tensor = generated_tensor.squeeze(0)

        # Convert the [1, 28, 28] tensor to a PIL Image
        # ToPILImage() handles tensors correctly
        pil_image = ToPILImage()(generated_tensor.cpu())

        # Save the PIL Image to an in-memory byte buffer
        img_buffer = io.BytesIO()
        pil_image.save(img_buffer, format="PNG")
        img_buffer.seek(0)  # Rewind the buffer to the beginning

        # Return the image as a streaming response
        return StreamingResponse(img_buffer, media_type="image/png")

    except Exception as e:
        return {"error": str(e)}, 500

# if __name__ == "__main__":
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)