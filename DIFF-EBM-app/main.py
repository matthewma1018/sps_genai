import io, uvicorn, torch
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid

from helper_lib.utils import get_device, load_model
from helper_lib.ebm.model import get_model as get_ebm
from helper_lib.ebm.generator import generate_samples as ebm_generate_samples
from helper_lib.diffusion.model import get_model as get_diffusion
from helper_lib.diffusion.generator import generate_samples as diffusion_generate_samples

app = FastAPI(title="DIFFUSION-EBM-app")
DEVICE = get_device()
to_pil = ToPILImage()

ebm_model = load_model(get_ebm("EBM"), model_type="ebm", device=DEVICE, eval_mode=True)
diff_model = load_model(get_diffusion("DIFF"), model_type="diffusion", device=DEVICE, eval_mode=True)

def _batch_to_png(imgs):
    if imgs.size(0) == 1:
        pil = to_pil(imgs[0].cpu())
    else:
        nrow = int(imgs.size(0) ** 0.5) or 1
        grid = make_grid(imgs.cpu(), nrow=nrow, padding=2)
        pil = to_pil(grid)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

@app.get("/ebm/generate", responses={200: {"content": {"image/png": {}}}})
def ebm_generate(
    num_samples: int = Query(16, ge=1, le=64),
    steps: int = Query(200, ge=1, le=5000),
    step_size: float = Query(0.05, gt=0),
    noise_scale: float = Query(0.01, ge=0),
):
    imgs = ebm_generate_samples(ebm_model, DEVICE, num_samples, steps, step_size, noise_scale, True)
    return _batch_to_png(imgs)

@app.get("/diffusion/generate", responses={200: {"content": {"image/png": {}}}})
def diffusion_generate(
    num_samples: int = Query(16, ge=1, le=64),
    timesteps: int = Query(1000, ge=10, le=4000),
):
    imgs = diffusion_generate_samples(diff_model, DEVICE, num_samples, timesteps, (3, 32, 32))
    return _batch_to_png(imgs)

def main():
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()