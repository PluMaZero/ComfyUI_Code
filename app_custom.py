import warnings
warnings.filterwarnings("ignore")

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import cv2
import torch
import imageio
from PIL import Image

from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.renderers import EnvMap
from trellis2.utils import render_utils
import o_voxel

# =========================
# Config
# =========================
IMAGE_PATH = "image/anime_girl.png"
HDRI_PATH  = "assets/hdri/forest.exr"
MODEL_NAME = "microsoft/TRELLIS.2-4B"

OUT_MP4 = "sample.mp4"
OUT_GLB = "sample.glb"

# =========================
# Load HDRI EnvMap
# =========================
print("Loading HDRI...")
hdri = cv2.imread(HDRI_PATH, cv2.IMREAD_UNCHANGED)
hdri = cv2.cvtColor(hdri, cv2.COLOR_BGR2RGB)

envmap = EnvMap(
    torch.tensor(hdri, dtype=torch.float32, device="cuda")
)

# =========================
# Load Pipeline (정답 방식)
# =========================
print("Loading TRELLIS.2 pipeline...")
pipeline = Trellis2ImageTo3DPipeline.from_pretrained(
    MODEL_NAME
)
pipeline.cuda()   # eval(), half() ❌

# =========================
# Run Image → 3D
# =========================
print("Running image to 3D...")
image = Image.open(IMAGE_PATH).convert("RGB")
image.thumbnail((512, 512), Image.BICUBIC)

with torch.no_grad():
    mesh = pipeline.run(image)[0]

# nvdiffrast limit
# mesh.simplify(16_777_216)
mesh.simplify(300_000)

# =========================
# Render Video    // 비디오 되도록 생성 안해야 작동됨
# =========================
# print("Rendering video...")
# frames = render_utils.make_pbr_vis_frames(
#     render_utils.render_video(
#         mesh,
#         envmap=envmap,
#         resolution=384,   # 🔽 해상도도 줄이기
#         num_frames=20     # 🔽 120 → 30
#     )
# )

# imageio.mimsave(OUT_MP4, frames, fps=15)

torch.cuda.empty_cache()
torch.cuda.synchronize()

# =========================
# Export GLB
# =========================
print("Exporting GLB...")
glb = o_voxel.postprocess.to_glb(
    vertices          = mesh.vertices,
    faces             = mesh.faces,
    attr_volume       = mesh.attrs,
    coords            = mesh.coords,
    attr_layout       = mesh.layout,
    voxel_size        = mesh.voxel_size,
    aabb              = [[-0.5]*3, [0.5]*3],
    decimation_target = 50_000,
    texture_size      = 512,
    remesh            = False,
    remesh_band       = 1,
    remesh_project    = 0,
    verbose           = False
)

glb.export(OUT_GLB, extension_webp=True)

print("✅ DONE")
print(f" - Video : {OUT_MP4}")
print(f" - GLB   : {OUT_GLB}")
