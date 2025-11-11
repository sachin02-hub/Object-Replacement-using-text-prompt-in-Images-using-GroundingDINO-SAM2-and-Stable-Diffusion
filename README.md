# Object Replacement using text prompt in Images using GroundingDINO SAM2 and Stable-Diffusion

This project performs object detection, segmentation, and removal (or replacement) in both images and videos using state-of-the-art models — Grounding DINO, Segment Anything 2 (SAM2), and Stable Diffusion Inpainting.

It combines grounding-based object detection with precise segmentation and realistic background inpainting for seamless, temporally consistent results across video frames.

```bash
NOTE: The project was run on Google Colab with T4 GPU
```

## Features

- Text-based object detection using Grounding DINO

- Accurate segmentation using Segment Anything 2 (SAM2)

- Realistic inpainting with Stable Diffusion Inpainting Pipeline

- Video support with temporal consistency via optical flow blending

- Supports object replacement prompts (e.g., “replace dog with a German Shepherd”)

- Fully automated frame extraction, processing, and video reconstruction

- Modular, clean design — adaptable for different image editing or video AI pipelines

## Model Overview

| Component                    | Model Used                    | Purpose                                             |
| ---------------------------- | ----------------------------- | --------------------------------------------------- |
| **Object Detection**         | Grounding DINO                | Detects target objects from a **text prompt**       |
| **Segmentation**             | SAM2 (Segment Anything 2)     | Generates pixel-accurate masks for detected objects |
| **Inpainting / Replacement** | Stable Diffusion or Kandinsky | Removes or replaces the object region               |
| **Temporal Smoothing**       | Optical Flow                  | Ensures smooth transitions across video frames      |

## Example Use-Cases

- Remove unwanted objects (e.g., “Remove the bike and fill background”)

- Replace detected objects (e.g., “Replace the dog with a German Shepherd”)

- Create consistent AI-generated edits across video frames

- Use for video cleanup, VFX prep, or AI-assisted post-processing

## Project Architecture

```bash
Input Image
   │
   ├──> Grounding DINO → Object Detection
   │
   ├──> SAM2 → Segmentation Mask
   │
   ├──> Stable Diffusion / Kandinsky → Inpainting / Replacement
   │
   └──> Modified Image
```

In case of Video based replacement or edits:

```bash
Input Video
   │
   ├──> Frame Extraction (Supervision)
   │
   ├──> Grounding DINO → Object Detection
   │
   ├──> SAM2 → Segmentation Mask
   │
   ├──> Stable Diffusion / Kandinsky → Inpainting / Replacement
   │
   ├──> Optical Flow → Frame Consistency
   │
   └──> Reconstructed Output Video (MP4)
```

## Installation

```bash
# Clone this repository
git clone https://github.com/sachin02-hub/Object-Replacement-using-text-prompt-in-Images-using-GroundingDINO-SAM2-and-Stable-Diffusion.git
cd Object-Replacement-using-text-prompt-in-Images-using-GroundingDINO-SAM2-and-Stable-Diffusion

# Install dependencies
pip install -q torch torchvision torchaudio opencv-python numpy matplotlib supervision kornia diffusers==0.32.2 transformers==4.49 accelerate safetensors scikit-image ffmpeg-python scikit-video
```

## Install and Configure Submodules

```bash
# Grounding DINO
mkdir grounding && cd grounding
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO/groundingdino/models/GroundingDINO/csrc/MsDeformAttn
sed -i 's/value.type()/value.scalar_type()/g' ms_deform_attn_cuda.cu
sed -i 's/value.scalar_type().is_cuda()/value.is_cuda()/g' ms_deform_attn_cuda.cu
cd ../../..
pip install -e .
cd ../..

# Segment Anything 2
mkdir sam_2 && cd sam_2
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2
pip install -e . -q
python setup.py build_ext --inplace
cd ../..
```

## Download Model Weights

```bash
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
wget -q https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
wget -q https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
wget -q https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

## How It Works

- Load an input image into memory.

- Detect objects matching a text prompt with Grounding DINO.

- Generate segmentation masks for detected regions using SAM2.

- Inpaint or replace the object area using Stable Diffusion or Kandinsky.

- Save the final image.


## Results

### For Image based replacement:

Input Image

<img width="432" height="411" alt="image" src="https://github.com/user-attachments/assets/c82675d5-82dd-4773-91de-cc0312c2b720" />

Annotated image with bounding boxes and labels:

<img width="794" height="713" alt="image" src="https://github.com/user-attachments/assets/5ba74791-91ac-4443-a52c-2600087ef5cb" />

Segmented images with semi-transparent color overlays:

<img width="1104" height="483" alt="image" src="https://github.com/user-attachments/assets/5019ae1b-3b4c-46df-8f3d-1b15ec5d3ec0" />

Output Image Generated based on the Prompt "replace with Flower pot":

<img width="389" height="389" alt="image" src="https://github.com/user-attachments/assets/5714f339-5cdf-4019-b828-ce4fe186c7d4" />



## How It Works for Video based replacement

- Extract frames from the input video using `supervision`.

- Detect objects matching a text prompt with Grounding DINO.

- Generate segmentation masks for detected regions using SAM2.

- Inpaint or replace the object area using Stable Diffusion or Kandinsky.

- Use optical flow to blend consecutive frames for temporal consistency.

- Reconstruct the final video.

```bash
NOTE: Current Attempt in implementing Video based object modification does work but it has temporal inconsistency. The modified section of the video has artifacts across frames and is visible.
```

## Configuration

You can modify the following parameters in the script:

```python
TEXT_PROMPT = "bike"  # Object to remove or replace
REPLACEMENT_PROMPT = "Remove the bike and fill the background."
BOX_THRESHOLD = 0.3
TEXT_THRESHOLD = 0.25
SCALE_FACTOR = 1.0
NUM_FRAMES = 80  # Limit for testing
```

## To do

- Optimize the pipeline for faster results
- Solve the issue of Temporal inconsistency for Smooth transition between frames
