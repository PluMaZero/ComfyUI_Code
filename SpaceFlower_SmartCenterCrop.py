import hashlib
import os
import json
import comfy
# import random
import random
import sys
# from tkinter import messagebox
import cv2
import torch
import numpy as np
import nodes
import folder_paths
from PIL import Image, ImageGrab
from .terminalcolors import tcolor, color_text
from server import PromptServer
from aiohttp import web
# import cv2
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw, ImageOps
from PIL.PngImagePlugin import PngInfo
from comfy.cli_args import args
# import subprocess
# from torchvision import transforms
from nodes import SaveImage
import time
from tkinter import filedialog
from .SpaceFlower_SaveImage import SpaceFlower_SaveImage
from comfy.model_management import InterruptProcessingException
import torchvision.transforms as transforms
from nodes import MAX_RESOLUTION
import torch.nn.functional as F
import shutil
try:
    import torchvision.transforms.v2 as T
except ImportError:
    import torchvision.transforms as T
    
from comfy.utils import ProgressBar, common_upscale

#32 스마트 객체 영역 자르기기
class SpaceFlower_SmartCenterCrop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 1536, "min": 64, "max": 8192, "step": 8}),
                # 감지할 배경색 (이 색이 아닌 부분을 캐릭터로 인식)
                "detect_bg_color": (["white", "black", "gray"], {"default": "white"}),
                # 여백이 생길 경우 채울 색상
                "pad_color": (["white", "black", "gray"], {"default": "white"}),
                # 캐릭터 인식 민감도 (0.01 ~ 1.0, 낮을수록 미세한 색차이도 캐릭터로 인식)
                "tolerance": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("cropped_image",)
    FUNCTION = "smart_crop"
    CATEGORY = "🌻SpaceFlower/Image"

    def smart_crop(self, image, width, height, detect_bg_color, pad_color, tolerance):
        # 결과 이미지를 담을 리스트
        out_images = []

        # 배치 내의 각 이미지에 대해 반복 처리
        for img in image:
            # 1. 텐서를 numpy로 변환 (H, W, C)
            np_img = img.cpu().numpy()
            h, w, c = np_img.shape

            # 2. 배경색 기준값 설정 (RGB)
            if detect_bg_color == "white":
                target_bg = np.array([1.0, 1.0, 1.0])
            elif detect_bg_color == "black":
                target_bg = np.array([0.0, 0.0, 0.0])
            else: # gray
                target_bg = np.array([0.5, 0.5, 0.5])

            # 3. 캐릭터 영역 감지 (배경색과 차이가 tolerance보다 큰 픽셀 찾기)
            # 픽셀별로 배경색과의 거리를 계산
            diff = np.abs(np_img - target_bg)
            # RGB 채널 중 하나라도 차이가 크면 객체로 간주
            mask = np.any(diff > tolerance, axis=-1)

            # 객체 픽셀의 좌표 찾기
            coords = np.argwhere(mask)

            if len(coords) > 0:
                # Bounding Box 계산 (y_min, x_min, y_max, x_max)
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)
                
                # 캐릭터의 중심점 계산
                center_y = (y_min + y_max) // 2
                center_x = (x_min + x_max) // 2
            else:
                # 캐릭터가 감지되지 않으면 이미지 정중앙을 기준으로 함
                center_y = h // 2
                center_x = w // 2

            # 4. 크롭할 영역 계산 (Target Box)
            # 중심점을 기준으로 width, height 크기의 박스 좌표 계산
            crop_y_min = center_y - (height // 2)
            crop_x_min = center_x - (width // 2)
            crop_y_max = crop_y_min + height
            crop_x_max = crop_x_min + width

            # 5. 결과 캔버스 생성 (지정된 pad_color로 채움)
            pad_val = 0.0
            if pad_color == "white": pad_val = 1.0
            elif pad_color == "gray": pad_val = 0.5
            
            # (H, W, C) 크기의 캔버스 생성
            canvas = np.full((height, width, c), pad_val, dtype=np.float32)

            # 6. 원본 이미지와 크롭 영역의 교차 구간(Intersection) 계산
            # 원본 이미지 내에서 유효한 좌표 범위
            src_x1 = max(0, crop_x_min)
            src_y1 = max(0, crop_y_min)
            src_x2 = min(w, crop_x_max)
            src_y2 = min(h, crop_y_max)

            # 캔버스 내에서 복사될 위치 좌표 범위
            dst_x1 = max(0, -crop_x_min)
            dst_y1 = max(0, -crop_y_min)
            # 복사할 너비와 높이
            copy_w = src_x2 - src_x1
            copy_h = src_y2 - src_y1

            # 유효한 영역이 있을 경우에만 복사
            if copy_w > 0 and copy_h > 0:
                canvas[dst_y1:dst_y1+copy_h, dst_x1:dst_x1+copy_w, :] = \
                    np_img[src_y1:src_y1+copy_h, src_x1:src_x1+copy_w, :]

            # 결과 리스트에 추가 (Tensor로 변환)
            out_images.append(torch.from_numpy(canvas))

        # 리스트를 스택하여 배치 텐서로 반환 (B, H, W, C)
        return (torch.stack(out_images),)