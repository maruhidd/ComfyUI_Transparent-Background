
import comfy.model_management
from comfy.sd import CLIP
from custom_nodes.AI_Assistant_ComfyUI.modules.draw_line.line_drawing import LineDrawingModule
from custom_nodes.AI_Assistant_ComfyUI.modules.prompt_analysis.prompt_analysis import PromptAnalysisModule
from .modules.canny.canny_image import canny_image
from nodes import CLIPTextEncode, ConditioningSetMask
from enum import Enum
import json
from typing import Literal, Tuple, TypedDict, NamedTuple
import sys
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL
from diffusers.utils import load_image
from PIL import Image
import torch
import numpy as np
import cv2
from PIL import Image
from typing import Optional
import re
from transparent_background import Remover
import torchvision.transforms as transforms
import torchvision

def tensor_to_pil(tensor):
    # [1, 1024, 1024, 3] -> [1024, 1024, 3] -> [3, 1024, 1024]
    tensor = tensor.squeeze(0).permute(2, 0, 1)
    p = torchvision.transforms.functional.to_pil_image(tensor)
    return p

def pil_to_tensor(pil_image):
    tensor = torchvision.transforms.functional.to_tensor(pil_image)
    # [1024, 1024, 3] -> [3, 1024, 1024] -> [1, 1024, 1024, 3]
    tensor = tensor.permute(1, 2, 0)
    tensor = tensor.unsqueeze(0)
    return tensor

    
class RemoveBackgroundNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "remove_background"

    def remove_background(self, image: torch.Tensor) -> Tuple[torch.Tensor]:
        """Remove background"""
        remover = Remover()
        pil_image = tensor_to_pil(image)
        out_pil = remover.process(pil_image)
        out_pil = pil_to_tensor(out_pil)
        #alpha部分のみ取り出してマスクを作成する
        mask = out_pil[:, :, 3]
        mask = mask.unsqueeze(2)
        mask = mask.expand(mask.shape[0], mask.shape[1], 3)
        mask = mask.float()
        return (out_pil, mask)


NODE_CLASS_MAPPINGS = {
    "RemoveBackgroundNode": RemoveBackgroundNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RemoveBackgroundNode": "Remove Background",
}
