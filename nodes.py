
from typing import Tuple
import torch
from transparent_background import Remover
import torchvision
import numpy as np
from PIL import Image

def tensor_to_pil(tensor):
    # [1, h, w, ch] -> [h, w, ch] -> [ch, h, w]
    __tensor = tensor.squeeze(0).permute(2, 0, 1)
    p = torchvision.transforms.functional.to_pil_image(__tensor)
    return p

def pil_to_tensor(pil_image):
    __tensor = torchvision.transforms.functional.to_tensor(pil_image)
    # [h, w, ch] -> [ch, h, w] -> [1, h, w, ch]
    __tensor = __tensor.permute(1, 2, 0)
    __tensor = __tensor.unsqueeze(0)
    return __tensor

    
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

    def remove_background(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Remove background"""
        remover = Remover()
        pil_image = tensor_to_pil(image)
        out_pil = remover.process(pil_image)
        out_tensor = pil_to_tensor(out_pil)
        # Extract only the alpha part to create a mask [1, h, w, ch] -> [1, h, w]
        mask = out_tensor[:, :, :, 3]
        return (out_tensor, mask,)

class FillTransparentNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "color_red": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1, "display": "color"}),
                "color_green": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1, "display": "color"}),
                "color_blue": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1, "display": "color"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "fill_transparent"

    def fill_transparent(self, image: torch.Tensor, color_red: int, color_green: int, color_blue: int) -> Tuple[torch.Tensor]:
        """Fill transparent"""
        # image [1, h, w, ch]
        # mask [1, h, w]
        image_pil = tensor_to_pil(image)
        color_back = Image.new("RGB", image_pil.size, (color_red, color_green, color_blue))
        # paste
        color_back.paste(image_pil, (0, 0), image_pil)
        filled_image = pil_to_tensor(color_back)
        return (filled_image,)




NODE_CLASS_MAPPINGS = {
    "RemoveBackgroundNode": RemoveBackgroundNode,
    "FillTransparentNode": FillTransparentNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RemoveBackgroundNode": "Remove Background",
    "FillTransparentNode": "Fill Transparent",
}
