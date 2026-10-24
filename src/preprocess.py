import cv2
import numpy as np
from PIL import Image


def preprocess_image(image: Image.Image) -> np.ndarray:
    rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    denoised = cv2.fastNlMeansDenoising(                #khử nhiễu 1 ít 
        gray,
        None,
        h=5,
        templateWindowSize=7,
        searchWindowSize=21,
    )

    return denoised