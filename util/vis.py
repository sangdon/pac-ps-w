import os, sys
from PIL import Image, ImageDraw

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from torchvision import transforms


def plot_bbox(img, bbox_xyxy, fn=None):
    img_pil = transforms.ToPILImage()(img)
            
    draw = ImageDraw.Draw(img_pil)
    draw.rectangle(bbox_xyxy.tolist(), outline="white", width=2)

    if fn is not None:
        img_pil.save(fn)


def plot_multiple_bbox(img, bbox_xyxy_list, color_list, fn=None):
    img_pil = transforms.ToPILImage()(img)
            
    draw = ImageDraw.Draw(img_pil)
    for bbox_xyxy, color in zip(bbox_xyxy_list, color_list):
        draw.rectangle(bbox_xyxy.tolist(), outline=color, width=2)

    if fn is not None:
        img_pil.save(fn)
