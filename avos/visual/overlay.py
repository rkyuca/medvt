import numpy as np
from PIL import Image


def PIL2array(img):
    return np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], 4)


def create_overlay(img, mask, colors):
    im = Image.fromarray(np.uint8(img))
    im = im.convert('RGBA')

    mask_color = np.zeros((mask.shape[0], mask.shape[1], 3))
    if len(colors) == 3:
        mask_color[mask == colors[1], 0] = 255
        mask_color[mask == colors[1], 1] = 255
        mask_color[mask == colors[2], 0] = 255
    else:
        mask_color[mask == colors[1], 2] = 255

    overlay = Image.fromarray(np.uint8(mask_color))
    overlay = overlay.convert('RGBA')

    im = Image.blend(im, overlay, 0.7)
    blended_arr = PIL2array(im)[:, :, :3]
    img2 = img.copy()
    img2[mask == colors[1], :] = blended_arr[mask == colors[1], :]
    return img2