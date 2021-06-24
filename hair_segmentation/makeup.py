import cv2
import os
import numpy as np
from skimage.filters import gaussian
from test import evaluate, vis_parsing_maps
import argparse
from skimage.transform import resize


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--img-path', default='imgs/116.jpg')
    parse.add_argument('--out-dir', default='results/')
    parse.add_argument('--model-path', default='cp/79999_iter.pth')
    return parse.parse_args()


def sharpen(img):
    img = img * 1.0
    gauss_out = gaussian(img, sigma=10, multichannel=True)

    alpha = 1.5
    img_out = (img - gauss_out) * alpha + img

    img_out = img_out / 255.0

    mask_1 = img_out < 0
    mask_2 = img_out > 1

    img_out = img_out * (1 - mask_1)
    img_out = img_out * (1 - mask_2) + mask_2
    img_out = np.clip(img_out, 0, 1)
    img_out = img_out * 255
    return np.array(img_out, dtype=np.uint8)


def hair(image, parsing, part=17, color=[255, 255, 255], save_im=False, save_path='results/hair.jpg'):
    b, g, r = color
    tar_color = np.zeros_like(image)
    tar_color[:, :, 0] = b
    tar_color[:, :, 1] = g
    tar_color[:, :, 2] = r

    # image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    tar_hsv = cv2.cvtColor(tar_color, cv2.COLOR_BGR2HSV)

    # image_hsv[:, :, 0:2] = tar_hsv[:, :, 0:2]

    changed = cv2.cvtColor(tar_hsv, cv2.COLOR_HSV2BGR)

    # if part == 17:
        # changed = sharpen(changed)

    changed[parsing != part] = [[0,0,0]]
    
    if save_im:
        cv2.imwrite(save_path, changed)
    
    return changed


if __name__ == '__main__':
    args = parse_args()

    image_path = args.img_path
    cp = args.model_path

    image = cv2.imread(image_path)
    name = os.path.splitext(os.path.basename(image_path))[0]
    out_path_hair = os.path.join(args.out_dir, name + '_hair.jpg')
    out_path_face = os.path.join(args.out_dir, name + '_face.jpg')
    
    ori = image.copy()
    parsing = evaluate(image_path, cp)
    parsing = cv2.resize(parsing, image.shape[0:2][::-1], interpolation=cv2.INTER_NEAREST)

    face = vis_parsing_maps(image, parsing, stride=1, save_im=True, save_path=out_path_face)
    hair_mask = hair(image, parsing, save_im=True, save_path=out_path_hair)
    
        