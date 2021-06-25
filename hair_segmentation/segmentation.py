import cv2
import os
import glob
import os.path as osp
import numpy as np
import argparse

from PIL import Image
from skimage.filters import gaussian
from skimage.transform import resize

import torch
import torchvision.transforms as transforms

try:
    from model import BiSeNet
except ImportError:
    from .model import BiSeNet
    
class FaceParsing:
    
    def __init__(self, model_path='hair_segmentation/cp/79999_iter.pth'):
        self.n_classes = 19
        self.net = BiSeNet(n_classes=self.n_classes)
        self.net.cuda()
        self.net.load_state_dict(torch.load(model_path))
        self.net.eval()
        
    def get_parsing(self, image):
        to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        with torch.no_grad():
            image = image.resize((512, 512), Image.BILINEAR)
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            out = self.net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            return parsing
        
def get_hair_mask(image, parsing, part=17, color=[255, 255, 255], save_im=False, save_path='results/hair.jpg'):
    b, g, r = color
    tar_color = np.zeros_like(image)
    tar_color[:, :, 0] = b
    tar_color[:, :, 1] = g
    tar_color[:, :, 2] = r

    # image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    tar_hsv = cv2.cvtColor(tar_color, cv2.COLOR_BGR2HSV)

    # image_hsv[:, :, 0:2] = tar_hsv[:, :, 0:2]

    changed = cv2.cvtColor(tar_hsv, cv2.COLOR_HSV2BGR)
    changed[parsing != part] = [[0,0,0]]

    if save_im:
        cv2.imwrite(save_path, cv2.resize(changed, (512, 512)))

    return changed

def get_face_segmentation(image, parsing, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(image)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = vis_im[index[0], index[1], :]
        # vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = vis_parsing_anno_color 
    #cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    # Save result or not
    if save_im:
        # cv2.imwrite(save_path[:-4] +'.png', vis_parsing_anno)
        cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    return vis_parsing_anno
    # return vis_im


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--in-dir', default='imgs/')
    parse.add_argument('--out-dir', default='results/')
    parse.add_argument('--model-path', default='cp/79999_iter.pth')
    return parse.parse_args()


if __name__ == '__main__':
    args = parse_args()

    save_path = args.out_dir
    os.makedirs(save_path, exist_ok=True)
    
    files = glob.glob(os.path.join(args.in_dir, "*.png")) + glob.glob(os.path.join(args.in_dir, "*.jpg")) + glob.glob(os.path.join(args.in_dir, "*.jpeg")) + glob.glob(os.path.join(args.in_dir, "*.bmp"))
    
    fp = FaceParsing(args.model_path)

    for f in files:
        img_name = f.split("/")[-1]

        image = cv2.imread(f)
        name = os.path.splitext(os.path.basename(img_name))[0]
        out_path_hair = os.path.join(save_path, name + '_hair.jpg')
        out_path_face = os.path.join(save_path, name + '_face.jpg')

        img = Image.open(f)
        parsing = fp.get_parsing(img)
        parsing = cv2.resize(parsing, image.shape[0:2][::-1], interpolation=cv2.INTER_NEAREST)
        face = get_face_segmentation(image, parsing, stride=1, save_im=True, save_path=out_path_face)
        hair_mask = get_hair_mask(image, parsing, save_im=True, save_path=out_path_hair)
    