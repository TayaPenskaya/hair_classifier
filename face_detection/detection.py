from __future__ import print_function
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import cv2
import math
import glob
import time
import numpy as np
from PIL import Image
from collections import namedtuple, OrderedDict
from torchvision.ops import nms as torch_nms

try:
    from model_search import Network
    from dataset.config import widerface_640 as cfg
except ImportError:
    from .model_search import Network
    from .dataset.config import widerface_640 as cfg

torch.set_default_tensor_type("torch.cuda.FloatTensor")


class TestBaseTransform:
    def __init__(self, mean):
        #self.size = size
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        return self.test_base_transform(image), boxes, labels

    def test_base_transform(self, image):
        #x = cv2.resize(image, (size, size)).astype(np.float32)
        x = image.astype(np.float32)
        x -= self.mean
        x = x.astype(np.float32)
        return x

    
class FaceDetection:
    
    def __init__(self, model_path="./weights/dsfdv2_r18.pth"):
        FPN_Genotype = namedtuple("FPN_Genotype", "Inter_Layer Out_Layer")
        AutoFPN = FPN_Genotype(
            Inter_Layer=[
                [("sep_conv_3x3", 1), ("conv_1x1", 0)],
                [("sep_conv_3x3", 2), ("sep_conv_3x3", 0), ("conv_1x1", 1)],
                [("sep_conv_3x3", 3), ("sep_conv_3x3", 1), ("conv_1x1", 2)],
                [("sep_conv_3x3", 4), ("sep_conv_3x3", 2), ("conv_1x1", 3)],
                [("sep_conv_3x3", 5), ("sep_conv_3x3", 3), ("conv_1x1", 4)],
                [("sep_conv_3x3", 4), ("conv_1x1", 5)],
            ],
            Out_Layer=[],
        )
        self.net = Network(
            C=64,
            criterion=None,
            num_classes=2,
            layers=1,
            phase="test",
            search=False,
            args=args,
            searched_fpn_genotype=AutoFPN,
            searched_cpm_genotype=None,
            fpn_layers=1,
            cpm_layers=1,
            auxiliary_loss=False,
        )
        
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
        print("Pretrained model loading OK...")
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if "auxiliary" not in k:
                name = k[7:]  # remove module.
                new_state_dict[name] = v
            else:
                print("Auxiliary loss is used when retraining.")

        self.net.load_state_dict(new_state_dict)
        self.net.cuda()
        self.net.eval()
        print("Finished loading model!")
        
        self.transform=TestBaseTransform((104, 117, 123))
        
    def preprocess(self, img):
        x = torch.from_numpy(self.transform(img)[0]).permute(2, 0, 1)
        x = x.unsqueeze(0).cuda()
        return x
    
    def get_box_with_best_score(self, img):
        # post-processing
        detections = self.net(img).view(-1, 5)
        # scale each detection back up to the image
        scale = torch.Tensor(
            [
                img_cv2.shape[1],
                img_cv2.shape[0],
                img_cv2.shape[1],
                img_cv2.shape[0]
            ]
        ).cuda()
        scores = detections[..., 0]
        boxes = detections[..., 1:] * scale

        # filter the boxes whose score is smaller than 0.8
        keep_mask = (scores >= args.thresh) & (boxes[..., -1] > 2.0)
        scores = scores[keep_mask]
        boxes = boxes[keep_mask]
        
        keep_idx = torch_nms(boxes, scores, iou_threshold=0.4)
        box = ()
        if len(keep_idx) > 0:
            keep_boxes = boxes[keep_idx].cpu().numpy()
            keep_scores = scores[keep_idx].cpu().numpy()

            max_score_id = np.argmax(keep_scores)
            box = keep_boxes[max_score_id]
        
        return box
    
def get_cropped_img(img_cv2, box, save_im=False, save_path=None):
    box = np.array(box, np.int32)
    c_x = round((box[2] + box[0])*0.5)
    c_y = round((box[3] + box[1])*0.5)
    diff_x = abs(box[2] - box[0])*0.5
    diff_y = abs(box[3] - box[1])*0.5
    diff = max(diff_x, diff_y)

    lb_x = max(0, c_x - round(diff*1.75))
    lb_y = max(0, c_y - round(diff*1.75))
    hb_x = min(c_x + round(diff*1.75), img_cv2.shape[1])
    hb_y = min(c_y + round(diff*1.75), img_cv2.shape[0])
    cropped_img = img_cv2[lb_y:hb_y, lb_x:hb_x]
    img_cv2 = cv2.rectangle(img_cv2, (lb_x, lb_y), (hb_x, hb_y), color=(0, 0, 255), thickness=2)
    
    if save_im:
        cv2.imwrite(save_path, cropped_img)
    
    return cropped_img, img_cv2 

def parse_args():
    parser = argparse.ArgumentParser(description="Single Shot MultiBox Detection")
    parser.add_argument("--model-path", default="./weights/dsfdv2_r18.pth", type=str, help="Trained state_dict file path to open")
    parser.add_argument("--in-dir", default='imgs', type=str, help="where images are stored")
    parser.add_argument("--out-dir", default='results/', type=str, help="where results will be stored")
    parser.add_argument("--thresh", default=0.8, type=float, help="Final confidence threshold")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    save_path = args.out_dir
    os.makedirs(save_path, exist_ok=True)

    files = glob.glob(os.path.join(args.in_dir, "*.png")) + glob.glob(os.path.join(args.in_dir, "*.jpg")) + glob.glob(os.path.join(args.in_dir, "*.jpeg")) + glob.glob(os.path.join(args.in_dir, "*.bmp"))
    
    fd = FaceDetection()
    
    for f in files:
        img_name = f.split("/")[-1]
        
        with torch.no_grad():
            
            img_cv2 = cv2.imread(f)
            img = fd.preprocess(img_cv2)
            box = fd.get_box_with_best_score(img)
            if len(box) == 4:
                cropped_img, img_cv2 = get_cropped_img(img_cv2, box, save_im=True, save_path=os.path.join(save_path, os.path.splitext(img_name)[0] + '_face.jpg'))
    
    