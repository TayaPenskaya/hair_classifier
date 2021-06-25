import argparse
import os
import glob
import torch
import cv2

from PIL import Image

from face_detection.detection import FaceDetection, get_cropped_img
from hair_segmentation.segmentation import FaceParsing, get_face_segmentation, get_hair_mask


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", default='imgs', type=str, help="where images are stored")
    parser.add_argument("--out-dir", default='results', type=str, help="where results will be stored")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    save_path = args.out_dir
    os.makedirs(save_path, exist_ok=True)
    
    out_path_hair = os.path.join(save_path, 'hair')
    #out_path_face = os.path.join(save_path, 'face')
    out_path_box = os.path.join(save_path, 'box')
    os.makedirs(out_path_hair, exist_ok=True)
    #os.makedirs(out_path_face, exist_ok=True)
    os.makedirs(out_path_box, exist_ok=True)

    files = glob.glob(os.path.join(args.in_dir, "*.png")) + glob.glob(os.path.join(args.in_dir, "*.jpg")) + glob.glob(os.path.join(args.in_dir, "*.jpeg")) + glob.glob(os.path.join(args.in_dir, "*.bmp"))
    
    fp = FaceParsing()
    fd = FaceDetection()
    
    for f in files:
        img_name = f.split("/")[-1]
        
        with torch.no_grad():
            
            img_cv2 = cv2.imread(f)
            img = fd.preprocess(img_cv2)
            box = fd.get_box_with_best_score(img, img_cv2)
            if len(box) == 4:
                box_path = os.path.join(out_path_box, img_name)
                cropped_img, img_cv2 = get_cropped_img(img_cv2, box, save_im=True, save_path=box_path)
                
                cropped_img_pil = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
                
                parsing = fp.get_parsing(cropped_img_pil)
                parsing = cv2.resize(parsing, cropped_img.shape[0:2][::-1], interpolation=cv2.INTER_NEAREST)
                #face = get_face_segmentation(image, parsing, stride=1, save_im=True, save_path=os.path.join(out_path_face, img_name))
                hair_mask = get_hair_mask(cropped_img, parsing, save_im=True, save_path=os.path.join(out_path_hair, img_name))
            else:
                print('no faces there')
