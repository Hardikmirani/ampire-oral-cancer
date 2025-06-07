import argparse
import torch
import cv2
import os
from pathlib import Path
from models.common import DetectMultiBackend
from utils.datasets import LoadImages
from utils.general import (non_max_suppression, scale_coords, check_img_size, increment_path)
from utils.plots import save_one_box
from utils.torch_utils import select_device, time_sync

def run(weights='best.pt', source='data/test_images', img_size=640, conf_thres=0.25, iou_thres=0.45):
    # Set up
    device = select_device('')
    model = DetectMultiBackend(weights, device=device)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(img_size, s=stride)  # check image size

    # Load data
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)

    # Create output directory
    save_dir = increment_path(Path('runs/detect/exp'), exist_ok=False)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Run inference
    model.eval()
    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im).to(device)
        im = im.float() / 255.0  # scale to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # add batch dimension

        # Inference
        pred = model(im, augment=False, visualize=False)
        pred = non_max_suppression(pred, conf_thres, iou_thres)

        # Process predictions
        for det in pred:
            if len(det):
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0s.shape).round()

                # Draw and save results
                for *xyxy, conf, cls in det:
                    label = f'{names[int(cls)]} {conf:.2f}'
                    save_one_box(xyxy, im0s, file=save_dir / Path(path).name, label=label, color=(255, 0, 0), line_thickness=2)

        # Save output image
        cv2.imwrite(str(save_dir / Path(path).name), im0s)

    print(f"Results saved to {save_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/best.pt', help='model.pt path')
    parser.add_argument('--source', type=str, default='data/test_images', help='image folder or image path')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    opt = parser.parse_args()

    run(**vars(opt))
