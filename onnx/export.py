"""Exports a YOLOv5 *.pt model to ONNX and TorchScript formats

Usage:
    $ export PYTHONPATH="$PWD" && python models/export.py --weights ./weights/yolov5s.pt --img 640 --batch 1
"""


# 转onnx的时候需要把symbolic_opset11_change.py替换到annaconda环境里面去
import argparse

from models.common import *
from utils import google_utils
import models.yolo as yolo
import datetime

if __name__ == '__main__':
    time = datetime.datetime.now().strftime('%Y-%m-%d')
    dirction = "top"
    task = "ganggan"
    pth_path = "/data/yolov5_wh/work_dir/ganggan_9_21_top/2020-09-21/2020-09-21_16:00:04/epoch_100.pth"
    onnx_path = "/data/yolov5_wh/onnx/out/{}_{}_{}.onnx".format(
        task, dirction, time)
    # Input
    img = torch.zeros((1, 12, 320, 320))  # image size(1,3,320,192) iDetection
    model = yolo.Model("models/yolov5x.yaml")
    # Load PyTorch model
    model_cc = torch.load(pth_path, map_location=torch.device('cpu')).float()
    model.load_state_dict(model_cc.state_dict())
    model.eval()
    model.model[-1].export = True  # set Detect() layer export=True
    y = model(img)  # dry run

    # ONNX export
    import torch.onnx
    model.fuse()  # only for ONNX
    torch.onnx.export(model, img, onnx_path, verbose=False,
                      opset_version=11, input_names=['images'])
