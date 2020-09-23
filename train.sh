#python train.py --data coco.yaml --cfg yolov5x.yaml --weight 'weights/yolov5x.pt' --batch-size 16 --evolve --device 0,1,2,3

python convert2yolov5.py &&
python train.py
