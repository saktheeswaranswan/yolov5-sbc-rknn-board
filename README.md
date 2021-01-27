模型训练：python3 train.py

模型导出：python3 models/export.py --weights "xxx.pt"

转rknn ：python3 onnx_to_rknn.py

模型推理：python3 rknn_detect_yolov5.py

原版仓库：https://github.com/ultralytics/yolov5

python version >= 3.6