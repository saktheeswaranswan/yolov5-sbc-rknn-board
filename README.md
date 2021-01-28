原版仓库：https://github.com/ultralytics/yolov5

环境要求：python version >= 3.6

模型训练：python3 train.py

模型导出：python3 models/export.py --weights "xxx.pt"

转换rknn：python3 onnx_to_rknn.py

模型推理：python3 rknn_detect_yolov5.py

注意事项：如果训练尺寸不是640那么，anchors会自动聚类重新生成，生成的结果在训练时打印在控制台，或者通过动态查看torch模型类属性获取，如果anchors不对应那么结果就会出现问题。

建议：在训练时如果size不是640，那么可以先通过聚类得到anchors并将新的anchors写入到模型配置文件中，然后再训练，防止动态获取的anchors在rknn上预测不准的问题。训练参数别忘记加上 --noautoanchor。

