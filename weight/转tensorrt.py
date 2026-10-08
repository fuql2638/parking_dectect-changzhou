from ultralytics import YOLO
from train_parking_UAV import UAVImageOnlyOcclusion
# 加载训练好的模型
model = YOLO(r'D:\Pycharm\PythonProject\car_park_detect\runs\segment\runs_uav_parking\yolo26s_seg_uav_aug_v1-2\weights\tcw0914.pt')

# 导出为 TensorRT engine（默认使用 GPU 0）
model.export(format='engine', device=0,half=True, imgsz=1280)  # 也可 device='cpu' 但推理需 GPU