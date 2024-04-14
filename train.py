import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('C:\\Users\\jsj_gjj\\Desktop\\utralytics-main\\ultralytics\\cfg\\models\\v8\\yolov8_RepGhost.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data='C:\\Users\jsj_gjj\\Desktop\\utralytics-main\\AITEX.yaml',

                imgsz=640,
                epochs=300,
                batch=32,

                device='0',
                # using SGD
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                project='C:\\Users\\jsj_gjj\\Desktop\\ultralytics-main\\runs\\detect',
                name='exp',
                )