from ultralytics import YOLO

model = YOLO("yolov8n.yaml")

results = model.train(data = "config.yaml", 
                      epochs=50,
                      save=True,
                      batch=32,
                      val=True,
                      verbose=True,
                      augment=True)