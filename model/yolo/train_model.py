from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)
train_data_path = r'C:\Users\annch\OneDrive\Desktop\master\ocr\model\yolo\dataset'
# Train the model
if __name__ == '__main__':
    model.train(data=train_data_path, epochs=20, batch=2)