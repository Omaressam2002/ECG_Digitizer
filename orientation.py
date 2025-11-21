# orientation model 
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

class OrientationModel():
    def __init__(self,model_path='',IMG_SIZE=224):
        self.angle_classes = [0, 5, -5, 10, -10, 15, -15, 20, -20, 25, -25, 30, -30, 45, -45, 60, -60,
                              90, -90, 120, -120, 150, -150, 180]
        self.img_size = IMG_SIZE
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(self.angle_classes))
        model.load_state_dict(torch.load(model_path, map_location=self.device))  
        model.to(self.device)
        self.model = model
        self.val_tf = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406], 
                                     std=[0.229,0.224,0.225])
            ])
    def predict_angle(self, image_path):
        img = Image.open(image_path).convert("RGB")
        x = self.val_tf(img).unsqueeze(0).to(self.device)   # [1, 3, 224, 224]
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)
            pred_class = logits.argmax(1).item()
    
        angle = self.angle_classes[pred_class]
        return angle
    def align_image(self, image_path, angle):
        img = Image.open(image_path).convert("RGB")
        # Reverse the rotation
        corrected_img = img.rotate(-angle, expand=True, resample=Image.Resampling.BICUBIC)
        # shoof yolo bya5od eh we raga3o
        return corrected_img
    def fix_image(self, image_path, angle):
        angle = self.predict_angle(image_path)
        retuen self.align_image(image_path, angle)


