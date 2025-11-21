import os
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm

###################################
#            Config
###################################

ANGLE_CLASSES = [0, 15, -15, 30, -30, 45, -45, 60, -60,
                 75, -75, 90, -90, 105, -105, 130, -130, 150, -150, 180]

angle_to_idx = {a: i for i, a in enumerate(ANGLE_CLASSES)}

IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 25
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



class AngleDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg",".png",".jpeg"))]
        self.transform = transform

    def extract_angle(self, filename):
        # Find angle inside filename (supports rot_15, rot_-45, _15_, etc.)
        match = re.search(r'(-?\d+)', filename)
        if not match:
            raise ValueError(f"Could not extract angle from: {filename}")
        angle = int(match.group(1))
        return angle_to_idx[angle]

    def __len__(self): 
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        path = os.path.join(self.folder, fname)

        img = Image.open(path).convert("RGB")
        y = self.extract_angle(fname)

        if self.transform:
            img = self.transform(img)

        return img, y



train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomApply([
        transforms.ColorJitter(0.1,0.1,0.1,0.05)
    ], p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])
# why? mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]


train_ds = AngleDataset("dataset/train", transform=train_tf)
val_ds   = AngleDataset("dataset/val", transform=val_tf)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_dl   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)



model = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(ANGLE_CLASSES))
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

scaler = torch.cuda.amp.GradScaler()


def evaluate():
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in val_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x)
            correct += (preds.argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total

best_acc = 0

for epoch in range(EPOCHS):
    model.train()
    loop = tqdm(train_dl, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for imgs, labels in loop:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()

        # AMP training
        with torch.cuda.amp.autocast():
            preds = model(imgs)
            loss = criterion(preds, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())

    acc = evaluate()
    scheduler.step()

    print(f"Val Accuracy: {acc:.4f}")

    # Save best model
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "best_mobilenetv3_angle.pth")
        print("🔥 Saved new best model!")

print("Training complete!")
