import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from torch.amp import autocast, GradScaler

# Reproducibility

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#  PATHS change to your directory
train_csv_path = r"D:\UK\00. 2024 QMUL\00. Course\SAV-ViolenceDetection\Data_split\Violence_Label_Only_split\train.csv"
test_csv_path  = r"D:\UK\00. 2024 QMUL\00. Course\SAV-ViolenceDetection\Data_split\Violence_Label_Only_split\test.csv"
NPY_DIR  = r"D:\UK\00. 2024 QMUL\00. Course\SAV-ViolenceDetection\Dataset\npy_segments_unimodal"
save_path = r"D:\UK\00. 2024 QMUL\00. Course\SAV-ViolenceDetection\Results\Model Training(w.o video)\ResNet50+GRU"

os.makedirs(save_path, exist_ok=True)

# Configuration 
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 4
EPOCHS = 20
MAX_FRAMES = 80
EARLY_STOPPING_PATIENCE = 4

# ResNet50 + GRU model setting
class ResNet50GRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.resnet.fc = nn.Identity()  # remove classifier
        self.gru = nn.GRU(2048, 256, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.attn = nn.Linear(512, 1)
        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)
        feats = self.resnet(x).view(B, T, -1)
        out, _ = self.gru(feats)
        weights = torch.softmax(self.attn(out), dim=1)
        out = torch.sum(weights * out, dim=1)
        out = self.dropout(out)
        return self.fc(out).squeeze(1)

# Dataset load and apply
class ViolenceDataset(Dataset):
    def __init__(self, csv_path, npy_dir):
        self.df = pd.read_csv(csv_path)
        self.npy_dir = npy_dir
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.485,0.456,0.406),
                                 (0.229,0.224,0.225))
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        frames = np.load(os.path.join(self.npy_dir, f"{row['Segment ID']}.npy"))[:MAX_FRAMES]
        frames = torch.stack([
            self.transform(torch.from_numpy(f).permute(2,0,1).float()/255.0)
            for f in frames
        ])
        label = torch.tensor(row['Violence label(video)'], dtype=torch.float32)
        return frames, label


# Train/test loader, model configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = ViolenceDataset(train_csv_path, NPY_DIR)
test_dataset  = ViolenceDataset(test_csv_path,  NPY_DIR)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

model = ResNet50GRU().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=1)
scaler = GradScaler()

best_loss = float("inf")
early_stop_counter = 0

# train
for epoch in range(EPOCHS):
    model.train()
    y_true, y_pred = [], []
    total_loss = 0.0
    optimizer.zero_grad()

    for i, (frames, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")):
        frames, labels = frames.to(device), labels.to(device)
        with autocast(device_type='cuda'):
            outputs = model(frames)
            loss = criterion(outputs, labels) / GRAD_ACCUM_STEPS

        scaler.scale(loss).backward()

        if (i+1) % GRAD_ACCUM_STEPS == 0 or (i+1) == len(train_loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * GRAD_ACCUM_STEPS
        preds = (torch.sigmoid(outputs) > 0.5).int()
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

    macro_f1 = f1_score(y_true, y_pred, average="macro")
    micro_f1 = f1_score(y_true, y_pred, average="micro")
    print(f"Epoch {epoch+1} | Loss: {total_loss:.4f} | Macro F1: {macro_f1:.4f} | Micro F1: {micro_f1:.4f}")

    scheduler.step(total_loss)

    if total_loss < best_loss:
        best_loss = total_loss
        torch.save(model.state_dict(), os.path.join(save_path, "resnet50_gru_best.pt"))
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= EARLY_STOPPING_PATIENCE:
            break

# testing
model.load_state_dict(torch.load(os.path.join(save_path, "resnet50_gru_best.pt")))
model.eval()

y_true, y_pred, test_losses = [], [], []
segment_ids = test_dataset.df['Segment ID'].tolist()

with torch.no_grad():
    for frames, labels in test_loader:
        frames, labels = frames.to(device), labels.to(device)
        outputs = model(frames)
        loss = criterion(outputs, labels)
        test_losses.append(loss.item())
        preds = (torch.sigmoid(outputs) > 0.5).int()
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

avg_test_loss = np.mean(test_losses)
report = classification_report(
    y_true, y_pred,
    target_names=["Non-violent", "Violent"],
    output_dict=True,
    zero_division=0)
conf_matrix = confusion_matrix(y_true, y_pred)

print(f"\n[TEST] BCE Loss: {avg_test_loss:.4f}")
print(f"[TEST] Macro F1: {report['macro avg']['f1-score']:.4f}")
print(f"[TEST] Micro F1: {f1_score(y_true, y_pred, average='micro'):.4f}")
print("[TEST] Per-Class F1 Scores:")
print(f" - Non-violent F1: {report['Non-violent']['f1-score']:.4f}")
print(f" - Violent F1: {report['Violent']['f1-score']:.4f}")
print("Confusion Matrix:\n", conf_matrix)

pd.DataFrame({
    "Segment ID": segment_ids,
    "True": y_true,
    "Pred": y_pred
}).to_csv(os.path.join(save_path, "resnet50_gru_predictions.csv"), index=False)

pd.DataFrame(report).to_csv(os.path.join(save_path, "resnet50_gru_test_metrics.csv"))
