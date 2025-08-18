import os, random, numpy as np, pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import swin_t, Swin_T_Weights
from torchvision.transforms import Resize
from torch.amp import autocast, GradScaler

# Reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

#  PATHS change to your directory
train_csv_path = r"D:\UK\00. 2024 QMUL\00. Course\SAV-ViolenceDetection\Data_split\Violence_Label_Only_split\train.csv"
test_csv_path  = r"D:\UK\00. 2024 QMUL\00. Course\SAV-ViolenceDetection\Data_split\Violence_Label_Only_split\test.csv"
NPY_DIR        = r"D:\UK\00. 2024 QMUL\00. Course\SAV-ViolenceDetection\Dataset\npy_segments_unimodal"
save_path      = r"D:\UK\00. 2024 QMUL\00. Course\SAV-ViolenceDetection\Results\Model Training(w.o video)\Swin+GRU"
os.makedirs(save_path, exist_ok=True)

# Configuration 
BATCH_SIZE = 2
MAX_FRAMES = 80
EPOCHS = 20
USE_WEIGHTED_LOSS = True
PATIENCE = 4

# Swin + GRU model setting
class SwinGRUClassifier(nn.Module):
    def __init__(self, hidden_size=256, num_layers=1):
        super().__init__()
        self.swin = swin_t(weights=Swin_T_Weights.DEFAULT)
        self.swin.head = nn.Identity()  # Fully unfrozen backbone
        self.gru = nn.GRU(
            input_size=768,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        features = self.swin(x).view(B, T, -1)  # Shape: [B, T, 768]
        gru_out, _ = self.gru(features)         # Shape: [B, T, 2*hidden_size]
        pooled = gru_out.mean(dim=1)
        return self.fc(pooled).squeeze(1)

# Dataset load and apply
class ViolenceDataset(Dataset):
    def __init__(self, csv_path, npy_dir):
        self.df = pd.read_csv(csv_path)
        self.npy_dir = npy_dir
        self.resize = Resize((224, 224))
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        frames = np.load(os.path.join(self.npy_dir, f"{row['Segment ID']}.npy"))[:MAX_FRAMES]
        frames = torch.stack([self.resize(torch.from_numpy(f).permute(2,0,1).float()/255.0) for f in frames])
        return frames, torch.tensor(row['Violence label(video)'], dtype=torch.float32)

# Train/test loader, model configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset = ViolenceDataset(train_csv_path, NPY_DIR)
test_dataset  = ViolenceDataset(test_csv_path,  NPY_DIR)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

pos = train_dataset.df['Violence label(video)'].sum()
neg = len(train_dataset) - pos
ratio = neg / max(pos, 1)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([ratio]).to(device)) if USE_WEIGHTED_LOSS else nn.BCEWithLogitsLoss()

model = SwinGRUClassifier().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=1, factor=0.5)
scaler = GradScaler()

best_f1, early_stop_counter = 0, 0

# train
for epoch in range(EPOCHS):
    model.train()
    y_true, y_pred, total_loss = [], [], 0.0
    for frames, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        frames, labels = frames.to(device), labels.to(device)
        with autocast(device_type='cuda'):
            outputs = model(frames)
            loss = criterion(outputs, labels)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).int()
        y_true.extend(labels.cpu().numpy()); y_pred.extend(preds.cpu().numpy())
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    micro_f1 = f1_score(y_true, y_pred, average='micro')
    print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | Macro F1: {macro_f1:.4f} | Micro F1: {micro_f1:.4f}")
    scheduler.step(macro_f1)
    if macro_f1 > best_f1:
        best_f1 = macro_f1
        torch.save(model.state_dict(), os.path.join(save_path, "swin_gru_best.pt"))
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= PATIENCE: break

# test
model.load_state_dict(torch.load(os.path.join(save_path, "swin_gru_best.pt")))
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
        y_true.extend(labels.cpu().numpy()); y_pred.extend(preds.cpu().numpy())

avg_test_loss = np.mean(test_losses)
report = classification_report(y_true, y_pred, target_names=["Non-violent","Violent"], output_dict=True, zero_division=0)
conf_matrix = confusion_matrix(y_true, y_pred)
print(f"\n[TEST] BCE Loss: {avg_test_loss:.4f}")
print(f"[TEST] Macro F1: {report['macro avg']['f1-score']:.4f}")
print(f"[TEST] Micro F1: {f1_score(y_true,y_pred,average='micro'):.4f}")
print("[TEST] Per-Class F1 Scores:")
print(f" - Non-violent F1: {report['Non-violent']['f1-score']:.4f}")
print(f" - Violent F1: {report['Violent']['f1-score']:.4f}")
print("Confusion Matrix:\n", conf_matrix)

pd.DataFrame({"Segment ID": segment_ids, "True": y_true, "Pred": y_pred}).to_csv(
    os.path.join(save_path, "swin_gru_predictions.csv"), index=False)
pd.DataFrame(report).to_csv(os.path.join(save_path, "swin_gru_test_metrics.csv"))
