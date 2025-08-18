import os, random, numpy as np, pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize
from torch.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint

# Reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#  PATHS change to your directory
train_csv_path = r"D:\UK\00. 2024 QMUL\00. Course\SAV-ViolenceDetection\Data_split\Video_Source_Considered_split\train.csv"
test_csv_path  = r"D:\UK\00. 2024 QMUL\00. Course\SAV-ViolenceDetection\Data_split\Video_Source_Considered_split\test.csv"
NPY_DIR        = r"D:\UK\00. 2024 QMUL\00. Course\SAV-ViolenceDetection\Dataset\npy_segments_videosource"
save_path      = r"D:\UK\00. 2024 QMUL\00. Course\SAV-ViolenceDetection\Results\Model Training(w video)\CUE-NET"
os.makedirs(save_path, exist_ok=True)

# Configuration 
BATCH_SIZE = 4
MAX_FRAMES = 80
EPOCHS = 10
USE_WEIGHTED_LOSS = True
NUM_SRC_CLASSES = 7

# CUE-NET backbone
class CUEBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv3d = nn.Conv3d(in_ch, out_ch, kernel_size=(3,3,3), padding=1)
        self.bn = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        def _f(inp):
            return self.relu(self.bn(self.conv3d(inp)))
        return checkpoint(_f, x)

class CUENetClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = CUEBlock(3, 32)
        self.enc2 = CUEBlock(32, 64)
        self.pool = nn.AdaptiveAvgPool3d((1,1,1))
        self.fc = nn.Linear(64 + NUM_SRC_CLASSES, 1)

    def forward(self, x, src_onehot):
        x = x.permute(0, 2, 1, 3, 4)  # B,C,T,H,W
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.pool(x).view(x.size(0), -1)
        fused = torch.cat([x, src_onehot], dim=1)
        return self.fc(fused).squeeze(1)

# Load the dataset
class ViolenceDataset(Dataset):
    def __init__(self, csv_path, npy_dir):
        self.df = pd.read_csv(csv_path)
        self.npy_dir = npy_dir
        self.resize = Resize((224, 224))
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        frames = np.load(os.path.join(self.npy_dir, f"{row['Segment ID']}.npy"))
        frames = [torch.from_numpy(f).permute(2,0,1).float()/255.0 for f in frames]
        frames = [self.resize(f) for f in frames]
        if len(frames) < MAX_FRAMES:
            pad_frame = torch.zeros_like(frames[0])
            frames += [pad_frame] * (MAX_FRAMES - len(frames))
        frames = torch.stack(frames[:MAX_FRAMES])

        src_label = torch.tensor(row['Video Source Label'], dtype=torch.long)
        src_onehot = torch.nn.functional.one_hot(src_label, num_classes=NUM_SRC_CLASSES).float()
        return frames, torch.tensor(row['Violence label(video)'], dtype=torch.float32), src_onehot

# Main script for train/test loader, model configuration
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = ViolenceDataset(train_csv_path, NPY_DIR)
    test_dataset  = ViolenceDataset(test_csv_path,  NPY_DIR)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

    pos = train_dataset.df['Violence label(video)'].sum()
    neg = len(train_dataset) - pos
    ratio = neg / max(pos, 1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([ratio]).to(device)) if USE_WEIGHTED_LOSS else nn.BCEWithLogitsLoss()

    model = CUENetClassifier().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = GradScaler()

    # train
    best_f1 = 0
    for epoch in range(EPOCHS):
        model.train()
        y_true, y_pred, total_loss = [], [], 0.0
        for frames, labels, src_onehot in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            frames, labels, src_onehot = frames.to(device), labels.to(device), src_onehot.to(device)
            with autocast(device_type='cuda'):
                outputs = model(frames, src_onehot)
                loss = criterion(outputs, labels)
            optimiser.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            total_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).int()
            y_true.extend(labels.cpu().numpy()); y_pred.extend(preds.cpu().numpy())
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | Macro F1: {macro_f1:.4f}")
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save(model.state_dict(), os.path.join(save_path, "cuenet_best.pt"))

    # test
    model.load_state_dict(torch.load(os.path.join(save_path, "cuenet_best.pt")))
    model.eval()
    y_true, y_pred, test_losses = [], [], []
    segment_ids = test_dataset.df['Segment ID'].tolist()
    with torch.no_grad():
        for frames, labels, src_onehot in test_loader:
            frames, labels, src_onehot = frames.to(device), labels.to(device), src_onehot.to(device)
            outputs = model(frames, src_onehot)
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
        os.path.join(save_path, "cuenet_predictions.csv"), index=False)
    pd.DataFrame(report).to_csv(os.path.join(save_path, "cuenet_test_metrics.csv"))

if __name__ == "__main__":
    main()
