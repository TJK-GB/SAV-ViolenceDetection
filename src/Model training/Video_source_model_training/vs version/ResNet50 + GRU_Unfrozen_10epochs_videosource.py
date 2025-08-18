import os, random, numpy as np, pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50, ResNet50_Weights
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
train_csv_path = r"D:\UK\00. 2024 QMUL\00. Course\SAV-ViolenceDetection\Data_split\Video_Source_Considered_split\train.csv"
test_csv_path  = r"D:\UK\00. 2024 QMUL\00. Course\SAV-ViolenceDetection\Data_split\Video_Source_Considered_split\test.csv"
NPY_DIR        = r"D:\UK\00. 2024 QMUL\00. Course\SAV-ViolenceDetection\Dataset\npy_segments_videosource"
save_path      = r"D:\UK\00. 2024 QMUL\00. Course\SAV-ViolenceDetection\Results\Model Training(w video)\ResNet50"
os.makedirs(save_path, exist_ok=True)

# Configuration
BATCH_SIZE = 4
MAX_FRAMES = 80
EPOCHS = 10
USE_WEIGHTED_LOSS = True
NUM_SOURCES = 7

# ResNet50 and forward
class ResNet50GRU_EarlyFusion(nn.Module):
    def __init__(self, hidden_size=256, num_layers=1):
        super().__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.resnet.fc = nn.Identity()  # Output: 2048-D
        self.gru = nn.GRU(2048, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2 + NUM_SOURCES, 1)  # + one-hot source vector

    def forward(self, frames, src_onehot):
        B, T, C, H, W = frames.shape
        x = frames.view(B*T, C, H, W)
        feats = self.resnet(x)                     # (B*T, 2048)
        feats = feats.view(B, T, -1)               # (B, T, 2048)
        _, h_n = self.gru(feats)                   # (num_layers*2, B, hidden_size)
        h_n = h_n.permute(1,0,2).reshape(B, -1)    # (B, hidden_size*2)
        combined = torch.cat([h_n, src_onehot], dim=1)
        return self.fc(combined).squeeze(1)

# Dataset load and earlyfusion configuration
class ViolenceDataset_EarlyFusion(Dataset):
    def __init__(self, csv_path, npy_dir):
        self.df = pd.read_csv(csv_path)
        self.npy_dir = npy_dir
        self.resize = Resize((224,224))
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        frames_np = np.load(os.path.join(self.npy_dir, f"{row['Segment ID']}.npy"))

        # Convert to tensor in range [0,1]
        frames_tensors = [self.resize(torch.from_numpy(f).permute(2,0,1).float()/255.0) for f in frames_np]

        # Pad or truncate
        if len(frames_tensors) < MAX_FRAMES:
            pad_len = MAX_FRAMES - len(frames_tensors)
            pad_frames = [torch.zeros(3,224,224)] * pad_len
            frames_tensors.extend(pad_frames)
        frames_tensors = frames_tensors[:MAX_FRAMES]

        frames = torch.stack(frames_tensors)  # Shape: [MAX_FRAMES, 3, 224, 224]

        label = torch.tensor(row['Violence label(video)'], dtype=torch.float32)
        src_label = int(row['Video Source Label'])
        src_onehot = torch.zeros(NUM_SOURCES); src_onehot[src_label] = 1.0
        return frames, label, src_onehot

# Main script for train/test loader, model configuration
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = ViolenceDataset_EarlyFusion(train_csv_path, NPY_DIR)
    test_dataset  = ViolenceDataset_EarlyFusion(test_csv_path, NPY_DIR)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

    pos = train_dataset.df['Violence label(video)'].sum()
    neg = len(train_dataset) - pos
    ratio = neg / max(pos, 1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([ratio]).to(device)) if USE_WEIGHTED_LOSS else nn.BCEWithLogitsLoss()

    model = ResNet50GRU_EarlyFusion().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = GradScaler()

    best_f1 = 0

    # train
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
            torch.save(model.state_dict(), os.path.join(save_path, "resnet50_gru_earlyfusion_best.pt"))

    # test
    model.load_state_dict(torch.load(os.path.join(save_path, "resnet50_gru_earlyfusion_best.pt")))
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

    pd.DataFrame({"Segment ID": segment_ids, "True": y_true, "Pred": y_pred}).to_csv(
        os.path.join(save_path, "resnet50_gru_earlyfusion_predictions.csv"), index=False)
    pd.DataFrame(report).to_csv(os.path.join(save_path, "resnet50_gru_earlyfusion_test_metrics.csv"))

    print(f"\n[TEST] BCE Loss: {avg_test_loss:.4f}")
    print(f"[TEST] Macro F1: {report['macro avg']['f1-score']:.4f}")
    print(f"[TEST] Micro F1: {f1_score(y_true,y_pred,average='micro'):.4f}")
    print("Per-Class F1 Scores:",
          report["Non-violent"]["f1-score"],
          report["Violent"]["f1-score"])
    print("Confusion Matrix:\n", conf_matrix)

if __name__ == "__main__":
    main()
