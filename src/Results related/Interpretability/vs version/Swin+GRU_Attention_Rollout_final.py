import os, math, numpy as np, torch, torch.nn as nn, matplotlib.pyplot as plt
from torchvision.models import swin_t, Swin_T_Weights
from torchvision.transforms import Resize

# Paths change these to your local paths
NPY_DIR   = r"D:\UK\00. 2024 QMUL\00. Course\SAV-ViolenceDetection\Dataset\npy_segments_unimodal"
CKPT_PATH = r"D:\UK\00. 2024 QMUL\00. Course\SAV-ViolenceDetection\Results\Model Training(w.o video)\Swin+GRU\swin_gru_best.pt"
SAVE_DIR  = r"D:\UK\00. 2024 QMUL\00. Course\SAV-ViolenceDetection\Results\Interpretability\Attention_RollOut_SwinGRU"
IMG_SIZE  = (224,224)
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(SAVE_DIR, exist_ok=True)

# set segments
SEGMENTS = {
    "FN": ("169_10", [0, 20, 40, 60]),
    "FP": ("10_1",   [0, 10, 30, 50]),
    "TN": ("102_3",  [0, 15, 35, 55]),
    "TP": ("133_1",  [0, 25, 45, 65]),
}

# Swin + GRU
class SwinGRUOptional(nn.Module):
    def __init__(self, use_gru=True):
        super().__init__()
        self.use_gru = use_gru
        self.swin = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        self.swin.head = nn.Identity()  # remove classifier

        if self.use_gru:
            self.gru = nn.GRU(768, 256, batch_first=True, bidirectional=True)
            self.fc  = nn.Linear(512, 1)   # GRU → FC
        else:
            self.fc  = nn.Linear(768, 1)   # Swin → FC

    def forward(self, x):   # GRU: [B,T,C,H,W], Swin: [B,C,H,W]
        if self.use_gru:
            B,T,C,H,W = x.shape
            x = x.view(B*T, C, H, W)
            feats = self.swin(x)          # [B*T,768]
            feats = feats.view(B, T, -1)  # [B,T,768]
            out, _ = self.gru(feats)      # [B,T,512]
            out = out[:, -1]              # last timestep
            return torch.sigmoid(self.fc(out)), feats
        else:
            feats = self.swin(x)          # [B,768]
            return torch.sigmoid(self.fc(feats)), feats

# Attention Rollout
class AttentionRollout:
    def __init__(self, model):
        self.model = model
        self.attentions = []
        self.handles = []
        self._register_hooks()

    def _register_hooks(self):
        # Only hook into Swin blocks
        for stage in self.model.swin.features:
            if isinstance(stage, torch.nn.Sequential):
                for block in stage:
                    if hasattr(block, 'attn'):
                        h = block.attn.register_forward_hook(self._get_attention)
                        self.handles.append(h)

    def _get_attention(self, module, input, output):
        qkv = module.qkv(input[0])
        if qkv.ndim == 3:
            B_, N, _ = qkv.shape
            qkv = qkv.reshape(B_, N, 3, module.num_heads, -1).permute(2, 0, 3, 1, 4)
        elif qkv.ndim == 4:
            B_, H, W, _ = qkv.shape
            N = H * W
            qkv = qkv.reshape(B_, N, 3, module.num_heads, -1).permute(2, 0, 3, 1, 4)
        else:
            raise ValueError(f"Unexpected qkv shape: {qkv.shape}")

        q, k, v = qkv[0], qkv[1], qkv[2]
        head_dim = q.shape[-1]
        scale = 1.0 / math.sqrt(head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)  # [B_, heads, N, N]
        self.attentions.append(attn.detach().cpu())

    def __call__(self, x):
        self.attentions = []
        _ = self.model(x)  # forward pass

        # take last attention map
        attn = self.attentions[-1]   # [B_, heads, N, N]
        attn = attn.mean(1)          # avg heads → [B_, N, N]
        return attn[0]               # first in batch

# Helpers
def load_segment(seg_id, frame_indices):
    arr = np.load(os.path.join(NPY_DIR, f"{seg_id}.npy"))  # [T,H,W,C]
    frames = []
    for idx in frame_indices:
        f = arr[idx]
        if f.ndim == 2:  # grayscale → RGB
            f = np.stack([f]*3, axis=-1)
        t = torch.from_numpy(f).permute(2,0,1).float()/255.0
        t = Resize(IMG_SIZE)(t)
        frames.append(t)
    return torch.stack(frames)  # [T,C,H,W]

def overlay_and_save(img_chw, cam, save_path):
    img = img_chw.permute(1,2,0).cpu().numpy().clip(0,1)

    # cam: [N, N] (square matrix, e.g. [49,49])
    if isinstance(cam, torch.Tensor):
        cam = cam.cpu().numpy()

    if cam.ndim == 2 and cam.shape[0] == cam.shape[1]:
        # collapse query dimension → importance per patch
        cam = cam.mean(0)
        N = int(cam.shape[0]**0.5) if int(cam.shape[0]**0.5)**2 == cam.shape[0] else cam.shape[0]
        if N*N == cam.shape[0]:
            cam = cam.reshape(N, N)
        else:
            cam = cam.reshape(int(math.sqrt(cam.shape[0])), -1)
    elif cam.ndim == 1:
        N = int(cam.shape[0]**0.5)
        cam = cam.reshape(N, N)
    else:
        raise ValueError(f"Unexpected cam shape: {cam.shape}")

    # upscale to full image
    cam = torch.tensor(cam).unsqueeze(0).unsqueeze(0)
    cam = torch.nn.functional.interpolate(cam, size=IMG_SIZE, mode='bilinear', align_corners=False)
    cam = cam.squeeze().numpy()
    cam = (cam - cam.min())/(cam.max() - cam.min() + 1e-8)

    heatmap = plt.cm.jet(cam)[...,:3]
    vis = (0.5*img + 0.5*heatmap).clip(0,1)
    plt.imsave(save_path, vis)

# Main
if __name__ == "__main__":
    USE_GRU = True   # flip this: True = Swin+GRU, False = Swin-only

    model = SwinGRUOptional(use_gru=USE_GRU).to(DEVICE)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
    model.eval()

    rollout = AttentionRollout(model)

    for cat,(seg_id,frame_idx_list) in SEGMENTS.items():
        frames = load_segment(seg_id, frame_idx_list).to(DEVICE)  # [T,C,H,W]

        if USE_GRU:
            frames_in = frames.unsqueeze(0).to(DEVICE)            # [1,T,C,H,W]
            cam = rollout(frames_in)
            for i,frame in enumerate(frames):
                overlay_and_save(frame, cam, os.path.join(SAVE_DIR, f"{cat}_{seg_id}_f{i}.png"))
        else:
            for i,frame in enumerate(frames):
                frame_in = frame.unsqueeze(0).to(DEVICE)          # [1,C,H,W]
                cam = rollout(frame_in)
                overlay_and_save(frame, cam, os.path.join(SAVE_DIR, f"{cat}_{seg_id}_f{i}.png"))

        print(f"Saved attention maps for {cat} {seg_id}")
