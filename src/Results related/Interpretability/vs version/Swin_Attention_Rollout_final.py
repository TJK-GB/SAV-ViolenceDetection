import os, numpy as np, torch, torch.nn as nn, matplotlib.pyplot as plt
from torchvision.models import swin_t, Swin_T_Weights
from torchvision.transforms import Resize
import math

# Paths change these to your local paths
NPY_DIR   = r"D:\UK\00. 2024 QMUL\00. Course\SAV-ViolenceDetection\Dataset\npy_segments_unimodal"
CKPT_PATH = r"D:\UK\00. 2024 QMUL\00. Course\SAV-ViolenceDetection\Results\Model Training(w.o video)\Swin\swin_best.pt"
SAVE_DIR  = r"D:\UK\00. 2024 QMUL\00. Course\SAV-ViolenceDetection\Results\Interpretability\Attention_RollOut_Swin"
IMG_SIZE  = (224,224)
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(SAVE_DIR, exist_ok=True)

# Set segments
SEGMENTS = {
    "FN": ("169_10", [0, 20, 40, 60]),
    "FP": ("10_1",   [0, 10, 30, 50]),
    "TN": ("102_3",  [0, 15, 35, 55]),
    "TP": ("133_1",  [0, 25, 45, 65]),
}

# Swin
class SwinWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.swin = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        self.swin.head = nn.Identity()
        self.fc = nn.Linear(768, 1)

    def forward(self, x):
        feats = self.swin(x)            # [B,768]
        return torch.sigmoid(self.fc(feats))

# Attention Rollout
class AttentionRollout:
    def __init__(self, model, discard_ratio=0.0):
        self.model = model
        self.attentions = []
        self.handles = []
        self.discard_ratio = discard_ratio
        self._register_hooks()

    def _register_hooks(self):
        for stage in self.model.swin.features:   # goes over stages/patch merging/etc.
            if isinstance(stage, torch.nn.Sequential):
                # this stage contains multiple SwinTransformerBlocks
                for block in stage:
                    if hasattr(block, 'attn'):   # real transformer block
                        h = block.attn.register_forward_hook(self._get_attention)
                        self.handles.append(h)
            else:
                # skip things like PatchMerging
                continue

    def _get_attention(self, module, input, output):
        qkv = module.qkv(input[0])   # [B_, N, 3*dim] or [B_, H, W, 3*dim]

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
        _ = self.model(x)

        # rollout: multiply (I + A) layer by layer
        attn = self.attentions[-1]      # [B_, heads, N, N]
        attn = attn.mean(1)             # average heads
        return attn

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

    # Cam must be numpy, 2D
    if isinstance(cam, torch.Tensor):
        cam = cam.squeeze()   # remove batch dim → [49,49]
        cam = cam.cpu().numpy()

    if cam.ndim == 1:
        N = int(cam.shape[0] ** 0.5)
        cam = cam.reshape(N, N)
    elif cam.ndim == 2 and cam.shape[0] == cam.shape[1]:
        pass
    else:
        raise ValueError(f"Unexpected cam shape after squeeze: {cam.shape}")
    # normalise
    cam = (cam - cam.min())/(cam.max() - cam.min() + 1e-8)
    # upscale to image size
    cam = torch.tensor(cam).unsqueeze(0).unsqueeze(0)
    cam = torch.nn.functional.interpolate(cam, size=IMG_SIZE, mode='bilinear', align_corners=False)
    cam = cam.squeeze().numpy()

    heatmap = plt.cm.jet(cam)[...,:3]
    vis = (0.5*img + 0.5*heatmap).clip(0,1)
    plt.imsave(save_path, vis)

# Main
if __name__ == "__main__":
    model = SwinWrapper().to(DEVICE)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
    model.eval()
    rollout = AttentionRollout(model)

    for cat,(seg_id,frame_idx_list) in SEGMENTS.items():
        frames = load_segment(seg_id, frame_idx_list).to(DEVICE)  # [T,C,H,W]
        for i,frame in enumerate(frames):
            frame_in = frame.unsqueeze(0).to(DEVICE)
            mask = rollout(frame_in)
            overlay_and_save(frame, mask, os.path.join(SAVE_DIR, f"{cat}_{seg_id}_f{i}.png"))
        print(f"Saved Attention Rollout maps for {cat} {seg_id}")