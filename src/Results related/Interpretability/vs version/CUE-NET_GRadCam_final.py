import os, numpy as np, torch, torch.nn as nn, matplotlib.pyplot as plt
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import Resize

# Paths change these to your local paths
NPY_DIR   = r"D:\UK\00. 2024 QMUL\00. Course\SAV-ViolenceDetection\Dataset\npy_segments_unimodal"
CKPT_PATH = r"D:\UK\00. 2024 QMUL\00. Course\SAV-ViolenceDetection\Results\Model Training(w.o video)\CUE-NET\cuenet_best_10.pt"
SAVE_DIR  = r"D:\UK\00. 2024 QMUL\00. Course\SAV-ViolenceDetection\Results\Interpretability\GradCAM_CUENET"

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

# CUE-NET model
class CUE_NET(nn.Module):
    def __init__(self):
        super().__init__()
        # ResNet backbone
        self.cnn = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.cnn.fc = nn.Identity()
        # GRU head
        self.gru = nn.GRU(2048, 256, batch_first=True, bidirectional=True)
        self.fc  = nn.Linear(512, 1)
    def forward(self, x):  # [B,T,C,H,W]
        B,T,C,H,W = x.shape
        x = x.view(B*T,C,H,W)
        feats = self.cnn(x)             # [B*T,2048]
        feats = feats.view(B,T,-1)      # [B,T,2048]
        out, _ = self.gru(feats)
        out = out[:,-1]                 # last timestep
        return torch.sigmoid(self.fc(out)), feats

# GradCAM hook
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook()
    def hook(self):
        def fwd_hook(module, inp, out):
            self.activations = out.detach()
        def bwd_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        self.target_layer.register_forward_hook(fwd_hook)
        self.target_layer.register_full_backward_hook(bwd_hook)
    def __call__(self, x):
        self.model.zero_grad()
        feats = self.model.cnn(x)   # only backbone, bypass GRU
        score = feats.mean()
        score.backward(torch.ones_like(score))
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(cam, size=IMG_SIZE, mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

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
    heatmap = plt.cm.jet(cam)[...,:3]
    vis = (0.5*img + 0.5*heatmap).clip(0,1)
    plt.imsave(save_path, vis)

# Main
if __name__ == "__main__":
    model = CUE_NET().to(DEVICE)

    # Load checkpoint and rename keys (backbone.* -> cnn.*)
    state_dict = torch.load(CKPT_PATH, map_location=DEVICE)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("backbone."):
            new_k = k.replace("backbone.", "cnn.")
        else:
            new_k = k
        new_state_dict[new_k] = v

    # load with relaxed matching
    model.load_state_dict(new_state_dict, strict=False)

    model.eval()

    # target layer for GradCAM (last conv block of ResNet50)
    target_layer = model.cnn.layer4[-1].conv3
    cam_gen = GradCAM(model, target_layer)

    for cat,(seg_id,frame_idx_list) in SEGMENTS.items():
        frames = load_segment(seg_id, frame_idx_list).to(DEVICE)  # [T,C,H,W]
        for i,frame in enumerate(frames):
            frame_in = frame.unsqueeze(0).to(DEVICE)  # [1,C,H,W]
            cam = cam_gen(frame_in)
            overlay_and_save(frame, cam, os.path.join(SAVE_DIR, f"{cat}_{seg_id}_f{i}.png"))
        print(f"Saved GradCAMs for {cat} {seg_id}")
