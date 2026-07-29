# %%
# preproc_benchmark.py
# Run: python preproc_benchmark.py
# Requirements: pip install torch torchvision opencv-python timm
# Place kan_convolutional/ folder and kan_l.pth in the same directory

import cv2, torch, time, numpy as np
import torch.fft
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
VIDEO_PATH  = "real.mp4"
N_WARMUP    = 20
N_TRIALS    = 50
TARGET_SIZE = 256

from kan_convolutional.KANLinear import KANLinear


class SimpleLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.kan = KANLinear(
            in_features=256 * 256, out_features=2,
            grid_size=10, spline_order=3,
            scale_noise=0.01, scale_base=1, scale_spline=1,
            base_activation=nn.SiLU, grid_eps=0.02, grid_range=[0, 1]
        )
        self.flatten = nn.Flatten()
    def forward(self, x):
        x = self.flatten(x)
        x = self.kan(x)
        return F.log_softmax(x, dim=1)


class F3NetProxy(nn.Module):
    """F3Net (ECCV 2020): dual-branch (FAD+LFS) Xception, 46.45M params."""
    def __init__(self):
        super().__init__()
        import timm
        self.branch1 = timm.create_model('xception', pretrained=False, num_classes=0)
        self.branch2 = timm.create_model('xception', pretrained=False, num_classes=0)
        self.attn = nn.MultiheadAttention(embed_dim=2048, num_heads=8, batch_first=True)
        self.head = nn.Linear(2048, 2)

    def forward(self, x):
        f1 = self.branch1(x)
        f2 = self.branch2(x)
        q = f1.unsqueeze(1)
        kv = torch.stack([f1, f2], dim=1)
        out, _ = self.attn(q, kv, kv)
        return self.head(out.squeeze(1))


class FREPGANProxy(nn.Module):
    """FREPGAN (CVPR 2022): PatchGAN discriminator + freq-domain residual, 29.14M params."""
    def __init__(self):
        super().__init__()
        def blk(ci, co, s=2):
            return nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(ci, co, 4, s, 1)),
                nn.LeakyReLU(0.2, inplace=True)
            )
        self.net = nn.Sequential(
            blk(1, 64, 2), blk(64, 128, 2), blk(128, 256, 2),
            blk(256, 512, 2), blk(512, 512, 2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.freq_branch = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten()
        )
        self.head = nn.Linear(512 + 256, 2)

    def forward(self, x):
        sp = self.net(x)
        fr = self.freq_branch(x)
        return self.head(torch.cat([sp, fr], dim=1))


class ThreeDCNNProxy(nn.Module):
    """3DCNN (IJCAI 2021, Temporal Dropout): 2.69M params, 16-frame clips at 112x112."""
    def __init__(self):
        super().__init__()
        def blk3d(ci, co):
            return nn.Sequential(
                nn.Conv3d(ci, co, kernel_size=(3,3,3), padding=1),
                nn.BatchNorm3d(co), nn.ReLU(inplace=True),
                nn.MaxPool3d((1,2,2))
            )
        self.net = nn.Sequential(
            blk3d(3, 64), blk3d(64, 128), blk3d(128, 256),
            blk3d(256, 256), blk3d(256, 512),
            nn.AdaptiveAvgPool3d(1), nn.Flatten(),
        )
        self.drop = nn.Dropout(p=0.5)
        self.head = nn.Linear(512, 2)

    def forward(self, x):
        f = self.net(x)
        f = self.drop(f)
        return self.head(f)


class ENSCNNProxy(nn.Module):
    """ENSCNN [7]: ensemble of 3 ResNet-34-class CNNs on DFT input, 19.34M params total."""
    def __init__(self):
        super().__init__()
        import timm
        self.cnns = nn.ModuleList([
            timm.create_model('resnet34', pretrained=False, num_classes=0, in_chans=1)
            for _ in range(3)
        ])
        self.head = nn.Linear(512 * 3, 2)

    def forward(self, x_list):
        feats = [cnn(x) for cnn, x in zip(self.cnns, x_list)]
        return self.head(torch.cat(feats, dim=1))


class GazeNetProxy(nn.Module):
    """GazeNet [11]: gaze-guided spatial detector, 13.53M params, 3 input streams."""
    def __init__(self):
        super().__init__()
        self.gaze_ext = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, 1, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4), nn.Flatten(),
        )
        self.gaze_fc = nn.Linear(8192, 256)
        self.clf_conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(4), nn.Flatten()
        )
        self.clf_fc = nn.Sequential(
            nn.Linear(4096 + 256 * 2, 512), nn.ReLU(),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        face, el, er = x[0:1], x[1:2], x[2:3]
        gz_l = F.relu(self.gaze_fc(self.gaze_ext(el)))
        gz_r = F.relu(self.gaze_fc(self.gaze_ext(er)))
        cf = self.clf_conv(face)
        return self.clf_fc(torch.cat([cf, gz_l, gz_r], dim=1))


class XceptionProxy(nn.Module):
    """Xception [16] deepfake detector, 22.85M params, direct timm load."""
    def __init__(self):
        super().__init__()
        import timm
        self.model = timm.create_model('xception', pretrained=False, num_classes=2)
    def forward(self, x):
        return self.model(x)


class SingleKANProxy(nn.Module):
    """SINGLE-KAN [49]: 13.11M params, KAN classifier on flattened RGB input."""
    def __init__(self):
        super().__init__()
        self.kan = KANLinear(
            in_features=256 * 256 * 3, out_features=2,
            grid_size=10, spline_order=3,
            scale_noise=0.01, scale_base=1, scale_spline=1,
            base_activation=nn.SiLU, grid_eps=0.02, grid_range=[0, 1]
        )
        self.flatten = nn.Flatten()
    def forward(self, x):
        x = self.flatten(x)
        x = self.kan(x)
        return F.log_softmax(x, dim=1)


class KolRGBProxy(nn.Module):
    """KOL-3-FREQ RGB ablation: KAN on flattened RGB input, 15.47M params."""
    def __init__(self):
        super().__init__()
        self.kan = KANLinear(
            in_features=256 * 256 * 3, out_features=2,
            grid_size=10, spline_order=3,
            scale_noise=0.01, scale_base=1, scale_spline=1,
            base_activation=nn.SiLU, grid_eps=0.02, grid_range=[0, 1]
        )
        self.flatten = nn.Flatten()
    def forward(self, x):
        x = self.flatten(x)
        x = self.kan(x)
        return F.log_softmax(x, dim=1)


def load_frames(path, n, size=TARGET_SIZE):
    cap, frames = cv2.VideoCapture(path), []
    while len(frames) < n:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        frames.append(cv2.resize(frame, (size, size)))
    cap.release()
    return frames


def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def benchmark_two_stage(preproc_fn, infer_fn, frames, n_warmup=N_WARMUP, n_trials=N_TRIALS):
    for f in frames[:n_warmup]:
        t = preproc_fn(f)
        if t is not None:
            infer_fn(t)
    sync()
    preproc_times, infer_times = [], []
    trial_count = 0
    idx = n_warmup
    while trial_count < n_trials:
        f = frames[idx % len(frames)]
        idx += 1
        t0 = time.perf_counter()
        tensor = preproc_fn(f)
        sync()
        preproc_times.append((time.perf_counter() - t0) * 1000)
        if tensor is None:          # 3DCNN: clip not yet full
            preproc_times.pop()
            continue
        t1 = time.perf_counter()
        infer_fn(tensor)
        sync()
        infer_times.append((time.perf_counter() - t1) * 1000)
        trial_count += 1
    return (np.mean(preproc_times), np.std(preproc_times),
            np.mean(infer_times), np.std(infer_times))


_to_tensor = transforms.ToTensor()
_clip_buf = []


def preproc_kol_dft(f):
    gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    t = torch.from_numpy(gray).unsqueeze(0).to(DEVICE)
    mag = torch.log2(torch.abs(torch.fft.fft2(t)) + 1e-8)
    mag = (mag - 0.5) / 0.5
    return mag.unsqueeze(0)


def preproc_kol_rgb(f):
    rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    return _to_tensor(rgb).unsqueeze(0).to(DEVICE)


def preproc_f3net(f):
    rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    return tf(rgb).unsqueeze(0).to(DEVICE)


def preproc_frepgan(f):
    gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    t = torch.from_numpy(gray).unsqueeze(0).to(DEVICE)
    mag = torch.abs(torch.fft.fft2(t))
    return mag.unsqueeze(0)


def preproc_3dcnn(f):
    global _clip_buf
    rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    small = cv2.resize(rgb, (112, 112)).astype(np.float32) / 255.0
    t = torch.from_numpy(small).permute(2, 0, 1)
    _clip_buf.append(t)
    if len(_clip_buf) == 16:
        clip = torch.stack(_clip_buf, dim=1).to(DEVICE)  # (3,16,112,112)
        clip = (clip - 0.5) / 0.5
        _clip_buf = []
        return clip.unsqueeze(0)
    return None


def preproc_enscnn(f):
    gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    t = torch.from_numpy(gray).unsqueeze(0).to(DEVICE)
    base = torch.log2(torch.abs(torch.fft.fft2(t)) + 1e-8)
    return [((base - 0.5) / 0.5).unsqueeze(0) for _ in range(3)]


def preproc_xception(f):
    rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    r = cv2.resize(rgb, (299, 299))
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    return tf(r).unsqueeze(0).to(DEVICE)


def preproc_gazenet(f):
    rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    face = cv2.resize(rgb, (224, 224))
    h, w = face.shape[:2]
    el = cv2.resize(face[:int(h*0.4), :w//2], (224, 224))
    er = cv2.resize(face[:int(h*0.4), w//2:], (224, 224))
    def to_t(img):
        return torch.from_numpy(img.astype(np.float32)/255.).permute(2,0,1).to(DEVICE)
    return torch.stack([to_t(face), to_t(el), to_t(er)])


def preproc_single_kan(f):
    rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    return _to_tensor(rgb).unsqueeze(0).to(DEVICE)


def make_infer(model):
    model.eval().to(DEVICE)
    def infer(tensor):
        with torch.no_grad():
            _ = model(tensor)
    return infer


REGISTRY = [
    ("KOL-3-FREQ (DFT) — Ours", preproc_kol_dft, None, "DFT"),
    ("KOL-3-FREQ (RGB ablation)", preproc_kol_rgb, KolRGBProxy, "RGB"),
    ("F3NET [8]", preproc_f3net, F3NetProxy, "DFT"),
    ("FREPGAN [4]", preproc_frepgan, FREPGANProxy, "DFT"),
    ("3DCNN [5]", preproc_3dcnn, ThreeDCNNProxy, "DFT"),
    ("ENSCNN [7]", preproc_enscnn, ENSCNNProxy, "DFT"),
    ("XCEPTION [16]", preproc_xception, XceptionProxy, "RGB"),
    ("GAZENET [11]", preproc_gazenet, GazeNetProxy, "RGB"),
    ("SINGLE-KAN [49]", preproc_single_kan, SingleKANProxy, "RGB"),
]

if __name__ == "__main__":
    print(f"Device : {DEVICE}")
    print(f"Trials : {N_TRIALS}  |  Warmup: {N_WARMUP}")
    print(f"Loading frames from '{VIDEO_PATH}' ...")
    n_needed = N_TRIALS + N_WARMUP + 200   # extra for 3DCNN clip accumulation
    frames = load_frames(VIDEO_PATH, n_needed)
    print(f"Loaded {len(frames)} frames.\n")

    print("Loading kan_l.pth for KOL-3-FREQ (DFT) ...")
    kol_dft_model = SimpleLinear().to(DEVICE)
    kol_dft_model.load_state_dict(torch.load("kan_l.pth", map_location=DEVICE))

    HDR = (f"\n{'Model':<35} {'Input':>8} {'Preproc (ms)':>17} "
           f"{'Inference (ms)':>17} {'Total (ms)':>12}")
    SEP = "-" * 95
    print(HDR)
    print(SEP)

    results = []
    for label, preproc_fn, model_cls, inp_type in REGISTRY:
        _clip_buf.clear()

        model = kol_dft_model if model_cls is None else model_cls().to(DEVICE)

        infer_fn = make_infer(model)
        p_m, p_s, i_m, i_s = benchmark_two_stage(preproc_fn, infer_fn, frames)
        total = p_m + i_m

        results.append((label, inp_type, p_m, p_s, i_m, i_s, total))
        print(f"{label:<35} {inp_type:>8} "
              f"{p_m:>8.3f}±{p_s:<6.3f} "
              f"{i_m:>8.3f}±{i_s:<6.3f} "
              f"{total:>10.3f}")

        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    print(SEP)

    kol_dft_total = next(r[6] for r in results if "DFT) — Ours" in r[0])
    print("\n── Key ratios (vs KOL-3-FREQ DFT) ────────────────────────────")
    for label, inp_type, p_m, p_s, i_m, i_s, total in results:
        if "Ours" not in label:
            print(f"  {label:<33}: {total/kol_dft_total:.2f}× slower end-to-end")

    import csv
    with open("benchmark_results.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Model", "Input", "Preproc_mean_ms", "Preproc_std_ms",
                    "Inference_mean_ms", "Inference_std_ms", "Total_ms"])
        for r in results:
            w.writerow([r[0], r[1], f"{r[2]:.3f}", f"{r[3]:.3f}",
                        f"{r[4]:.3f}", f"{r[5]:.3f}", f"{r[6]:.3f}"])
    print("\nSaved: benchmark_results.csv")