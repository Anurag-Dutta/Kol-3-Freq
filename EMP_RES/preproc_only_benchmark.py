# %%
import cv2, torch, time, numpy as np
import torch.fft
from torchvision import transforms
import warnings
warnings.filterwarnings("ignore")

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
VIDEO_PATH  = "real.mp4"
N_WARMUP    = 100
N_TRIALS    = 1000
TARGET_SIZE = 256


def load_frames(path, n=N_TRIALS + N_WARMUP, size=TARGET_SIZE):
    cap = cv2.VideoCapture(path)
    frames = []
    while len(frames) < n:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        frame = cv2.resize(frame, (size, size))
        frames.append(frame)
    cap.release()
    return frames


def benchmark(fn, frames, n_warmup=N_WARMUP, n_trials=N_TRIALS):
    for f in frames[:n_warmup]:
        fn(f)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    times = []
    for f in frames[n_warmup:n_warmup + n_trials]:
        t0 = time.perf_counter()
        fn(f)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return np.mean(times), np.std(times)


# 1. KOL-3-FREQ (DFT) — grayscale → fft2 → log-magnitude
def preproc_kol_dft(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    t = torch.from_numpy(gray).unsqueeze(0).to(DEVICE)
    F = torch.fft.fft2(t)
    _ = torch.log2(torch.abs(F) + 1e-8)


# 2. KOL-3-FREQ (RGB ablation) — BGR→RGB → tensor
_to_tensor = transforms.ToTensor()
def preproc_kol_rgb(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    _ = _to_tensor(rgb).unsqueeze(0).to(DEVICE)


# 3. F3NET — block-wise 8x8 DCT (frequency branch) + normalize
def preproc_f3net(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    t = torch.from_numpy(gray).to(DEVICE)
    blocks = t.unfold(0, 8, 8).unfold(1, 8, 8)   # (32,32,8,8)
    _ = torch.fft.fft2(blocks).real               # block-DCT via FFT
    _ = (t - 0.5) / 0.5


# 4. FREPGAN — fft2 + magnitude (feeds raw spectrum to discriminator)
def preproc_frepgan(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    t = torch.from_numpy(gray).unsqueeze(0).to(DEVICE)
    F = torch.fft.fft2(t)
    _ = torch.abs(F)


# 5. 3DCNN — resize to 112x112, buffer 16 frames, stack into volume
_clip_buffer = []
def preproc_3dcnn(frame_bgr):
    global _clip_buffer
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    small = cv2.resize(rgb, (112, 112)).astype(np.float32) / 255.0
    t = torch.from_numpy(small).permute(2, 0, 1)           # (3, 112, 112)
    _clip_buffer.append(t)
    if len(_clip_buffer) == 16:
        clip = torch.stack(_clip_buffer, dim=1).to(DEVICE)  # (3, 16, 112, 112)
        mean = torch.tensor([0.5, 0.5, 0.5], device=DEVICE).view(3, 1, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5], device=DEVICE).view(3, 1, 1, 1)
        _ = (clip - mean) / std
        _clip_buffer = []


# 6. ENSCNN — 3x DFT (one per ensemble member)
def preproc_enscnn(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    t = torch.from_numpy(gray).unsqueeze(0).to(DEVICE)
    for _ in range(3):
        F = torch.fft.fft2(t)
        _ = torch.log2(torch.abs(F) + 1e-8)


# 7. XCEPTION — BGR→RGB, resize 299x299, normalize
_xception_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])
def preproc_xception(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (299, 299))
    _ = _xception_tf(resized).unsqueeze(0).to(DEVICE)


# 8. GAZENET — face resize 224x224 + two eye ROI crops + interpolate
def preproc_gazenet(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    face = cv2.resize(rgb, (224, 224))
    h, w = face.shape[:2]
    eye_l = cv2.resize(face[:int(h*0.4), :w//2], (112, 56))
    eye_r = cv2.resize(face[:int(h*0.4), w//2:], (112, 56))
    t_face = torch.from_numpy(face.astype(np.float32)/255.).permute(2,0,1).to(DEVICE)
    t_el = torch.from_numpy(eye_l.astype(np.float32)/255.).permute(2,0,1).to(DEVICE)
    t_er = torch.from_numpy(eye_r.astype(np.float32)/255.).permute(2,0,1).to(DEVICE)
    _ = torch.stack([
        t_face,
        torch.nn.functional.interpolate(t_el.unsqueeze(0),(224,224)).squeeze(0),
        torch.nn.functional.interpolate(t_er.unsqueeze(0),(224,224)).squeeze(0),
    ])


# 9. SINGLE-KAN — same as KOL-RGB (raw RGB tensor)
def preproc_single_kan(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    _ = _to_tensor(rgb).unsqueeze(0).to(DEVICE)


MODELS = {
    "KOL-3-FREQ (DFT) — Ours": preproc_kol_dft,
    "KOL-3-FREQ (RGB) — Ours": preproc_kol_rgb,
    "F3NET [8]": preproc_f3net,
    "FREPGAN [4]": preproc_frepgan,
    "3DCNN [5]": preproc_3dcnn,
    "ENSCNN [7]": preproc_enscnn,
    "XCEPTION [16]": preproc_xception,
    "GAZENET [11]": preproc_gazenet,
    "SINGLE-KAN [49]": preproc_single_kan,
}

if __name__ == "__main__":
    print(f"Device : {DEVICE}")
    print(f"Trials : {N_TRIALS}  |  Warmup: {N_WARMUP}")
    print(f"Loading frames from '{VIDEO_PATH}' ...")
    frames = load_frames(VIDEO_PATH)
    print(f"Loaded {len(frames)} frames.\n")
    print(f"{'Model':<35} {'Mean (ms/frame)':>18} {'±Std':>8}")
    print("-" * 64)
    for name, fn in MODELS.items():
        mean_ms, std_ms = benchmark(fn, frames)
        print(f"{name:<35} {mean_ms:>16.3f}   ±{std_ms:>5.3f}")
    print("\nDone. Use Mean (ms/frame) as Preproc. column in Table IV.")