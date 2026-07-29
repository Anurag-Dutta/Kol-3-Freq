import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# CONFIG
FF_ROOT = r"C:\Users\Anurag Dutta\Desktop\Kol-3-Freq\FF++"  # change this
TEST_RATIO = 0.2
SEED = 42
J = 128
FRAME_LIMIT = 10
IMG_SIZE = (256, 256)

random.seed(SEED)
np.random.seed(SEED)


def get_videos(folder):
    exts = {".mp4", ".avi", ".mov", ".mkv"}
    return [str(p) for p in Path(folder).rglob("*") if p.suffix.lower() in exts]


def test_split(paths):
    paths = sorted(paths)
    random.shuffle(paths)
    n = max(1, int(len(paths) * TEST_RATIO))
    return paths[-n:]


def sample_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total <= 0:
        cap.release()
        return []

    indices = np.linspace(0, total - 1, min(FRAME_LIMIT, total), dtype=int)
    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, IMG_SIZE)
            frames.append(gray.astype(np.float32))

    cap.release()
    return frames


def annular_features(frame):
    H, W = frame.shape
    F = np.fft.fftshift(np.fft.fft2(frame))
    log_mag = np.log2(np.abs(F) + 1e-8)

    cy, cx = H // 2, W // 2
    y, x = np.ogrid[:H, :W]

    r_map = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    r_max = np.sqrt(cx ** 2 + cy ** 2)
    bw = r_max / J

    feats = np.zeros(J, dtype=np.float32)
    for j in range(J):
        mask = (r_map >= j * bw) & (r_map < (j + 1) * bw)
        if mask.sum() > 0:
            feats[j] = log_mag[mask].mean()

    return feats


def video_feature(path):
    frames = sample_frames(path)
    if not frames:
        return None
    return np.stack([annular_features(f) for f in frames]).mean(axis=0)


def extract_all(paths, label):
    feats = []
    for i, vp in enumerate(paths):
        f = video_feature(vp)
        if f is not None:
            feats.append(f)
        if (i + 1) % 20 == 0:
            print(f"[{label}] {i+1}/{len(paths)} done")
    return np.array(feats)


def pixel_mean(paths, n=50):
    sample = random.sample(sorted(paths), min(n, len(paths)))
    pixels = []
    for vp in sample:
        for f in sample_frames(vp)[:3]:
            pixels.append(f.flatten() / 255.0)
    return np.mean(pixels, axis=0) if pixels else None


# COLLECT VIDEOS AND SPLIT
fake_videos = get_videos(os.path.join(FF_ROOT, "Deepfakes"))
real_videos = get_videos(os.path.join(FF_ROOT, "original"))
print(f"Found: {len(fake_videos)} fake, {len(real_videos)} real")

fake_test = test_split(fake_videos)
real_test = test_split(real_videos)
print(f"Test: {len(fake_test)} fake, {len(real_test)} real")

# FEATURE EXTRACTION
print("Extracting fake features...")
fake_feats = extract_all(fake_test, "fake")

print("Extracting real features...")
real_feats = extract_all(real_test, "real")

np.save("fake_features.npy", fake_feats)
np.save("real_features.npy", real_feats)
print(f"Saved. Shapes: fake={fake_feats.shape}, real={real_feats.shape}")

# SPECTRAL DISCRIMINABILITY
Dj = np.abs(fake_feats.mean(axis=0) - real_feats.mean(axis=0))
np.save("Dj_values.npy", Dj)
spectral_disc = float(np.sum(Dj ** 2))

# SPATIAL DISCRIMINABILITY
mu_f = pixel_mean(fake_test)
mu_r = pixel_mean(real_test)
spatial_disc = float(np.mean((mu_f - mu_r) ** 2)) if mu_f is not None else 0.0

print("\n── RESULTS ──────────────────────────")
print(f"sum(Dj^2) [spectral] = {spectral_disc:.6f}")
print(f"pixel MSE [spatial]  = {spatial_disc:.6f}")
print(f"Ratio                = {spectral_disc / max(spatial_disc, 1e-10):.1f}x")
print(f"Max Dj = {Dj.max():.4f} at band j = {int(Dj.argmax())}")
print(f"Top-5 bands: {np.argsort(Dj)[::-1][:5].tolist()}")

# PLOT
threshold = np.percentile(Dj, 75)
colors = ["#E07B54" if Dj[j] >= threshold else "#5B8DB8" for j in range(J)]

plt.figure(figsize=(10, 4))
plt.bar(range(J), Dj, color=colors, width=1.0)
plt.axvspan(J // 2 - 8, J // 2 + 8, color="yellow", alpha=0.2, label="Aliasing region")
plt.xlabel("Band index j (low → high frequency)")
plt.ylabel("$D_j$")
plt.title("Band Discriminability $D_j$ across 128 Annular Frequency Bins (FF++)")
plt.legend()
plt.tight_layout()
plt.savefig("dj_discriminability.png", dpi=300)
plt.show()

print("Saved: dj_discriminability.png")