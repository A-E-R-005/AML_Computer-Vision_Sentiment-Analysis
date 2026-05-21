import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
import time

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

data_train  = np.load('face_alignment_training_data.npz', allow_pickle=True)
images_train = data_train['images']
pts_train    = data_train['points']

data_val    = np.load('face_alignment_validation_data.npz', allow_pickle=True)
images_val  = data_val['images']
pts_val     = data_val['points']

data_test   = np.load('face_alignment_test_data.npz', allow_pickle=True)
images_test = data_test['images']

print(f"Train: {images_train.shape}, {pts_train.shape}")
print(f"Val:   {images_val.shape},   {pts_val.shape}")
print(f"Test:  {images_test.shape}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

POINT_NAMES   = ['Left eye', 'Right eye', 'Nose',
                 'Left mouth', 'Right mouth']
POINT_COLOURS = ['red', 'blue', 'green', 'orange', 'purple']


def save_as_csv(points, location='.'):
    assert points.shape[0] == 554
    assert np.prod(points.shape[1:]) == 5 * 2
    np.savetxt(location + '/results_task2.csv',
               np.reshape(points, (points.shape[0], -1)), delimiter=',')


def point_error(pred_pts, true_pts):
    """Euclidean distance per point per image. Returns (n_images, 5)."""
    diff = pred_pts - true_pts
    return np.sqrt((diff ** 2).sum(axis=2))


def mean_error(pred_pts, true_pts):
    return point_error(pred_pts, true_pts).mean()


def pck(pred_pts, true_pts, threshold=10.0):
    """Proportion of correct keypoints within threshold pixels."""
    return (point_error(pred_pts, true_pts) < threshold).mean()


def plot_cumulative_error(errors_dict, title, filename):
    """
    errors_dict: {'method_name': errors_array (n_images, 5)}
    Plots cumulative error distribution for each method.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, errors in errors_dict.items():
        flat_errors = errors.flatten()
        sorted_e    = np.sort(flat_errors)
        cumulative  = np.arange(1, len(sorted_e)+1) / len(sorted_e)
        ax.plot(sorted_e, cumulative, label=name, linewidth=2)
    ax.set_xlabel('Euclidean Error (pixels)', fontsize=12)
    ax.set_ylabel('Proportion of Keypoints', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 60)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved: {filename}")


def plot_boxplot(errors_dict, title, filename):
    """Boxplot comparison of per-point errors across methods."""
    fig, axes = plt.subplots(1, len(errors_dict),
                             figsize=(5*len(errors_dict), 5),
                             sharey=True)
    if len(errors_dict) == 1:
        axes = [axes]
    for ax, (name, errors) in zip(axes, errors_dict.items()):
        ax.boxplot([errors[:, p] for p in range(5)],
                   tick_labels=POINT_NAMES, patch_artist=True)
        ax.set_title(name, fontsize=11)
        ax.set_ylabel('Error (pixels)', fontsize=10)
        ax.tick_params(axis='x', rotation=30)
        ax.grid(True, alpha=0.3)
    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved: {filename}")


def visualise_predictions(images, true_pts, pred_pts, n=6,
                          title='', filename='predictions.png'):
    """Show true (+) vs predicted (x) landmarks on sample images."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    indices = np.random.choice(len(images), min(n, len(images)),
                               replace=False)
    for ax, idx in zip(axes, indices):
        ax.imshow(images[idx])
        for p in range(5):
            ax.plot(true_pts[idx, p, 0], true_pts[idx, p, 1],
                    '+', color=POINT_COLOURS[p],
                    markersize=12, markeredgewidth=2,
                    label=f'{POINT_NAMES[p]} true')
            ax.plot(pred_pts[idx, p, 0], pred_pts[idx, p, 1],
                    'x', color=POINT_COLOURS[p],
                    markersize=12, markeredgewidth=2,
                    label=f'{POINT_NAMES[p]} pred')
        err = point_error(pred_pts[idx:idx+1], true_pts[idx:idx+1]).mean()
        ax.set_title(f'Image {idx} | Mean error: {err:.1f}px', fontsize=9)
        ax.axis('off')
    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved: {filename}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess_images(images):
    """
    Normalise pixel values from [0, 255] to [0, 1].
    Convert RGB to grayscale for HOG features.
    Keep colour for CNN.
    """
    images_float = images.astype(np.float32) / 255.0
    # Grayscale: standard luminance weights
    images_gray  = (0.2989 * images_float[:, :, :, 0] +
                    0.5870 * images_float[:, :, :, 1] +
                    0.1140 * images_float[:, :, :, 2])
    return images_float, images_gray


images_train_f, images_train_gray = preprocess_images(images_train)
images_val_f,   images_val_gray   = preprocess_images(images_val)
images_test_f,  images_test_gray  = preprocess_images(images_test)

# Flatten points to (n_images, 10) for regression targets
pts_train_flat = pts_train.reshape(len(pts_train), -1)
pts_val_flat   = pts_val.reshape(len(pts_val), -1)

print(f"Points flattened: {pts_train_flat.shape}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: APPROACH 1 — MEAN FACE BASELINE
# ═══════════════════════════════════════════════════════════════════════════════

print("\n--- Approach 1: Mean Face Baseline ---")

mean_pts    = pts_train.mean(axis=0)  # shape (5, 2)
baseline_val_preds = np.tile(mean_pts, (len(images_val), 1, 1))

baseline_errors = point_error(baseline_val_preds, pts_val)
baseline_mean   = baseline_errors.mean()
baseline_pck10  = pck(baseline_val_preds, pts_val, threshold=10)

print(f"Mean error:      {baseline_mean:.2f} pixels")
print(f"PCK@10px:        {baseline_pck10*100:.1f}%")
print(f"Per-point errors:")
for p in range(5):
    print(f"  {POINT_NAMES[p]:<15}: {baseline_errors[:, p].mean():.2f}px")

visualise_predictions(images_val, pts_val, baseline_val_preds,
                      title='Mean Face Baseline Predictions',
                      filename='baseline_predictions.png')


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: APPROACH 2 — HOG FEATURES + LINEAR REGRESSION
# ═══════════════════════════════════════════════════════════════════════════════

print("\n--- Approach 2: HOG + Ridge Regression ---")

def extract_hog_features(images_gray):
    """
    Extract HOG descriptors from grayscale images.
    HOG captures edge orientations in local regions.
    Invariant to brightness changes and small rotations.
    pixels_per_cell: size of each cell in the histogram grid
    cells_per_block: normalisation block size
    orientations: number of orientation bins
    """
    features = []
    for img in images_gray:
        feat = hog(img,
                   orientations=9,
                   pixels_per_cell=(16, 16),
                   cells_per_block=(2, 2),
                   block_norm='L2-Hys',
                   visualize=False)
        features.append(feat)
    return np.array(features)


print("Extracting HOG features (train)...")
t0 = time.time()
X_train_hog = extract_hog_features(images_train_gray)
print(f"  Done in {time.time()-t0:.1f}s | Shape: {X_train_hog.shape}")

print("Extracting HOG features (val)...")
t0 = time.time()
X_val_hog   = extract_hog_features(images_val_gray)
print(f"  Done in {time.time()-t0:.1f}s")

print("Extracting HOG features (test)...")
X_test_hog  = extract_hog_features(images_test_gray)

# Standardise features
scaler_hog      = StandardScaler()
X_train_hog_sc  = scaler_hog.fit_transform(X_train_hog)
X_val_hog_sc    = scaler_hog.transform(X_val_hog)
X_test_hog_sc   = scaler_hog.transform(X_test_hog)

# Ridge regression — L2 regularisation prevents overfitting on HOG features
# Alpha sweep to find best regularisation strength
print("\n--- Ridge alpha sweep ---")
print(f"{'Alpha':<10} {'Mean error':>12} {'PCK@10px':>10}")
print("-" * 34)

best_hog_error = np.inf
best_hog_alpha = 1.0
best_hog_model = None

for alpha in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_hog_sc, pts_train_flat)
    preds_flat = ridge.predict(X_val_hog_sc)
    preds      = preds_flat.reshape(-1, 5, 2)
    err        = point_error(preds, pts_val).mean()
    p10        = pck(preds, pts_val, threshold=10)
    print(f"alpha={alpha:<6} {err:>12.2f}px {p10*100:>9.1f}%")
    if err < best_hog_error:
        best_hog_error = err
        best_hog_alpha = alpha
        best_hog_model = ridge

print(f"\nBest alpha: {best_hog_alpha} → {best_hog_error:.2f}px")

hog_val_preds = best_hog_model.predict(X_val_hog_sc).reshape(-1, 5, 2)
hog_errors    = point_error(hog_val_preds, pts_val)
hog_pck10     = pck(hog_val_preds, pts_val, threshold=10)

print(f"HOG mean error:  {hog_errors.mean():.2f}px")
print(f"HOG PCK@10px:    {hog_pck10*100:.1f}%")
print(f"Per-point errors:")
for p in range(5):
    print(f"  {POINT_NAMES[p]:<15}: {hog_errors[:, p].mean():.2f}px")

visualise_predictions(images_val, pts_val, hog_val_preds,
                      title='HOG + Ridge Regression Predictions',
                      filename='hog_predictions.png')


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: COMPARISON PLOTS (BASELINE vs HOG)
# ═══════════════════════════════════════════════════════════════════════════════

errors_so_far = {
    'Mean baseline':    baseline_errors,
    'HOG + Ridge':      hog_errors,
}

plot_cumulative_error(errors_so_far,
                      'Cumulative Error Distribution',
                      'cumulative_error.png')

plot_boxplot(errors_so_far,
             'Per-Point Error Comparison',
             'boxplot_comparison.png')

print(f"\n--- Summary so far ---")
print(f"{'Method':<20} {'Mean error':>12} {'PCK@10px':>10}")
print("-" * 44)
print(f"{'Mean baseline':<20} {baseline_mean:>12.2f}px "
      f"{baseline_pck10*100:>9.1f}%")
print(f"{'HOG + Ridge':<20} {hog_errors.mean():>12.2f}px "
      f"{hog_pck10*100:>9.1f}%")

# ── EXTENDED ALPHA SWEEP ──────────────────────────────────────────────────────
print("\n--- Extended alpha sweep ---")
print(f"{'Alpha':<12} {'Mean error':>12} {'PCK@10px':>10}")
print("-" * 36)

for alpha in [1000, 2000, 5000, 10000, 50000]:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_hog_sc, pts_train_flat)
    preds = ridge.predict(X_val_hog_sc).reshape(-1, 5, 2)
    err   = point_error(preds, pts_val).mean()
    p10   = pck(preds, pts_val, threshold=10)
    print(f"alpha={alpha:<8} {err:>12.2f}px {p10*100:>9.1f}%")
    if err < best_hog_error:
        best_hog_error = err
        best_hog_alpha = alpha
        best_hog_model = ridge

print(f"\nFinal best alpha: {best_hog_alpha} → {best_hog_error:.2f}px")

# Update hog predictions with best model
hog_val_preds = best_hog_model.predict(X_val_hog_sc).reshape(-1, 5, 2)
hog_errors    = point_error(hog_val_preds, pts_val)
hog_pck10     = pck(hog_val_preds, pts_val, threshold=10)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: APPROACH 3 — CNN
# ═══════════════════════════════════════════════════════════════════════════════

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

print(f"\n--- Approach 3: CNN ---")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")


# ── DATASET ───────────────────────────────────────────────────────────────────

class FaceDataset(Dataset):
    def __init__(self, images, points):
        # images: (N, 256, 256, 3) float32 [0,1]
        # Convert to (N, 3, 256, 256) for PyTorch
        self.images = torch.tensor(
            images.transpose(0, 3, 1, 2), dtype=torch.float32)
        self.points = torch.tensor(
            points.reshape(len(points), -1), dtype=torch.float32)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.points[idx]


class FaceTestDataset(Dataset):
    def __init__(self, images):
        self.images = torch.tensor(
            images.transpose(0, 3, 1, 2), dtype=torch.float32)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]


train_dataset = FaceDataset(images_train_f, pts_train)
val_dataset   = FaceDataset(images_val_f,   pts_val)
test_dataset  = FaceTestDataset(images_test_f)

train_loader  = DataLoader(train_dataset, batch_size=32,
                           shuffle=True,  num_workers=0)
val_loader    = DataLoader(val_dataset,   batch_size=32,
                           shuffle=False, num_workers=0)
test_loader   = DataLoader(test_dataset,  batch_size=32,
                           shuffle=False, num_workers=0)


# ── MODEL ARCHITECTURE ────────────────────────────────────────────────────────

class FaceCNN(nn.Module):
    """
    CNN for face landmark regression.
    Input: (batch, 3, 256, 256) RGB image
    Output: (batch, 10) — 5 landmarks x (x, y)

    Architecture:
    - 4 convolutional blocks with max pooling
    - Each block: Conv → BatchNorm → ReLU → MaxPool
    - Fully connected layers to predict 10 coordinates
    - Dropout for regularisation
    """
    def __init__(self):
        super(FaceCNN, self).__init__()

        self.conv_blocks = nn.Sequential(
            # Block 1: 256x256 → 128x128
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2: 128x128 → 64x64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3: 64x64 → 32x32
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 4: 32x32 → 16x16
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # After 4 max pools: 256 → 16x16
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 16 * 16, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.fc(x)
        return x


model     = FaceCNN().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=5, factor=0.5)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")


# ── TRAINING ──────────────────────────────────────────────────────────────────

EPOCHS = 30
best_val_error  = np.inf
best_cnn_state  = None
train_losses    = []
val_errors      = []

print(f"\nTraining for {EPOCHS} epochs...")
print(f"{'Epoch':>6} {'Train loss':>12} {'Val error':>10} {'PCK@10':>8}")
print("-" * 40)

t_start = time.time()

for epoch in range(EPOCHS):
    # Training
    model.train()
    train_loss = 0.0
    for batch_imgs, batch_pts in train_loader:
        batch_imgs = batch_imgs.to(DEVICE)
        batch_pts  = batch_pts.to(DEVICE)
        optimizer.zero_grad()
        preds = model(batch_imgs)
        loss  = criterion(preds, batch_pts)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(batch_imgs)
    train_loss /= len(train_dataset)

    # Validation
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch_imgs, _ in val_loader:
            preds = model(batch_imgs.to(DEVICE))
            all_preds.append(preds.cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=0).reshape(-1, 5, 2)
    val_error = point_error(all_preds, pts_val).mean()
    val_pck10 = pck(all_preds, pts_val, threshold=10)

    train_losses.append(train_loss)
    val_errors.append(val_error)

    scheduler.step(val_error)

    if val_error < best_val_error:
        best_val_error = val_error
        best_cnn_state = {k: v.clone()
                          for k, v in model.state_dict().items()}

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"{epoch+1:>6} {train_loss:>12.4f} "
              f"{val_error:>10.2f}px {val_pck10*100:>7.1f}%")

t_end = time.time()
print(f"\nTraining time: {(t_end-t_start)/60:.1f} minutes on {DEVICE}")


# ── TRAINING CURVE ────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(train_losses, label='Train loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('MSE Loss')
axes[0].set_title('Training Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(val_errors, label='Val error', color='orange')
axes[1].axhline(y=best_val_error, color='red', linestyle='--',
                label=f'Best: {best_val_error:.2f}px')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Mean Error (px)')
axes[1].set_title('Validation Error')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('CNN Training Curves', fontsize=13)
plt.tight_layout()
plt.savefig('cnn_training_curves.png', dpi=150)
plt.close()
print("Saved: cnn_training_curves.png")


# ── EVALUATE BEST CNN ─────────────────────────────────────────────────────────

model.load_state_dict(best_cnn_state)
model.eval()

all_preds = []
with torch.no_grad():
    for batch_imgs, _ in val_loader:
        preds = model(batch_imgs.to(DEVICE))
        all_preds.append(preds.cpu().numpy())

cnn_val_preds = np.concatenate(all_preds, axis=0).reshape(-1, 5, 2)
cnn_errors    = point_error(cnn_val_preds, pts_val)
cnn_pck10     = pck(cnn_val_preds, pts_val, threshold=10)

print(f"\n--- CNN Results ---")
print(f"Best val error:  {best_val_error:.2f}px")
print(f"PCK@10px:        {cnn_pck10*100:.1f}%")
print(f"Per-point errors:")
for p in range(5):
    print(f"  {POINT_NAMES[p]:<15}: {cnn_errors[:, p].mean():.2f}px")

visualise_predictions(images_val, pts_val, cnn_val_preds,
                      title='CNN Predictions',
                      filename='cnn_predictions.png')


# ── FULL COMPARISON ───────────────────────────────────────────────────────────

all_errors = {
    'Mean baseline': baseline_errors,
    'HOG + Ridge':   hog_errors,
    'CNN':           cnn_errors,
}

plot_cumulative_error(all_errors,
                      'Cumulative Error Distribution — All Methods',
                      'cumulative_error_all.png')

plot_boxplot(all_errors,
             'Per-Point Error — All Methods',
             'boxplot_all.png')

print(f"\n{'Method':<20} {'Mean error':>12} {'PCK@10px':>10}")
print("-" * 44)
for name, errors in all_errors.items():
    print(f"{name:<20} {errors.mean():>12.2f}px "
          f"{(errors < 10).mean()*100:>9.1f}%")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: TEST PREDICTIONS — BEST MODEL
# ═══════════════════════════════════════════════════════════════════════════════

# Use CNN (best performing model)
model.eval()
all_test_preds = []
with torch.no_grad():
    for batch_imgs in test_loader:
        preds = model(batch_imgs.to(DEVICE))
        all_test_preds.append(preds.cpu().numpy())

test_preds = np.concatenate(all_test_preds, axis=0).reshape(-1, 5, 2)

print(f"\n--- Test Predictions ---")
print(f"Shape: {test_preds.shape}")
print(f"x range: {test_preds[:,:,0].min():.1f} to {test_preds[:,:,0].max():.1f}")
print(f"y range: {test_preds[:,:,1].min():.1f} to {test_preds[:,:,1].max():.1f}")

assert test_preds.shape == (554, 5, 2)
save_as_csv(test_preds)
print("Saved: results_task2.csv")