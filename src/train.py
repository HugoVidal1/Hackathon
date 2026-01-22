"""
Training script with Leave-One-Patient-Out (LOPO) cross-validation.

This script enforces proper LOPO evaluation to prevent data leakage and
uses a forced linear classifier head to ensure rich embeddings are learned.
"""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Use EmbeddingModel or EmbeddingTransformer or ContextualEmbeddingTransformer
from models import EmbeddingTransformer, ContextualEmbeddingTransformer, LinearClassifierHead
from utils import (
    extract_date_from_recording_id,
    compute_per_patient_auc,
    aggregate_patient_aucs,
    plot_embeddings_2d,
    print_evaluation_results,
    compute_random_baseline,
)
from config import get_config


# Set random seeds for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class VoiceDataset(Dataset):
    """PyTorch Dataset for voice recordings."""

    def __init__(self, features, labels, patient_ids, recording_ids=None):
        """
        Parameters
        ----------
        features : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        labels : np.ndarray
            Labels of shape (n_samples,).
        patient_ids : np.ndarray
            Patient identifiers of shape (n_samples,).
        recording_ids : np.ndarray, optional
            Recording identifiers of shape (n_samples,).
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.patient_ids = patient_ids
        self.recording_ids = recording_ids if recording_ids is not None else patient_ids

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.features[idx],
            self.labels[idx],
            self.patient_ids[idx],
            self.recording_ids[idx],
        )


class ContextSequenceDataset(Dataset):
    """The new equivalent "VoiceDataset(Dataset)" class for "ContextualEmbeddingTransformer"."""
    
    def __init__(self, features, labels, patient_ids, recording_ids, timestamps, context_len=64):
        self.X = torch.as_tensor(features, dtype=torch.float32)         # [N, D]
        self.y_raw = torch.as_tensor(labels, dtype=torch.long)          # [N]
        self.patient_ids = patient_ids
        self.recording_ids = recording_ids
        self.t = torch.as_tensor(timestamps, dtype=torch.float32)       # [N]
        self.K = int(context_len)

        # 预计算：每个 pid 的 “按时间排序的索引序列”
        self.unique_pids = np.unique(patient_ids)
        self.sorted_idx_per_pid = {}
        self.pos_in_sorted = np.empty(len(patient_ids), dtype=np.int32)

        endpoints = []
        t_np = np.asarray(timestamps, dtype=np.float32)  # 只转一次

        for pid in self.unique_pids:
            idx = np.where(patient_ids == pid)[0]
            sidx = idx[np.argsort(t_np[idx], kind="mergesort")]  # 稳定排序更好
            self.sorted_idx_per_pid[pid] = sidx
            endpoints.extend(sidx.tolist())
            # 记录每个样本在该病人序列中的位置
            self.pos_in_sorted[sidx] = np.arange(len(sidx), dtype=np.int32)

        self.endpoints = np.asarray(endpoints, dtype=np.int64)
        self.labels = self.y_raw[self.endpoints].cpu().numpy()  # 用于 class weight 统计

    def __len__(self):
        return len(self.endpoints)

    def __getitem__(self, i):
        end = int(self.endpoints[i])
        pid = self.patient_ids[end]

        sidx = self.sorted_idx_per_pid[pid]
        pos = int(self.pos_in_sorted[end])

        start = max(0, pos - self.K + 1)
        win = sidx[start:pos + 1]  # 变长

        pad = self.K - len(win)
        if pad > 0:
            pad_idx = np.full((pad,), win[0], dtype=np.int64)
            win_full = np.concatenate([pad_idx, win])
            key_padding_mask = torch.tensor([True]*pad + [False]*len(win), dtype=torch.bool)
        else:
            win_full = win
            key_padding_mask = torch.zeros((self.K,), dtype=torch.bool)

        win_full_t = torch.as_tensor(win_full, dtype=torch.long)

        x_seq = self.X[win_full_t]                       # [K, D]
        t_end = self.t[end]
        delta_t = (t_end - self.t[win_full_t])           # [K]
        y = self.y_raw[end]

        return x_seq, y, self.patient_ids[end], self.recording_ids[end], delta_t, key_padding_mask


def load_data(data_path: str):
    """
    Load and preprocess the dataset.

    Parameters
    ----------
    data_path : str
        Path to the parquet file.

    Returns
    -------
    features : np.ndarray
        Feature matrix.
    labels : np.ndarray
        Binary labels.
    patient_ids : np.ndarray
        Patient identifiers.
    recording_ids : np.ndarray
        Recording identifiers.
    feature_names : list
        List of feature column names.
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)

    # Define metadata columns to exclude from features
    metadata_cols = [
        "recording_id",
        "patient_short_id",
        "label",
    ]

    # Get feature columns
    feature_cols = [col for col in df.columns if col not in metadata_cols]

    # Extract features, labels, patient IDs, and recording IDs
    features = df[feature_cols].values.astype(np.float32)
    labels = df["label"].values.astype(np.int64)
    patient_ids = df["patient_short_id"].values
    recording_ids = df["recording_id"].values

    # Handle missing values
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"Loaded {len(df)} samples from {len(np.unique(patient_ids))} patients")
    print(f"Unique recordings: {len(np.unique(recording_ids))}")
    print(f"Feature dimension: {features.shape[1]}")
    print(f"Label distribution: {np.bincount(labels)}")

    return features, labels, patient_ids, recording_ids, feature_cols


def load_contextual_data(data_path: str):
    """The new equivalent 'load_data()' function for ContextualEmbeddingTransformer."""
    
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)

    # Define metadata columns to exclude from features
    metadata_cols = [
        "recording_id",
        "patient_short_id",
        "label",
        "start_time",
        "end_time",
        "augmentation_dict",
    ]

    # Get feature columns
    feature_cols = [c for c in df.columns if c not in metadata_cols]

    # Extract arrays
    features = df[feature_cols].to_numpy(dtype=np.float32)
    labels = df["label"].to_numpy(dtype=np.int64)
    patient_ids = df["patient_short_id"].to_numpy()
    recording_ids = df["recording_id"].to_numpy()
    start_time = df["start_time"].to_numpy(dtype=np.float32)
    end_time = df["end_time"].to_numpy(dtype=np.float32)

    # Parse date from recording_id (required)
    dates = extract_date_from_recording_id(df["recording_id"])  # pd.Series[datetime]
    day_ord = dates.map(lambda d: d.toordinal()).to_numpy(dtype=np.int32)
    base_day = int(day_ord.min())
    day_index = (day_ord - base_day).astype(np.float32)  # 0..(span_days)

    # Parse augmentation_dict to get time_stretch and "original" flag
    def _safe_parse_aug(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return {}
        if isinstance(v, dict):
            return v
        if isinstance(v, str):
            try:
                return json.loads(v)
            except Exception:
                return {}
        return {}

    aug_series = df["augmentation_dict"].apply(_safe_parse_aug)

    # time_stretch: if missing / null -> 1.0
    def _get_time_stretch(d):
        ts = d.get("time_stretch", None)
        if ts is None or (isinstance(ts, float) and np.isnan(ts)):
            return 1.0
        try:
            return float(ts)
        except Exception:
            return 1.0

    time_stretch = aug_series.apply(_get_time_stretch).to_numpy(dtype=np.float32)

    # Identify "original" rows
    is_original = aug_series.apply(lambda d: d.get("augmentation", None) == "original").to_numpy(dtype=bool)

    # Estimate canonical (pre-augmentation) slice position
    # If time_stretch < 1 (slow), augmented indices are larger; map back via * time_stretch.
    start_est = start_time * time_stretch
    end_est = end_time * time_stretch  # currently not used directly, but helpful if you later want durations

    # Align each (patient, recording_id) row to nearest ORIGINAL start_time (to fix cases like your #4)
    # Build lookup: key -> sorted original start_time list
    keys = list(zip(patient_ids, recording_ids))
    orig_lookup = {}
    for k in set(keys):
        orig_starts = start_time[(patient_ids == k[0]) & (recording_ids == k[1]) & is_original]
        if orig_starts.size > 0:
            orig_lookup[k] = np.sort(orig_starts.astype(np.float32))

    def _nearest_original(k, s_est):
        arr = orig_lookup.get(k, None)
        if arr is None or arr.size == 0:
            return float(s_est)
        idx = int(np.searchsorted(arr, s_est))
        if idx <= 0:
            return float(arr[0])
        if idx >= arr.size:
            return float(arr[-1])
        left = float(arr[idx - 1])
        right = float(arr[idx])
        return left if abs(s_est - left) <= abs(s_est - right) else right

    aligned_start = np.empty_like(start_est, dtype=np.float32)
    for i, k in enumerate(keys):
        aligned_start[i] = _nearest_original(k, float(start_est[i]))

    # Build timestamps:
    # Use day_index to keep magnitude small (good for float32 precision),
    # and use "slice index" (aligned_start) as intra-day ordering.
    # stride must be > max possible intra-day position to avoid overlap across days.
    max_intra = float(np.nanmax(np.maximum(end_time, aligned_start)))  # safe upper bound in your index units
    stride = max(max_intra + 1.0, 1.0)

    timestamps = day_index * np.float32(stride) + aligned_start  # float32

    # Handle missing values in features
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"Loaded {len(df)} samples from {len(np.unique(patient_ids))} patients")
    print(f"Unique recordings: {len(np.unique(recording_ids))}")
    print(f"Feature dimension: {features.shape[1]}")
    print(f"Label distribution: {np.bincount(labels)}")

    # Return timestamps (NOT raw start_time) for contextual modeling
    return features, labels, patient_ids, recording_ids, timestamps, feature_cols


def train_epoch(model, classifier, train_loader, criterion, optimizer, max_grad_norm, device, scheduler=None):
    """Train for one epoch."""
    
    model.train()
    classifier.train()

    total_loss = 0.0
    for features, labels, _, _ in train_loader:  # Added recording_id to unpack
        features = features.to(device)
        labels = labels.to(device)

        # Forward pass
        embeddings = model(features)
        logits = classifier(embeddings)
        loss = criterion(logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        if max_grad_norm and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(classifier.parameters()), max_norm=max_grad_norm
            )
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * features.size(0)

    return total_loss / len(train_loader.dataset)


def train_compatible(model, classifier, train_loader, criterion, optimizer, max_grad_norm, device, scheduler=None):
    """Train for one epoch."""
    
    model.train()
    classifier.train()

    total_loss = 0.0
    n_samples = 0  # 用 labels 的 batch size 统计样本数, 弃用 features.size(0)（features 可能是序列）

    for batch in train_loader:
        # 兼容两种 dataset：
        # 旧：features, labels, pid, rid
        # 新：x_seq, labels, pid, rid, delta_t, mask
        if len(batch) == 4:
            features, labels, _, _ = batch
            features = features.to(device, non_blocking=True)                           # [B, D]
            labels = labels.to(device, non_blocking=True)
            embeddings = model(features)                                                # model 也可兼容 [B, D]
        else:
            x_seq, labels, _, _, delta_t, mask = batch
            x_seq = x_seq.to(device, non_blocking=True)                                 # [B, T, D]
            labels = labels.to(device, non_blocking=True)
            delta_t = delta_t.to(device, non_blocking=True)                             # [B, T]
            mask = mask.to(device, non_blocking=True) if mask is not None else None     # [B, T] or None

            embeddings = model(x_seq, delta_t=delta_t, key_padding_mask=mask)           # [B, emb_dim]

        logits = classifier(embeddings)                                                 # [B, 2]
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()

        if max_grad_norm and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(classifier.parameters()),
                max_norm=max_grad_norm
            )

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        bs = labels.size(0)
        total_loss += loss.item() * bs
        n_samples += bs

    return total_loss / max(n_samples, 1)


def evaluate(model, classifier, data_loader, device):
    """
    Evaluate the model and return predictions.

    Returns
    -------
    embeddings : np.ndarray
        Learned embeddings.
    predictions : np.ndarray
        Predicted probabilities for class 1.
    labels : np.ndarray
        True labels.
    patient_ids : np.ndarray
        Patient identifiers.
    recording_ids : np.ndarray
        Recording identifiers.
    """
    model.eval()
    classifier.eval()

    all_embeddings = []
    all_predictions = []
    all_labels = []
    all_patient_ids = []
    all_recording_ids = []

    with torch.no_grad():
        for features, labels, patient_ids, recording_ids in data_loader:
            features = features.to(device)

            # Get embeddings and predictions
            embeddings = model(features)
            logits = classifier(embeddings)
            probs = torch.softmax(logits, dim=1)

            all_embeddings.append(embeddings.cpu().numpy())
            all_predictions.append(probs[:, 1].cpu().numpy())  # Probability of class 1
            all_labels.append(labels.numpy())
            all_patient_ids.extend(patient_ids)
            all_recording_ids.extend(recording_ids)

    embeddings = np.vstack(all_embeddings)
    predictions = np.concatenate(all_predictions)
    labels = np.concatenate(all_labels)
    patient_ids = np.array(all_patient_ids)
    recording_ids = np.array(all_recording_ids)

    return embeddings, predictions, labels, patient_ids, recording_ids


def evaluate_compatible(model, classifier, data_loader, device):
    model.eval()
    classifier.eval()

    all_embeddings = []
    all_predictions = []
    all_labels = []
    all_patient_ids = []
    all_recording_ids = []

    with torch.no_grad():
        for batch in data_loader:
            # 兼容两种 dataset：
            # Previous: features, labels, patient_ids, recording_ids
            # Contextual: x_seq, labels, patient_ids, recording_ids, delta_t, mask
            if len(batch) == 4:
                features, labels, patient_ids, recording_ids = batch
                features = features.to(device)
                embeddings = model(features)
            else:
                x_seq, labels, patient_ids, recording_ids, delta_t, mask = batch
                x_seq = x_seq.to(device)
                delta_t = delta_t.to(device)
                mask = mask.to(device) if mask is not None else None
                embeddings = model(x_seq, delta_t=delta_t, key_padding_mask=mask)

            logits = classifier(embeddings)
            probs = torch.softmax(logits, dim=1)

            all_embeddings.append(embeddings.cpu().numpy())
            all_predictions.append(probs[:, 1].cpu().numpy())
            all_labels.append(labels.cpu().numpy())   # Better use "cpu()"
            all_patient_ids.extend(patient_ids)
            all_recording_ids.extend(recording_ids)

    embeddings = np.vstack(all_embeddings)
    predictions = np.concatenate(all_predictions)
    labels = np.concatenate(all_labels)
    patient_ids = np.array(all_patient_ids)
    recording_ids = np.array(all_recording_ids)

    return embeddings, predictions, labels, patient_ids, recording_ids


def train_lopo(
    features: np.ndarray,
    labels: np.ndarray,
    patient_ids: np.ndarray,
    recording_ids: np.ndarray,
    embedding_dim: int = 64,
    hidden_dims: list = [256, 128],
    batch_size: int = 128,
    num_epochs: int = 50,
    learning_rate: float = 0.001,
    warmup_ratio: float = 0.0,
    weight_decay: float = 0.0,
    max_grad_norm: float = 1.0,
    dropout: float = 0.3,
    device: str = "cpu",
):
    """
    Train using Leave-One-Patient-Out (LOPO) cross-validation.

    Parameters
    ----------
    features : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    labels : np.ndarray
        Binary labels.
    patient_ids : np.ndarray
        Patient identifiers.
    recording_ids : np.ndarray
        Recording identifiers.
    embedding_dim : int, default=64
        Dimension of learned embeddings.
    hidden_dims : list, default=[256, 128]
        Hidden layer dimensions.
    batch_size : int, default=128
        Batch size for training.
    num_epochs : int, default=50
        Number of training epochs per fold.
    learning_rate : float, default=0.001
        Learning rate for optimizer.
    warmup_ratio : float, default=0.0
        Linear warmup ratio over total training steps (0 = disable)
    weight_decay : float, default=0.0
        L2 regularization.
    max_grad_norm : float, default=1.0
        Max gradient norm for clipping.
    dropout: float, default=0.3
        Dropout rate for training.
    device : str, default='cpu'
        Device to use ('cpu' or 'cuda').

    Returns
    -------
    all_results : dict
        Dictionary containing results for all folds.
    """
    
    unique_patients = np.unique(patient_ids)
    print(f"\nStarting LOPO cross-validation with {len(unique_patients)} folds...\n")

    all_per_patient_aucs = {}
    all_test_embeddings = []
    all_test_labels = []
    all_test_patient_ids = []
    all_test_recording_ids = []

    for test_patient in unique_patients:
        print(f"\n{'=' * 40}")
        print(f"Fold: Holding out {test_patient}")
        print(f"{'=' * 40}")

        # Split data: train on all patients except "test_patient"
        train_mask = patient_ids != test_patient
        test_mask = patient_ids == test_patient

        X_train, y_train = features[train_mask], labels[train_mask]
        X_test, y_test = features[test_mask], labels[test_mask]
        
        patient_ids_train = patient_ids[train_mask]
        patient_ids_test = patient_ids[test_mask]
        
        recording_ids_train = recording_ids[train_mask]
        recording_ids_test = recording_ids[test_mask]

        counts_train = np.bincount(y_train)     # [neg, pos]
        counts_test = np.bincount(y_test)
        
        print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")
        print(f"Train label dist: {counts_train} | Test label dist: {counts_test}")

        # Standardize features (fit on train, transform both)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_test = scaler.transform(X_test).astype(np.float32)

        # Create datasets and dataloaders
        train_dataset = VoiceDataset(X_train, y_train, patient_ids_train, recording_ids_train)
        test_dataset = VoiceDataset(X_test, y_test, patient_ids_test, recording_ids_test)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Initialize model and classifier (use EmbeddingModel or EmbeddingTransformer)
        model = EmbeddingTransformer(
            input_dim=features.shape[1], embedding_dim=embedding_dim, hidden_dims=hidden_dims, dropout=dropout
        ).to(device)
        classifier = LinearClassifierHead(embedding_dim=embedding_dim, num_classes=2).to(device)
        print(f"Model parameters: {model.get_num_parameters():,}")

        # Criterion with comparable contributions between 0/1 labels
        weights = torch.tensor([1.0, counts_train[0] / counts_train[1]], device=device).float()
        criterion = nn.CrossEntropyLoss(weight=weights)

        # Optimizer (EmbeddingModel: Adam; EmbeddingTransformer: AdamW)
        # optimizer = optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=learning_rate)
        optimizer = optim.AdamW(
            list(model.parameters()) + list(classifier.parameters()), lr=learning_rate, weight_decay=weight_decay
        )
        
        # LR scheduler updates the optimizer
        total_steps = num_epochs * len(train_loader)
        warmup_steps = int(warmup_ratio * total_steps)
        scheduler = None
        if warmup_steps > 0:
            def lr_lambda(step: int):
                return min((step + 1) / warmup_steps, 1.0)
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        print(f"Warmup steps: {warmup_steps}/{total_steps}")

        # Training loop
        best_loss = float("inf")
        for epoch in range(num_epochs):
            train_loss = train_epoch(
                model, classifier, train_loader, criterion, optimizer, max_grad_norm, device, scheduler
            )
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {train_loss:.4f}")
            if train_loss < best_loss:
                best_loss = train_loss

        # Evaluate on test patient
        test_embeddings, test_preds, test_labels_array, test_pids, test_rids = evaluate(
            model, classifier, test_loader, device
        )

        # Store results
        all_test_embeddings.append(test_embeddings)
        all_test_labels.append(test_labels_array)
        all_test_patient_ids.append(test_pids)
        all_test_recording_ids.append(test_rids)

        # Compute per-patient AUC for this fold (at recording level)
        per_patient_auc = compute_per_patient_auc(test_pids, test_labels_array, test_preds, test_rids)
        all_per_patient_aucs.update(per_patient_auc)

        # Print fold results
        for pid, auc in per_patient_auc.items():
            if auc is not None:
                print(f"\n{test_patient} ROC AUC (recording level): {auc:.4f}")

    # Aggregate results across all folds
    print(f"\n\n{'#' * 60}")
    print("FINAL RESULTS - Leave-One-Patient-Out Cross-Validation")
    print(f"{'#' * 60}")

    mean_auc, std_auc, valid_aucs = aggregate_patient_aucs(all_per_patient_aucs)

    # Compute random baseline
    print("\nComputing random baseline (Monte Carlo, N=100)...")
    all_test_recording_ids_array = np.concatenate(all_test_recording_ids)
    random_mean, random_std, _ = compute_random_baseline(
        np.concatenate(all_test_patient_ids),
        np.concatenate(all_test_labels),
        all_test_recording_ids_array,
        n_iterations=100,
    )

    print_evaluation_results(all_per_patient_aucs, mean_auc, std_auc, "LOPO", (random_mean, random_std))

    # Concatenate all test results for visualization
    all_test_embeddings = np.vstack(all_test_embeddings)
    all_test_labels = np.concatenate(all_test_labels)
    all_test_patient_ids = np.concatenate(all_test_patient_ids)
    all_test_recording_ids_array = np.concatenate(all_test_recording_ids)

    # Plot embeddings
    print("Generating embedding visualization...")
    plot_embeddings_2d(
        all_test_embeddings,
        all_test_labels,
        all_test_patient_ids,
        title=f"Learned Embeddings (LOPO) - Mean AUC: {mean_auc:.4f}",
        save_path="embeddings_visualization.png",
    )
    plt.show()

    return {
        "per_patient_auc": all_per_patient_aucs,
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "embeddings": all_test_embeddings,
        "labels": all_test_labels,
        "patient_ids": all_test_patient_ids,
    }


def train_contextual_lopo(
    features: np.ndarray,
    labels: np.ndarray,
    patient_ids: np.ndarray,
    recording_ids: np.ndarray,
    timestamps: np.ndarray,                 # New: 每条样本的时间戳（用于构造上下文与 delta_t）
    context_len: int = 64,
    embedding_dim: int = 64,
    hidden_dims: list = [256, 128],
    batch_size: int = 128,
    num_epochs: int = 50,
    learning_rate: float = 0.001,
    warmup_ratio: float = 0.0,
    weight_decay: float = 0.0,
    max_grad_norm: float = 1.0,
    dropout: float = 0.3,
    device: str = "cpu",
):
    """
    Contextual LOPO: Leave-One-Patient-Out cross-validation, but each training sample is a
    context window [T, input_dim] built within the same patient timeline.
    """

    unique_patients = np.unique(patient_ids)
    print(f"\nStarting CONTEXTUAL LOPO with {len(unique_patients)} folds...\n")

    all_per_patient_aucs = {}
    all_test_embeddings = []
    all_test_labels = []
    all_test_patient_ids = []
    all_test_recording_ids = []

    input_dim = features.shape[1]

    for test_patient in unique_patients:
        print(f"\n{'=' * 40}")
        print(f"Fold: Holding out {test_patient}")
        print(f"{'=' * 40}")

        # Row-level split
        train_mask = patient_ids != test_patient
        test_mask = patient_ids == test_patient

        X_train, y_train = features[train_mask], labels[train_mask]
        X_test,  y_test  = features[test_mask],  labels[test_mask]

        pid_train = patient_ids[train_mask]
        pid_test  = patient_ids[test_mask]

        rid_train = recording_ids[train_mask]
        rid_test  = recording_ids[test_mask]

        t_train = timestamps[train_mask]
        t_test  = timestamps[test_mask]

        counts_train_rows = np.bincount(y_train, minlength=2)
        counts_test_rows  = np.bincount(y_test, minlength=2)

        print(f"Row-level Train samples: {len(X_train)} | Test samples: {len(X_test)}")
        print(f"Row-level Train label dist: {counts_train_rows} | Test label dist: {counts_test_rows}")

        # Standardize (fit on train rows, transform both)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_test  = scaler.transform(X_test).astype(np.float32)

        # Build contextual datasets (sequence samples)
        # ContextSequenceDataset 在 __getitem__ 返回：x_seq, y, patient_id, recording_id, delta_t, key_padding_mask
        train_dataset = ContextSequenceDataset(
            features=X_train,
            labels=y_train,
            patient_ids=pid_train,
            recording_ids=rid_train,
            timestamps=t_train,
            context_len=context_len,
        )
        test_dataset = ContextSequenceDataset(
            features=X_test,
            labels=y_test,
            patient_ids=pid_test,
            recording_ids=rid_test,
            timestamps=t_test,
            context_len=context_len,
        )

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True
        )

        # Sequence-level class weights (IMPORTANT)
        seq_counts = np.bincount(np.asarray(train_dataset.labels), minlength=2)  # [neg, pos]
        neg, pos = int(seq_counts[0]), int(seq_counts[1])
        print(f"Seq-level Train samples: {len(train_dataset)} | label dist: {seq_counts}")

        if pos == 0:
            raise RuntimeError(f"Fold {test_patient}: no positive sequence samples in training set.")

        weights = torch.tensor([1.0, neg / pos], device=device, dtype=torch.float32)
        weights = weights / weights.mean()  # 可选：稳定尺度
        criterion = nn.CrossEntropyLoss(weight=weights)

        # Model & classifier
        model = ContextualEmbeddingTransformer(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        ).to(device)
        classifier = LinearClassifierHead(embedding_dim=embedding_dim, num_classes=2).to(device)

        if hasattr(model, "get_num_parameters"):
            print(f"Model parameters: {model.get_num_parameters():,}")
        else:
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Model parameters: {n_params:,}")

        # Optimizer + warmup scheduler (step-wise)
        optimizer = optim.AdamW(
            list(model.parameters()) + list(classifier.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        total_steps = num_epochs * len(train_loader)
        warmup_steps = int(warmup_ratio * total_steps)
        scheduler = None
        if warmup_steps > 0:
            def lr_lambda(step: int):
                return min((step + 1) / warmup_steps, 1.0)
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        print(f"Warmup steps: {warmup_steps}/{total_steps}")

        # Training loop
        best_loss = float("inf")
        for epoch in range(num_epochs):
            train_loss = train_compatible(
                model=model,
                classifier=classifier,
                train_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                max_grad_norm=max_grad_norm,
                device=device,
                scheduler=scheduler,
            )
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {train_loss:.4f}")
            best_loss = min(best_loss, train_loss)

        # Evaluate
        test_embeddings, test_preds, test_labels_array, test_pids, test_rids = evaluate_compatible(
            model=model,
            classifier=classifier,
            data_loader=test_loader,
            device=device,
        )

        # Store for overall viz/baseline
        all_test_embeddings.append(test_embeddings)
        all_test_labels.append(test_labels_array)
        all_test_patient_ids.append(test_pids)
        all_test_recording_ids.append(test_rids)

        # Recording-level AUC for this fold
        per_patient_auc = compute_per_patient_auc(test_pids, test_labels_array, test_preds, test_rids)
        all_per_patient_aucs.update(per_patient_auc)

        for pid, auc in per_patient_auc.items():
            if auc is not None:
                print(f"\n{test_patient} ROC AUC (recording level): {auc:.4f}")

    # Aggregate
    print(f"\n\n{'#' * 60}")
    print("FINAL RESULTS - CONTEXTUAL LOPO")
    print(f"{'#' * 60}")

    mean_auc, std_auc, valid_aucs = aggregate_patient_aucs(all_per_patient_aucs)

    # Random baseline (same as before)
    print("\nComputing random baseline (Monte Carlo, N=100)...")
    all_test_recording_ids_array = np.concatenate(all_test_recording_ids)
    random_mean, random_std, _ = compute_random_baseline(
        np.concatenate(all_test_patient_ids),
        np.concatenate(all_test_labels),
        all_test_recording_ids_array,
        n_iterations=100,
    )

    print_evaluation_results(
        all_per_patient_aucs,
        mean_auc,
        std_auc,
        "CONTEXTUAL_LOPO",
        (random_mean, random_std),
    )

    # Visualization
    all_test_embeddings = np.vstack(all_test_embeddings)
    all_test_labels = np.concatenate(all_test_labels)
    all_test_patient_ids = np.concatenate(all_test_patient_ids)

    print("Generating embedding visualization...")
    plot_embeddings_2d(
        all_test_embeddings,
        all_test_labels,
        all_test_patient_ids,
        title=f"Contextual Embeddings (LOPO) - Mean AUC: {mean_auc:.4f}",
        save_path="embeddings_visualization.png",
    )
    plt.show()

    return {
        "per_patient_auc": all_per_patient_aucs,
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "embeddings": all_test_embeddings,
        "labels": all_test_labels,
        "patient_ids": all_test_patient_ids,
    }


def main():
    """Main training function."""
    # Load configuration from config.py
    config = get_config()
    config.print_config()

    DEVICE = "cuda" if (torch.cuda.is_available() and config.use_cuda) else "cpu"
    print(f"Using device: {DEVICE}")

    # Load data
    features, labels, patient_ids, recording_ids, feature_names = load_data(config.data_path)

    # Train with LOPO
    results = train_lopo(
        features=features,
        labels=labels,
        patient_ids=patient_ids,
        recording_ids=recording_ids,
        embedding_dim=config.embedding_dim,
        hidden_dims=config.hidden_dims,
        batch_size=config.batch_size,
        num_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        dropout=config.dropout,
        device=DEVICE,
    )

    print("\nTraining complete!")
    print(f"Final Mean ROC AUC: {results['mean_auc']:.4f} ± {results['std_auc']:.4f}")
    print("Embedding visualization saved to: embeddings_visualization.png")


def main_contextual():
    config = get_config()
    config.print_config()

    DEVICE = "cuda" if (torch.cuda.is_available() and config.use_cuda) else "cpu"
    print(f"Using device: {DEVICE}")

    # Load data (新版：多返回 start_time 或 timestamps)
    features, labels, patient_ids, recording_ids, timestamps, _ = load_contextual_data(config.data_path)

    # Train with LOPO (新版：传入 start_time + context_len)
    results = train_contextual_lopo(
        features=features,
        labels=labels,
        patient_ids=patient_ids,
        recording_ids=recording_ids,
        timestamps=timestamps,              # New
        context_len=config.context_len,     # Mew (Added in the config.py)
        embedding_dim=config.embedding_dim,
        hidden_dims=config.hidden_dims,
        batch_size=config.batch_size,
        num_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        dropout=config.dropout,
        device=DEVICE,
    )

    print("\nTraining complete!")
    print(f"Final Mean ROC AUC: {results['mean_auc']:.4f} ± {results['std_auc']:.4f}")
    print("Embedding visualization saved to: embeddings_visualization.png")


if __name__ == "__main__":
    USE_CONTEXTUAL = True
    if USE_CONTEXTUAL:
        main_contextual()
    else:
        main()
