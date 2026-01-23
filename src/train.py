"""
Training script with Leave-One-Patient-Out (LOPO) cross-validation.

This script enforces proper LOPO evaluation to prevent data leakage and
uses a forced linear classifier head to ensure rich embeddings are learned.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import datetime
from models import EmbeddingModel, LinearClassifierHead, Transformer_Decoder
from utils import (
    FocalLossBinary,
    SupConLoss,
    CenterLoss,
    compute_per_patient_auc,
    aggregate_patient_aucs,
    plot_embeddings_2d,
    print_evaluation_results,
    compute_random_baseline,
    load_data,
    load_aggregate_data
)
from config import get_config, get_config_Transformer

from sklearn.model_selection import GridSearchCV 
import itertools
import copy


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
    
class Transformer_VoiceDataset(Dataset):
    """PyTorch Dataset for voice recordings with temporal contexts."""

    def __init__(self, features, labels, patient_ids, recording_ids=None, context_size=4, times=None):
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
        times : np.ndarray, optional
            Timestamps aligned with features for temporal attention weighting.
        """
        self.features = torch.empty((1, context_size, features.shape[1]))
        self.times = torch.empty((1, context_size))

        feature_context = torch.zeros((context_size, features.shape[1]))
        time_context = torch.zeros((context_size,))

        torch_features = torch.FloatTensor(features)
        torch_times = torch.FloatTensor(times) if times is not None else torch.arange(len(features)).float()

        for feat, t in zip(torch_features, torch_times):
            feature_context = torch.concatenate((feature_context[1:, :], feat.unsqueeze(0)), dim=0)
            time_context = torch.concatenate((time_context[1:], t.unsqueeze(0)), dim=0)

            self.features = torch.concatenate((self.features, feature_context.unsqueeze(0)), dim=0)
            self.times = torch.concatenate((self.times, time_context.unsqueeze(0)), dim=0)

        # Drop the initial padding row
        self.features = self.features[1:]
        self.times = self.times[1:]

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
            self.times[idx],
        )
    
def train_epoch_encoder(model, classifier, train_loader, criterion_classification, criterion_supcon, criterion_center, encoding, optimizer, train_classifier, device):
    """Train for one epoch."""
    model.train()
    a_contrastive, a_centering = encoding[0], encoding[1]
    total_loss = 0.0
    for features, labels, _, _ in train_loader:  # Added recording_id to unpack
        features = features.to(device)
        labels = labels.to(device)

        # Forward pass
        embeddings = model(features)
        loss_center = criterion_center(embeddings, labels)
        loss_contrastive = criterion_supcon(embeddings, labels)
        loss = a_contrastive*loss_contrastive + a_centering*loss_center
        if train_classifier :
            logits = classifier(embeddings)
            loss += criterion_classification(logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * features.size(0)

    return total_loss / len(train_loader.dataset)

def train_epoch(model, classifier, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    classifier.train()

    total_loss = 0.0
    for features, labels, _, _, time_ctx in train_loader:
        features = features.to(device)
        labels = labels.to(device)
        time_ctx = time_ctx.to(device)

        # Forward pass
        embeddings = model(features, time_ctx)
        logits = classifier(embeddings)
        loss = criterion(logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * features.size(0)

    return total_loss / len(train_loader.dataset)


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
        for batch in data_loader:
            if len(batch) == 4:
                features, labels, patient_ids, recording_ids = batch
                time_ctx = None
            else:
                features, labels, patient_ids, recording_ids, time_ctx = batch
            features = features.to(device)
            time_ctx = time_ctx.to(device) if time_ctx is not None else None

            # Get embeddings and predictions
            if time_ctx is not None:
                embeddings = model(features, time_ctx)
            else:
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


def train_lopo_Transformer(
    features: np.ndarray,
    labels: np.ndarray,
    patient_ids: np.ndarray,
    recording_ids: np.ndarray,
    n_features: int,
    feature_names: list,
    n_layer: int = 3,
    embedding_dim: int = 64,
    num_heads: int = 10,
    block_size: int = 4,
    dropout: float = 0.3,
    batch_size: int = 128,
    num_epochs: int = 50,
    learning_rate: float = 0.001,
    device: str = "cpu",
    plot=False,
    non_linear_classifier=False,
    train_encoder=True,
    encoding=None,
    train_classifier=False
):
    """
    Train using Leave-One-Patient-Out cross-validation.

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
        print(f"\n{'=' * 60}")
        print(f"Fold: Holding out {test_patient}")
        print(f"{'=' * 60}")

        # Split data: train on all patients except test_patient
        train_mask = patient_ids != test_patient
        test_mask = patient_ids == test_patient

        X_train, y_train = features[train_mask], labels[train_mask]
        X_test, y_test = features[test_mask], labels[test_mask]
        patient_ids_train = patient_ids[train_mask]
        patient_ids_test = patient_ids[test_mask]
        recording_ids_train = recording_ids[train_mask]
        recording_ids_test = recording_ids[test_mask]

        # Use start_time (it could also be end_time) to order sequences and compute temporal gaps
        time_train = features[train_mask][:, feature_names.index("start_time")]
        time_test = features[test_mask][:, feature_names.index("start_time")]

        # Ensure numeric times (datetime64 -> int64) then sort so contexts reflect temporal progression
        time_train = time_train.astype(np.float64)
        time_test = time_test.astype(np.float64)

        train_order = np.argsort(time_train)
        test_order = np.argsort(time_test)

        X_train, y_train = X_train[train_order], y_train[train_order]
        patient_ids_train, recording_ids_train = patient_ids_train[train_order], recording_ids_train[train_order]
        time_train = time_train[train_order]

        X_test, y_test = X_test[test_order], y_test[test_order]
        patient_ids_test, recording_ids_test = patient_ids_test[test_order], recording_ids_test[test_order]
        time_test = time_test[test_order]

        print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")
        print(f"Train label dist: {np.bincount(y_train)}")
        print(f"Test label dist: {np.bincount(y_test)}")

        # Standardize features (fit on train, transform both)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Create datasets and dataloaders
        train_dataset = Transformer_VoiceDataset(
            X_train, y_train, patient_ids_train, recording_ids_train, block_size, time_train
        )
        test_dataset = Transformer_VoiceDataset(
            X_test, y_test, patient_ids_test, recording_ids_test, block_size, time_test
        )

        # # Handle class imbalance (DO NOT IMPROVE THE RESULTS)
        # sample_weights = np.where(
        #     y_train == 1, 
        #     len(y_train) / (2 * np.sum(y_train == 1)),
        #     len(y_train) / (2 * np.sum(y_train == 0))
        # )
        # sampler = WeightedRandomSampler(sample_weights, len(y_train), replacement=True)
        # train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Initialize model and classifier
        input_dim = features.shape[1]
        model = Transformer_Decoder(
            n_features=n_features,
            n_layer=n_layer,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            block_size=block_size,
            dropout=dropout,
            device=device,
            time_decay=time_decay,
            ).to(device)
        
        if non_linear_classifier :
            classifier = nn.Sequential(
                            EmbeddingModel(input_dim=embedding_dim,embedding_dim=embedding_dim), 
                            LinearClassifierHead(
                                embedding_dim=embedding_dim, num_classes=2
                                ).to(device)
                            )
        else :
            classifier = LinearClassifierHead(
                embedding_dim=embedding_dim, num_classes=2
            ).to(device)
        

        print(f"Model parameters: {model.get_num_parameters():,}")

        # Setup training
        if encoding is None :
            criterion = FocalLossBinary() 
            if train_encoder :
                optimizer = optim.Adam(
                    list(model.parameters()) + list(classifier.parameters()), lr=learning_rate, weight_decay=1e-1
                )
            else :
                optimizer = optim.Adam(
                    list(classifier.parameters()), lr=learning_rate, weight_decay=1e-1
                )

            # Training loop
            best_loss = float("inf")
            for epoch in range(num_epochs):
                train_loss = train_epoch(
                    model, classifier, train_loader, criterion, optimizer, device
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
            per_patient_auc = compute_per_patient_auc(
                test_pids, test_labels_array, test_preds, test_rids
            )
            all_per_patient_aucs.update(per_patient_auc)

            # Print fold results
            for pid, auc in per_patient_auc.items():
                if auc is not None:
                    print(f"\n{test_patient} ROC AUC (recording level): {auc:.4f}")
        
        else :
            criterion_classification = FocalLossBinary()
            criterion_supcon = SupConLoss(temperature=0.3) 
            criterion_center = CenterLoss(2,embedding_dim,device=device)

            if train_classifier :
                optimizer = optim.Adam(
                    list(model.parameters()) + list(classifier.parameters()), lr=learning_rate, weight_decay=1e-1
                )
            else :
                optimizer = optim.Adam(
                    list(model.parameters()), lr=learning_rate, weight_decay=1e-1
                )
            # Training loop
            best_loss = float("inf")
            for epoch in range(num_epochs):
                train_loss = train_epoch_encoder(
                    model, classifier, train_loader, criterion_classification, criterion_supcon, criterion_center, encoding, optimizer, train_classifier, device
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
            per_patient_auc = compute_per_patient_auc(
                test_pids, test_labels_array, test_preds, test_rids
            )
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

    print_evaluation_results(
        all_per_patient_aucs, mean_auc, std_auc, "LOPO", (random_mean, random_std)
    )

    # Concatenate all test results for visualization
    all_test_embeddings = np.vstack(all_test_embeddings)
    all_test_labels = np.concatenate(all_test_labels)
    all_test_patient_ids = np.concatenate(all_test_patient_ids)
    all_test_recording_ids_array = np.concatenate(all_test_recording_ids)

    # Plot embeddings
    print("Generating embedding visualization...")
    time = datetime.datetime.now().strftime('%Y_%m_%d-%H_%M')
    
    if plot :
        plot_embeddings_2d(
            all_test_embeddings,
            all_test_labels,
            all_test_patient_ids,
            title=f"Learned Embeddings (LOPO) - Mean AUC: {mean_auc:.4f}",
            save_path=f"experiments/{time}_embeddings_visualization.png",
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



def train_lopo_MLP(
    features: np.ndarray,
    labels: np.ndarray,
    patient_ids: np.ndarray,
    recording_ids: np.ndarray,
    embedding_dim: int = 64,
    hidden_dims: list = [256, 128],
    batch_size: int = 128,
    num_epochs: int = 50,
    learning_rate: float = 0.001,
    device: str = "cpu",
    plot=False
):
    """
    Train using Leave-One-Patient-Out cross-validation.

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
        print(f"\n{'=' * 60}")
        print(f"Fold: Holding out {test_patient}")
        print(f"{'=' * 60}")

        # Split data: train on all patients except test_patient
        train_mask = patient_ids != test_patient
        test_mask = patient_ids == test_patient

        X_train, y_train = features[train_mask], labels[train_mask]
        X_test, y_test = features[test_mask], labels[test_mask]
        patient_ids_train = patient_ids[train_mask]
        patient_ids_test = patient_ids[test_mask]
        recording_ids_train = recording_ids[train_mask]
        recording_ids_test = recording_ids[test_mask]

        print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")
        print(f"Train label dist: {np.bincount(y_train)}")
        print(f"Test label dist: {np.bincount(y_test)}")

        # Standardize features (fit on train, transform both)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Create datasets and dataloaders
        train_dataset = VoiceDataset(
            X_train, y_train, patient_ids_train, recording_ids_train
        )
        test_dataset = VoiceDataset(
            X_test, y_test, patient_ids_test, recording_ids_test
        )

        # # Handle class imbalance (DO NOT IMPROVE THE RESULTS)
        # sample_weights = np.where(
        #     y_train == 1, 
        #     len(y_train) / (2 * np.sum(y_train == 1)),
        #     len(y_train) / (2 * np.sum(y_train == 0))
        # )
        # sampler = WeightedRandomSampler(sample_weights, len(y_train), replacement=True)
        # train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Initialize model and classifier
        input_dim = features.shape[1]
        model = EmbeddingModel(
            input_dim=input_dim, embedding_dim=embedding_dim, hidden_dims=hidden_dims
        ).to(device)

        classifier = LinearClassifierHead(
                    embedding_dim=embedding_dim, num_classes=2
                    ).to(device)

        print(f"Model parameters: {model.get_num_parameters():,}")

        # Setup training
        # Handle class imbalance (DO NOT IMPROVE THE RESULTS)
        # # Handle class imbalance
        # class_weights = torch.tensor(
        #     [
        #         # # 1 for class 0 and negative/positive ratio for class 1
        #         # 1.0,  # Class 0
        #         # (len(y_train) - np.sum(y_train)) / np.sum(y_train)  # Class 1

        #         # # Inverse frequency (normalized)
        #         # 1.0 / (np.sum(y_train == 0) / len(y_train)),  # Class 0
        #         # 1.0 / (np.sum(y_train == 1) / len(y_train))  # Class 1

        #         # Square root of inverse frequency (gentler)
        #         np.sqrt(len(y_train) / (np.sum(y_train == 0) + 1)),  # Class 0
        #         np.sqrt(len(y_train) / (np.sum(y_train == 1) + 1))  # Class 1
        #     ], device=device, dtype=torch.float32
        # )
        # criterion = nn.CrossEntropyLoss(weight=class_weights)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            list(model.parameters()) + list(classifier.parameters()), lr=learning_rate
        )

        # Training loop
        best_loss = float("inf")
        for epoch in range(num_epochs):
            train_loss = train_epoch(
                model, classifier, train_loader, criterion, optimizer, device
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
        per_patient_auc = compute_per_patient_auc(
            test_pids, test_labels_array, test_preds, test_rids
        )
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

    print_evaluation_results(
        all_per_patient_aucs, mean_auc, std_auc, "LOPO", (random_mean, random_std)
    )

    # Concatenate all test results for visualization
    all_test_embeddings = np.vstack(all_test_embeddings)
    all_test_labels = np.concatenate(all_test_labels)
    all_test_patient_ids = np.concatenate(all_test_patient_ids)
    all_test_recording_ids_array = np.concatenate(all_test_recording_ids)

    # Plot embeddings
    print("Generating embedding visualization...")
    time = datetime.datetime.now().strftime('%Y_%m_%d-%H_%M')
    if plot:
        plot_embeddings_2d(
            all_test_embeddings,
            all_test_labels,
            all_test_patient_ids,
            title=f"Learned Embeddings (LOPO) - Mean AUC: {mean_auc:.4f}",
            save_path=f"experiments/{time}_embeddings_visualization.png",
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




def main_MLP():
    """Main training function."""
    # Load configuration from config.py
    config = get_config()
    config.print_config()

    DEVICE = "cuda" if (torch.cuda.is_available() and config.use_cuda) else "cpu"
    print(f"Using device: {DEVICE}")

    # Load data
    features, labels, patient_ids, recording_ids, feature_names = load_data(
        config.data_path
    )

    # Train with LOPO
    results = train_lopo_MLP(
        features=features,
        labels=labels,
        patient_ids=patient_ids,
        recording_ids=recording_ids,
        embedding_dim=config.embedding_dim,
        hidden_dims=config.hidden_dims,
        batch_size=config.batch_size,
        num_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        device=DEVICE,
        plot=True
    )

    print("\nTraining complete!")
    print(f"Final Mean ROC AUC: {results['mean_auc']:.4f} ± {results['std_auc']:.4f}")
    print("Embedding visualization saved to: embeddings_visualization.png")

    config.save(additionnal_text=f"Results per patient : \t{results['per_patient_auc']}\nFinal Mean ROC AUC: \t{results['mean_auc']:.4f} ± {results['std_auc']:.4f}")

def main_Transformer():
    """Main training function."""
    # Load configuration from config.py
    config = get_config_Transformer()
    config.print_config()

    DEVICE = "cuda" if (torch.cuda.is_available() and config.use_cuda) else "cpu"
    print(f"Using device: {DEVICE}")

    # Load data
    features, labels, patient_ids, recording_ids, feature_names = load_aggregate_data(
        config.data_path
    )

    # Train with LOPO
    results = train_lopo_Transformer(
        features=features,
        labels=labels,
        patient_ids=patient_ids,
        recording_ids=recording_ids,
        n_features=config.n_features,
        feature_names=feature_names,
        n_layer=config.n_layer,
        embedding_dim=config.embedding_dim,
        num_heads=config.num_heads,
        block_size=config.block_size,
        batch_size=config.batch_size,
        num_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        device=DEVICE,
        plot=True,
        non_linear_classifier=False,
        encoding=[0,0],
        train_classifier=True,
    )

    print("\nTraining complete!")
    print(f"Final Mean ROC AUC: {results['mean_auc']:.4f} ± {results['std_auc']:.4f}")
    print("Embedding visualization saved to: embeddings_visualization.png")

    config.save(additionnal_text=f"Results per patient : \t{results["per_patient_auc"]}\nFinal Mean ROC AUC: \t{results['mean_auc']:.4f} ± {results['std_auc']:.4f}")

def GridSearch_Transformer():
    """Main training function."""
    # Load configuration from config.py
    config = get_config_Transformer()
    config.print_config()

    DEVICE = "cuda" if (torch.cuda.is_available() and config.use_cuda) else "cpu"
    print(f"Using device: {DEVICE}")

    # Load data
    features, labels, patient_ids, recording_ids, feature_names = load_aggregate_data(
        config.data_path
    )


    params_grid = {
        "n_features" : [600],
        "n_layer": [3],
        "embedding_dim": [32],
        "num_heads": [15],
        "block_size": [15],
        "batch_size": [512],
        "num_epochs": [2],
        "learning_rate": [1e-2],
        "dropout": [0.3],
    }

    all_params = list(itertools.product(*params_grid.values()))
    param_names = list(params_grid.keys())

    results_grid = []

    for param_values in all_params:
        params = dict(zip(param_names, param_values))

        print(f"\nTraining with params: {params}")

        results = train_lopo_Transformer(
            features=features,
            labels=labels,
            patient_ids=patient_ids,
            recording_ids=recording_ids,
            n_features=params["n_features"],
            n_layer=params["n_layer"],
            embedding_dim=params["embedding_dim"],
            num_heads=params["num_heads"],
            block_size=params["block_size"],
            batch_size=params["batch_size"],
            num_epochs=params["num_epochs"],
            learning_rate=params["learning_rate"],
            dropout=params["dropout"],
            device=DEVICE,
        )

        results_grid.append({
            "params": params,
            "results": results
        })

    print("\nTraining complete!")
    print(f"Final Mean ROC AUC: {results['mean_auc']:.4f} ± {results['std_auc']:.4f}")
    print("Embedding visualization saved to: embeddings_visualization.png")
    
    # Extraire les AUC
    mean_aucs = [entry["results"]["mean_auc"] for entry in results_grid]

    # Trouver le meilleur
    argmax_auc = np.argmax(mean_aucs)
    max_auc = mean_aucs[argmax_auc]

    best_params = results_grid[argmax_auc]["params"]
    best_results = results_grid[argmax_auc]["results"]

    print("\n Training complete!")
    print(f"Best Mean ROC AUC: {max_auc:.4f} ± {best_results['std_auc']:.4f}")
    print("Best params:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    
    return best_results, best_params


if __name__ == "__main__":
    main_Transformer()

    # Add embedding model with contrastive supervised loss
    # Pour embedder un recording, un attention pooling peut être plus intelligent qu'une simple moyenne
    # Améliorer l'embedding de la donnée aggrégée en ajoutant plus qu'une simple couche linéaire.

    ###### Taches  ######

    # Hugo : Améliorer l'augmentation des données en séparant les données d'un même jour lors de l'aggrégation, ajouter la variance 
    # éventuellement ajouter des données avec du dropout. Ajouter l'age et le sexe dans le plot final
    # Centrer les features du patient par rapport à son ensemble de recordings
    # Haochen : Pondérer l'attention par l'écart de temps entre le dernier segment et les segments du contexte. 
    # (Travailler sur le positionnal encoding)
    # PL : Travailler sur la contrastive loss pour améliorer le transformer embedder. 
    # Regarder pour prendre en compte l'imbalance des classes.
    # Regarder s'il est intéressant d'avoir un classifier plus complexe qu'un classifier linéaire après l'embeddeur
    