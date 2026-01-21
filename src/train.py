"""
Training script with Leave-One-Patient-Out (LOPO) cross-validation.

This script enforces proper LOPO evaluation to prevent data leakage and
uses a forced linear classifier head to ensure rich embeddings are learned.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import datetime
import os
from models import EmbeddingModel, LinearClassifierHead, Transformer_Decoder
from utils import (
    compute_per_patient_auc,
    aggregate_patient_aucs,
    plot_embeddings_2d,
    print_evaluation_results,
    compute_random_baseline,
    load_data,
    load_aggregate_data
)
from config import get_config, get_config_Transformer


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
    """PyTorch Dataset for voice recordings."""

    def __init__(self, features, labels, patient_ids, recording_ids=None, context_size=4):
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
        
        self.features = torch.empty((1, context_size, features.shape[1]))
        context = torch.zeros((context_size, features.shape[1]))
        torch_features = torch.FloatTensor(features)
        for feature in torch_features :
            context = torch.concatenate((context[1:,:],feature.unsqueeze(0)), dim=0)
            self.features = torch.concatenate((self.features, context.unsqueeze(0)), dim=0)

        # self.features = torch.FloatTensor(features)
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

def train_epoch(model, classifier, train_loader, criterion, optimizer, device):
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
        optimizer.step()

        total_loss += loss.item() * features.size(0)

    return total_loss / len(train_loader.dataset)

def train_epoch_Transformer(model, classifier, train_loader, criterion, optimizer, device):
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

def train_lopo_Transformer(
    features: np.ndarray,
    labels: np.ndarray,
    patient_ids: np.ndarray,
    recording_ids: np.ndarray,
    n_features: int,
    n_layer: int = 3,
    embedding_dim: int = 64,
    num_heads: int = 10,
    block_size: int = 4,
    dropout: float = 0.3,
    batch_size: int = 128,
    num_epochs: int = 50,
    learning_rate: float = 0.001,
    device: str = "cpu",
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
        train_dataset = Transformer_VoiceDataset(
            X_train, y_train, patient_ids_train, recording_ids_train, block_size
        )
        test_dataset = Transformer_VoiceDataset(
            X_test, y_test, patient_ids_test, recording_ids_test, block_size
        )

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
            device=device
            ).to(device)

        classifier = LinearClassifierHead(
            embedding_dim=embedding_dim, num_classes=2
        ).to(device)

        print(f"Model parameters: {model.get_num_parameters():,}")

        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            list(model.parameters()) + list(classifier.parameters()), lr=learning_rate
        )

        # Training loop
        best_loss = float("inf")
        for epoch in range(num_epochs):
            train_loss = train_epoch_Transformer(
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
    aggregate_recordings=False
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
    )

    print("\nTraining complete!")
    print(f"Final Mean ROC AUC: {results['mean_auc']:.4f} ± {results['std_auc']:.4f}")
    print("Embedding visualization saved to: embeddings_visualization.png")

    config.save(additionnal_text=f"Results per patient : \t{results["per_patient_auc"]}\nFinal Mean ROC AUC: \t{results['mean_auc']:.4f} ± {results['std_auc']:.4f}")

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
        n_layer=config.n_layer,
        embedding_dim=config.embedding_dim,
        num_heads=config.num_heads,
        block_size=config.block_size,
        batch_size=config.batch_size,
        num_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        device=DEVICE,
    )

    print("\nTraining complete!")
    print(f"Final Mean ROC AUC: {results['mean_auc']:.4f} ± {results['std_auc']:.4f}")
    print("Embedding visualization saved to: embeddings_visualization.png")

    config.save(additionnal_text=f"Results per patient : \t{results["per_patient_auc"]}\nFinal Mean ROC AUC: \t{results['mean_auc']:.4f} ± {results['std_auc']:.4f}")

if __name__ == "__main__":
    # df = pd.read_parquet("data/dataset_with_aug_dict.parquet")
    # print(df)
    main_Transformer()
    # Add embedding model with contrastive supervised loss
    # Change the classification loss and the contrastive loss to take the imbalance into account
    # Pour embedder un recording, un attention pooling peut être plus intelligent qu'une simple moyenne
    # Augmentation de données
    # Centrer les features du patient par rapport à son ensemble de recordings
    # Pondérer le positionnal encoding en fonction 
    # dans l'attention prendre en compte la disatnce de temps entre les tokens
    # intégreer sigma dans le dataset reduit
    # pendre en compte l'augmentation et éventuellemnt augmenté avec du dropout

    ### Taches 
    #
