import torch
import numpy as np
from .autoencoder import TabularAutoencoder

def detect_anomalies(model: TabularAutoencoder, X: np.ndarray, threshold: float):
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X).float()
        recon = model(X_t)
        errors = torch.mean((recon - X_t)**2, dim=1).cpu().numpy()
    mask = errors > threshold
    return mask, errors
