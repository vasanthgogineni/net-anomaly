from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import joblib
import torch
from sklearn.preprocessing import RobustScaler
from .autoencoder import AE, reconstruction_error
from .lstm import LSTMAE, sequence_recon_error

class EnsembleModels:
    def __init__(self, artifacts: dict[str, str], device: str = 'cpu'):
        self.paths = artifacts
        self.device = torch.device(device)
        # load scaler and columns
        self.scaler: RobustScaler = joblib.load(self.paths['scaler_path'])
        with open(self.paths['columns_path'], 'r') as f:
            self.columns: list[str] = json.load(f)
        # sklearn models
        self.iforest = joblib.load(self.paths['iforest_path'])
        self.ocsvm = joblib.load(self.paths['ocsvm_path'])
        # torch models
        self.ae = AE(in_dim=len(self.columns))
        self.ae.load_state_dict(torch.load(self.paths['ae_path'], map_location=self.device))
        self.ae.to(self.device).eval()
        self.lstm = LSTMAE(in_dim=len(self.columns))
        self.lstm.load_state_dict(torch.load(self.paths['lstm_path'], map_location=self.device))
        self.lstm.to(self.device).eval()

        try:
            with open(self.paths['thresholds_path'], 'r') as f:
                th = json.load(f)
        except (FileNotFoundError, KeyError, TypeError):
            th = {'iforest': 1.0, 'ocsvm': 1.0, 'ae': 1.0, 'lstm': 1.0}
        self.thresholds = th

        self.thresholds = th

    def transform(self, X_df):
        X = X_df[self.columns].to_numpy(dtype=float)
        Xs = self.scaler.transform(X)
        return Xs

    def score_point(self, x_scaled: np.ndarray, seq_buffer: np.ndarray | None, weights: dict[str,float]):
        i_s = -float(self.iforest.decision_function([x_scaled])[0])
        o_s = -float(self.ocsvm.decision_function([x_scaled])[0])
        x_t = torch.tensor(x_scaled[None, :], dtype=torch.float32, device=self.device)
        ae_s = float(reconstruction_error(self.ae, x_t).cpu().numpy()[0])
        if seq_buffer is None:
            lstm_s = ae_s
        else:
            seq = torch.tensor(seq_buffer[None, :, :], dtype=torch.float32, device=self.device)
            lstm_s = float(sequence_recon_error(self.lstm, seq).cpu().numpy()[0])
        th = self.thresholds
        def norm(s, name):
            t = th.get(name, 1e-6)
            return s / (t if t>0 else 1.0)
        scores = {
            'iforest': norm(i_s,'iforest'),
            'ocsvm': norm(o_s,'ocsvm'),
            'ae': norm(ae_s,'ae'),
            'lstm': norm(lstm_s,'lstm'),
        }
        ensemble = sum(scores[k]*weights.get(k,0.25) for k in scores)
        return scores, float(ensemble)
