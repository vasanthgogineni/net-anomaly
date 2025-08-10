from __future__ import annotations
import argparse
import json
from pathlib import Path
import os
import time
import yaml
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn as nn
import torch.optim as optim

from .sniff import PacketSniffer
from .features import OnlineFeatureExtractor
from .models.autoencoder import AE
from .models.lstm import LSTMAE

def parse_args():
    ap = argparse.ArgumentParser(description="Baseline training for network anomaly detector")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--iface", help="Interface to capture on (overrides config)", default=None)
    ap.add_argument("--capture-seconds", type=int, default=None, help="Baseline capture duration (overrides config)")
    return ap.parse_args()

def capture_baseline(cfg) -> pd.DataFrame:
    iface = cfg.get('iface')
    bpf = cfg.get('bpf_filter') or None
    windows = cfg['windows']
    res = int(cfg.get('feature_resolution_sec',1))
    min_pp = int(cfg.get('min_packets_per_tick',1))
    duration = int(cfg['training']['capture_seconds'])

    fe = OnlineFeatureExtractor(windows=windows, resolution_sec=res, max_retention_sec=cfg.get('max_retention_sec',900))
    sniffer = PacketSniffer(iface=iface, bpf_filter=bpf)
    sniffer.start()

    rows = []
    start = time.time()
    next_tick = int(start) + 1
    try:
        while time.time() - start < duration:
            while True:
                try:
                    pkt = sniffer.get_queue().get_nowait()
                except Exception:
                    break
                fe.on_packet(pkt)
            now = time.time()
            if now >= next_tick:
                row = fe.build_feature_row()
                last_1s = row.get('1s_pps') or row.get('5s_pps') or 0
                if last_1s >= min_pp:
                    rows.append(row)
                next_tick = int(now) + 1
            time.sleep(0.01)
    finally:
        sniffer.stop()

    df = pd.DataFrame(rows).sort_values('ts')
    df = df.dropna(axis=1, how='any')
    return df

def fit_models(cfg, df: pd.DataFrame):
    artifacts = Path(cfg['artifacts_dir'])
    artifacts.mkdir(parents=True, exist_ok=True)

    feature_cols = [c for c in df.columns if c != 'ts']
    with open(artifacts / 'columns.json','w') as f:
        json.dump(feature_cols, f)

    X = df[feature_cols].to_numpy(dtype=float)
    scaler = RobustScaler().fit(X)
    Xs = scaler.transform(X)
    joblib.dump(scaler, artifacts / 'scaler.joblib')

    iforest = IsolationForest(n_estimators=200, contamination=0.01, random_state=42)
    iforest.fit(Xs)
    joblib.dump(iforest, artifacts / 'iforest.joblib')

    ocsvm = OneClassSVM(kernel='rbf', nu=0.05, gamma='scale')
    ocsvm.fit(Xs)
    joblib.dump(ocsvm, artifacts / 'ocsvm.joblib')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ae = AE(in_dim=Xs.shape[1], latent=16).to(device)
    crit = nn.MSELoss()
    opt = optim.Adam(ae.parameters(), lr=float(cfg['training']['lr']))
    ds = TensorDataset(torch.tensor(Xs, dtype=torch.float32))
    val_split = float(cfg['training']['validation_split'])
    n_total = len(ds)
    n_val = int(n_total*val_split)
    n_train = n_total - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=int(cfg['training']['batch_size']), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=int(cfg['training']['batch_size']), shuffle=False)

    best_val = 1e9
    for epoch in range(int(cfg['training']['max_epochs'])):
        ae.train()
        for (xb,) in train_loader:
            xb = xb.to(device)
            opt.zero_grad()
            recon = ae(xb)
            loss = crit(recon, xb)
            loss.backward()
            opt.step()
        ae.eval()
        with torch.no_grad():
            val_losses = []
            for (xb,) in val_loader:
                xb = xb.to(device)
                recon = ae(xb)
                val_losses.append(crit(recon, xb).item())
            v = float(np.mean(val_losses)) if val_losses else 0.0
            if v < best_val:
                best_val = v
                torch.save(ae.state_dict(), artifacts / 'autoencoder.pt')

    seq_len = int(cfg['training']['seq_len'])
    if Xs.shape[0] > seq_len:
        seqs = []
        for i in range(Xs.shape[0]-seq_len):
            seqs.append(Xs[i:i+seq_len])
        import numpy as _np
        Xseq = _np.stack(seqs, axis=0)
        from .models.lstm import LSTMAE
        lstm = LSTMAE(in_dim=Xs.shape[1], hidden=64).to(device)
        crit = nn.MSELoss()
        opt = optim.Adam(lstm.parameters(), lr=float(cfg['training']['lr']))
        ds = TensorDataset(torch.tensor(Xseq, dtype=torch.float32))
        val_split = float(cfg['training']['validation_split'])
        n_total = len(ds)
        n_val = int(n_total*val_split)
        n_train = n_total - n_val
        train_ds, val_ds = random_split(ds, [n_train, n_val])
        train_loader = DataLoader(train_ds, batch_size=int(cfg['training']['batch_size']), shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=int(cfg['training']['batch_size']), shuffle=False)
        best_val = 1e9
        for epoch in range(int(cfg['training']['max_epochs'])):
            lstm.train()
            for (xb,) in train_loader:
                xb = xb.to(device)
                opt.zero_grad()
                recon = lstm(xb)
                loss = crit(recon, xb)
                loss.backward()
                opt.step()
            lstm.eval()
            with torch.no_grad():
                val_losses = []
                for (xb,) in val_loader:
                    xb = xb.to(device)
                    recon = lstm(xb)
                    val_losses.append(crit(recon, xb).item())
                v = float(np.mean(val_losses)) if val_losses else 0.0
                if v < best_val:
                    best_val = v
                    torch.save(lstm.state_dict(), artifacts / 'lstm.pt')

    from .models.ensemble import EnsembleModels
    em = EnsembleModels(
        artifacts={
            'scaler_path': str(artifacts/'scaler.joblib'),
            'columns_path': str(artifacts/'columns.json'),
            'iforest_path': str(artifacts/'iforest.joblib'),
            'ocsvm_path': str(artifacts/'ocsvm.joblib'),
            'ae_path': str(artifacts/'autoencoder.pt'),
            'lstm_path': str(artifacts/'lstm.pt'),
            'thresholds_path': str(artifacts/'thresholds.json'),
        },
        device=device,
    )

    seq_len = int(cfg['training']['seq_len'])
    Xs_list = Xs.tolist()
    scores_if, scores_oc, scores_ae, scores_lstm = [], [], [], []
    buf = []
    import numpy as _np
    for x in Xs_list:
        buf.append(x)
        if len(buf) > seq_len:
            buf = buf[-seq_len:]
        seq = _np.array(buf, dtype=float) if len(buf) == seq_len else None
        sc, _ = em.score_point(_np.array(x), seq, weights={'iforest':.25,'ocsvm':.25,'ae':.25,'lstm':.25})
        scores_if.append(sc['iforest'])
        scores_oc.append(sc['ocsvm'])
        scores_ae.append(sc['ae'])
        scores_lstm.append(sc['lstm'])
    def p99(a):
        import numpy as _np
        return float(_np.percentile(_np.array(a, dtype=float), 99))
    thresholds = {
        'iforest': p99(scores_if),
        'ocsvm': p99(scores_oc),
        'ae': p99(scores_ae),
        'lstm': p99(scores_lstm),
    }
    with open(artifacts/'thresholds.json','w') as f:
        json.dump(thresholds, f, indent=2)

    print("Artifacts saved to:", artifacts.resolve())

def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.iface:
        cfg['iface'] = args.iface
    if args.capture_seconds is not None:
        cfg['training']['capture_seconds'] = int(args.capture_seconds)

    print("[1/2] Capturing baseline ...")
    df = capture_baseline(cfg)
    print(f"Collected {len(df)} feature rows")

    print("[2/2] Fitting models ...")
    fit_models(cfg, df)

if __name__ == "__main__":
    main()
