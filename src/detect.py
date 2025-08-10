from __future__ import annotations
import argparse
import os
import time
import yaml
import pandas as pd
import numpy as np

from .sniff import PacketSniffer
from .features import OnlineFeatureExtractor
from .models.ensemble import EnsembleModels
from .influx_writer import InfluxWriter

def parse_args():
    ap = argparse.ArgumentParser(description="Real-time anomaly detection loop")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--iface", default=None)
    return ap.parse_args()

def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.iface:
        cfg['iface'] = args.iface

    artifacts = {
        'scaler_path': cfg['scaler_path'],
        'columns_path': cfg['columns_path'],
        'iforest_path': cfg['iforest_path'],
        'ocsvm_path': cfg['ocsvm_path'],
        'ae_path': cfg['ae_path'],
        'lstm_path': cfg['lstm_path'],
        'thresholds_path': cfg['thresholds_path'],
    }
    device = 'cuda' if os.environ.get('USE_CUDA','0')=='1' else 'cpu'
    ensemble = EnsembleModels(artifacts, device=device)

    windows = cfg['windows']
    fe = OnlineFeatureExtractor(windows=windows, resolution_sec=int(cfg['feature_resolution_sec']), max_retention_sec=int(cfg['max_retention_sec']))

    sniffer = PacketSniffer(iface=cfg['iface'], bpf_filter=cfg.get('bpf_filter') or None)
    sniffer.start()

    influx_cfg = cfg['influx']
    token = influx_cfg['token']
    if isinstance(token, str) and token.startswith('${'):
        env_name = token.strip('${}').strip()
        token = os.environ.get(env_name, '')
    writer = InfluxWriter(
        url=influx_cfg['url'], org=influx_cfg['org'], bucket=influx_cfg['bucket'],
        token=token, node_name=cfg.get('node_name','node'), iface=cfg['iface']
    )

    seq_len = int(cfg['training']['seq_len'])
    seq_buf: list[np.ndarray] = []
    weights = cfg.get('ensemble_weights', {'iforest':.25,'ocsvm':.25,'ae':.25,'lstm':.25})

    next_tick = int(time.time()) + 1
    try:
        while True:
            while True:
                try:
                    pkt = sniffer.get_queue().get_nowait()
                except Exception:
                    break
                fe.on_packet(pkt)

            now = time.time()
            if now >= next_tick:
                row = fe.build_feature_row()
                ts = row.get('ts', now)
                writer.write_metrics(ts, row)

                df = pd.DataFrame([row])
                if 'ts' in df.columns:
                    df = df.drop(columns=['ts'])
                for col in ensemble.columns:
                    if col not in df.columns:
                        df[col] = 0.0
                df = df[ensemble.columns]
                x_scaled = ensemble.transform(df)[0]

                seq_buf.append(x_scaled)
                if len(seq_buf) > seq_len:
                    seq_buf = seq_buf[-seq_len:]
                seq_arr = np.array(seq_buf, dtype=float) if len(seq_buf) == seq_len else None

                scores, ens = ensemble.score_point(x_scaled, seq_arr, weights)
                writer.write_scores(ts, scores, ens)

                next_tick = int(now) + 1
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        sniffer.stop()
        writer.close()

if __name__ == "__main__":
    main()
