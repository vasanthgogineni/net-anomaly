# AI-based Network Anomaly Detector

## Description
AI-Based Network Anomaly Detector is a real-time pipeline which can sniff packets with Scapy, build multi-window traffic features (e.g., 5s/60s/300s), and scores anomalies using an ensemble of Isolation Forest, One-Class SVM, and deep autoencoders (dense + LSTM) built with scikit-learn and PyTorch.

A short baseline capture computes per-model p99 thresholds. The online detector then normalizes and combines scores into a single ensemble signal each second.

Metrics and scores stream to InfluxDB 2.x and are visualized in Grafana.

Currently works on MacOS and Linux.

## 0) Prereqs
- Python 3.10+
- InfluxDB 2.x running locally (port 8086) and Grafana configured to read from InfluxDB

## 1) Install
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# edit config.yaml (iface, org/bucket, token, etc.)
```
> Default iface in `config.yaml` is `wlan0` for Wi‑Fi. Change as needed (e.g., `en0` on macOS, `wlp3s0`, etc.).

### Run InfluxDB
Make sure your InfluxDB instance follow these configurations, or set your own:
```bash
url: "http://localhost:8086"
org: "my-org"
bucket: "netanoms"
```
You can also use Grafana if you want, adding InfluxDB as the source.

## 2) Train on baseline traffic
```bash
# capture baseline for 10 minutes (adjust --capture-seconds)
python -m src.train --config config.yaml --iface wlan0 --capture-seconds 600
```
Artifacts will be saved under `artifacts/`.

## 3) Run detector
```bash
python -m src.detect --config config.yaml --iface wlan0
```
You should see metrics/scores flowing to InfluxDB.

## 4) Grafana
- Add InfluxDB data source (URL `http://localhost:8086`, org+token).
- Create panels using queries from `grafana/flux_queries.md`.
- Visualize `network_metrics` (e.g., `5s_pps`, `60s_bps`, unique counts) and `anomaly_scores.ensemble`.

## 5) systemd service (not required)
```bash
sudo cp systemd/net-anomaly.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now net-anomaly
```
