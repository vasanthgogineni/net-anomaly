from __future__ import annotations
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

class InfluxWriter:
    def __init__(self, url: str, org: str, bucket: str, token: str, node_name: str, iface: str):
        self.client = InfluxDBClient(url=url, token=token, org=org)
        self.bucket = bucket
        self.org = org
        self.node_name = node_name
        self.iface = iface
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)

    def write_metrics(self, ts: float, metrics: dict[str,float]):
        p = Point("network_metrics").tag("node", self.node_name).tag("iface", self.iface)
        p = p.time(int(ts*1e9))
        for k,v in metrics.items():
            if k == 'ts':
                continue
            p = p.field(k, float(v))
        self.write_api.write(bucket=self.bucket, record=p)

    def write_scores(self, ts: float, scores: dict[str,float], ensemble: float):
        p = Point("anomaly_scores").tag("node", self.node_name).tag("iface", self.iface)
        p = p.time(int(ts*1e9))
        for k,v in scores.items():
            p = p.field(k, float(v))
        p = p.field("ensemble", float(ensemble))
        self.write_api.write(bucket=self.bucket, record=p)

    def close(self):
        try:
            self.client.close()
        except Exception:
            pass
