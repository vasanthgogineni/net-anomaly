# Grafana: Flux queries (InfluxDB 2.x)

Assuming `org = my-org`, `bucket = netanoms`.

## Panel: Packets per second (short window)
```flux
from(bucket: "netanoms")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r._measurement == "network_metrics" and r._field == "5s_pps")
  |> aggregateWindow(every: 1s, fn: mean, createEmpty: false)
  |> yield(name: "pps")
```

## Panel: Bytes per second (medium window)
```flux
from(bucket: "netanoms")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r._measurement == "network_metrics" and r._field == "60s_bps")
  |> aggregateWindow(every: 5s, fn: mean, createEmpty: false)
```

## Panel: Unique src IPs (long window)
```flux
from(bucket: "netanoms")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r._measurement == "network_metrics" and r._field == "300s_srcip_unique")
  |> aggregateWindow(every: 10s, fn: mean, createEmpty: false)
```

## Panel: Anomaly score (ensemble)
```flux
from(bucket: "netanoms")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r._measurement == "anomaly_scores" and r._field == "ensemble")
  |> aggregateWindow(every: 1s, fn: mean, createEmpty: false)
```

## Panel: Model breakdown (stacked series)
```flux
from(bucket: "netanoms")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r._measurement == "anomaly_scores")
  |> filter(fn: (r) => r._field == "iforest" or r._field == "ocsvm" or r._field == "ae" or r._field == "lstm")
  |> aggregateWindow(every: 1s, fn: mean, createEmpty: false)
  |> yield(name: "models")
```
(Then select fields `iforest`, `ocsvm`, `ae`, `lstm` in Grafana's UI.)

## Threshold alert (ensemble > 1.0 for 10s)
Use Grafana alert rule:
```flux
from(bucket: "netanoms")
  |> range(start: -15m)
  |> filter(fn: (r) => r._measurement == "anomaly_scores" and r._field == "ensemble")
  |> aggregateWindow(every: 5s, fn: mean)
  |> map(fn: (r) => ({ r with _value: if r._value > 1.0 then 1.0 else 0.0 }))
  |> yield(name: "alert")
```
