"""Prometheus metrics untuk observabilitas pipeline RAG.

Metric didefinisikan di default registry `prometheus_client`, jadi otomatis
ter-expose lewat endpoint `/metrics` (prometheus_fastapi_instrumentator di
backend/main.py). Cukup meng-import modul ini di suatu tempat pada startup
(rag_pipeline & moderation meng-import-nya) agar metric ter-register.

Instrumentasi bersifat additive & best-effort: pemanggil membungkus .inc()/
.observe() dengan guard supaya kegagalan metric TIDAK pernah menjatuhkan request.
"""

from prometheus_client import Histogram, Counter, Gauge

# Latensi per layer pipeline (detik). Diisi dari rag_pipeline._timed().
LAYER_LATENCY = Histogram(
    "rag_layer_latency_seconds",
    "Latensi per layer pipeline",
    ["layer"],
    buckets=(.01, .05, .1, .25, .5, 1, 2, 5, 10, 20, 60),
)

# Jumlah panggilan query()/query_stream() yang sedang berjalan.
INFLIGHT = Gauge("rag_inflight_requests", "Request query() yang sedang berjalan")

# Cache lookup result: label result=hit|miss.
CACHE_HITS = Counter("rag_cache_hits_total", "Cache lookup", ["result"])

# L2 moderation di-bypass (fail-open) karena Llama Guard gagal/circuit open.
MODERATION_BYPASSED = Counter("rag_moderation_bypassed_total", "L2 dilewati karena gagal")

# Rerank TEI gagal → fallback ke urutan dense retrieval.
RERANK_FALLBACK = Counter("rag_rerank_fallback_total", "Rerank gagal, fallback dense order")
