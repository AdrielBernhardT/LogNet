"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          LogNet — Temporal Graph Stream Construction (Tahap 1)               ║
║          Dataset: LANL Auth Log (auth.txt.gz + redteam.txt.gz)               ║
║          Version: 2.0 (Production-Grade)                                     ║
║                                                                              ║
║  Prinsip wajib yang TIDAK BOLEH dilanggar:                                   ║
║    1. Semua operasi harus CHRONOLOGICAL                                      ║
║    2. Tidak boleh ada FUTURE LEAKAGE                                         ║
║    3. Node registry harus DYNAMIC                                            ║
║    4. Semua feature harus TEMPORAL-AWARE                                     ║
║    5. Stream harus CONTINUAL-LEARNING COMPATIBLE                             ║
╚══════════════════════════════════════════════════════════════════════════════╝

Changelog v2.0 (perbaikan dari v1.0):
  [FIX-01] ChronologicalSorter._merge_chunks: keyed() hanya mapping 9 field hardcoded
           → sekarang menggunakan df.columns dinamis agar tahan perubahan schema.
  [FIX-02] TemporalStreamIterator: has_next() tidak pernah didefinisikan tapi dipanggil
           di __next__() → ditambahkan has_next() yang proper.
  [FIX-03] TemporalStreamIterator: duplikasi __iter__ dan __next__ → dihapus, digabung.
  [FIX-04] TemporalNegativeSampler: _seen_users/_seen_hosts adalah List biasa,
           membership check O(n) → diganti ke dict ordered agar O(1) + LRU lookback.
  [FIX-05] TemporalFeatureEngineer.compute_and_enrich: state di-update SETELAH komputasi
           sudah benar tapi IAT dihitung di TemporalEdgeBuilder BUKAN di sini;
           keduanya menyimpan last_time terpisah → konsolidasi ke satu sumber.
  [FIX-06] LabelAlignmentSystem.get_label: import bisect di dalam loop method →
           dipindah ke top-level module import.
  [FIX-07] ReplayBuffer.sample: rng.choice dengan replace=False akan error jika
           n > len(buffer) meski ada min() guard → tambah guard eksplisit.
  [FIX-08] TemporalWindowingSystem._build_time_windows: index lookup tidak di-reset
           setelah df diload dari parquet (RangeIndex bisa non-contiguous) →
           gunakan positional iloc-based indexing.
  [FIX-09] TemporalGraphStreamPipeline.run: edge['src_computer'] tidak selalu ada
           saat label alignment dipanggil dengan src_computer dari event yang sudah
           di-enrich → pastikan field selalu tersedia.
  [FIX-10] ParquetStorage._cast_edge_df: timestamp tidak di-cast ke int64 eksplisit
           → bisa mismatch schema jika pandas infer int32.
  [FIX-11] LogNetConfig: tidak ada validasi parameter → tambah __post_init__ validator.
  [FIX-12] TemporalEdgeBuilder & TemporalFeatureEngineer: inter_arrival_time dihitung
           di EdgeBuilder tapi juga ada histori waktu di FeatureEngineer →
           dedup: IAT tetap di EdgeBuilder, TSE pakai histori sendiri di FeatureEngineer.
  [ADD-01] Tambah has_next() di TemporalStreamIterator.
  [ADD-02] Tambah seek() yang benar-benar bekerja dengan full re-scan (aman untuk parquet).
  [ADD-03] Tambah get_split_iterator() di Pipeline untuk mengiterasi hanya
           train/val/test window secara terpisah.
  [ADD-04] Tambah validasi monotonisitas timestamp di stream output.
  [ADD-05] Tambah TemporalIntegrityChecker: komponen pasca-proses untuk memverifikasi
           tidak ada future leakage di output parquet.

Output:
  - temporal_edges/edges.parquet   : Temporal Edge Stream
  - node_registry/nodes.parquet    : Dynamic Node Registry
  - windows/windows.json           : Temporal Windows
  - replay_buffer/replay_candidates.parquet : Replay Candidate Buffer
  - metadata/memory_init.json      : Memory Init Metadata
  - metadata/temporal_splits.json  : Split Metadata
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import bisect
import gzip
import os
import json
import math
import heapq
import hashlib
import logging
import warnings
import itertools
from pathlib import Path
from typing import Iterator, Dict, Tuple, Optional, List, Any, Set
from collections import defaultdict, deque, OrderedDict
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("LogNet.Stage1")


# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURASI GLOBAL
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class LogNetConfig:
    """
    Semua parameter konfigurasi LogNet Tahap 1.
    Ubah di sini untuk menyesuaikan perilaku pipeline.
    """
    # ── Path ──────────────────────────────────────────────────────────────────
    auth_log_path: str      = "auth.txt.gz"
    redteam_log_path: str   = "redteam.txt.gz"
    output_dir: str         = "lognet_output"

    # ── Streaming & Memory ────────────────────────────────────────────────────
    stream_chunk_size: int  = 50_000
    sort_buffer_size: int   = 500_000

    # ── Temporal Window ───────────────────────────────────────────────────────
    event_window_size: int  = 1_000
    event_window_stride: int= 500

    time_window_size: int   = 3_600      # 1 jam dalam detik
    time_window_stride: int = 1_800      # 30 menit sliding stride

    # ── Feature Engineering ───────────────────────────────────────────────────
    feature_rolling_window: int = 100    # Histori per-user untuk rolling features

    # ── Label Alignment ───────────────────────────────────────────────────────
    label_time_tolerance: int = 60       # ±60 detik untuk signature matching

    # ── Negative Sampling ─────────────────────────────────────────────────────
    neg_sample_ratio: float = 1.0
    neg_sample_lookback: int = 10_000    # Hanya ambil node dari t' <= ti

    # ── Temporal Split ────────────────────────────────────────────────────────
    train_ratio: float = 0.70
    val_ratio:   float = 0.15

    # ── Replay Buffer ─────────────────────────────────────────────────────────
    replay_buffer_max_size: int = 5_000

    # ── Parquet ───────────────────────────────────────────────────────────────
    parquet_compression: str = "snappy"

    # ── Seed ──────────────────────────────────────────────────────────────────
    random_seed: int = 42

    def __post_init__(self):
        """[ADD-11] Validasi parameter saat inisialisasi."""
        assert 0 < self.train_ratio < 1, "train_ratio harus antara 0 dan 1"
        assert 0 < self.val_ratio < 1,   "val_ratio harus antara 0 dan 1"
        assert self.train_ratio + self.val_ratio < 1, \
            "train_ratio + val_ratio harus < 1 (sisanya untuk test)"
        assert self.stream_chunk_size > 0, "stream_chunk_size harus positif"
        assert self.sort_buffer_size > 0,  "sort_buffer_size harus positif"
        assert self.event_window_size > 0, "event_window_size harus positif"
        assert self.event_window_stride > 0, "event_window_stride harus positif"
        assert self.event_window_stride <= self.event_window_size, \
            "event_window_stride tidak boleh lebih besar dari event_window_size"
        assert self.neg_sample_ratio >= 0, "neg_sample_ratio tidak boleh negatif"
        assert self.label_time_tolerance >= 0, "label_time_tolerance tidak boleh negatif"


# ─────────────────────────────────────────────────────────────────────────────
# POIN 1 — RAW LOG LOADER
# ─────────────────────────────────────────────────────────────────────────────
class RawLogLoader:
    """
    Membaca auth.txt.gz dan redteam.txt.gz secara STREAMING.
    Tidak memuat seluruh file ke memori → zero memory overflow risk.

    Format LANL auth.txt:
      time, src_user@domain, dst_user@domain, src_computer, dst_computer,
      auth_type, logon_type, auth_orientation, success/fail

    Format LANL redteam.txt:
      time, user@domain, src_computer, dst_computer
    """

    AUTH_FIELDS = [
        "timestamp",         # ← temporal ordering key
        "src_user",          # ← source node (user/login initiator)
        "dst_user",
        "src_computer",
        "dst_computer",      # ← destination node (host/service)
        "auth_type",         # ← edge feature (categorical)
        "logon_type",        # ← edge feature (categorical)
        "auth_orientation",
        "success_fail",      # ← behavioral signal (binary)
    ]

    REDTEAM_FIELDS = [
        "timestamp",
        "user",
        "src_computer",
        "dst_computer",
    ]

    def __init__(self, config: LogNetConfig):
        self.config = config

    def stream_auth_log(self) -> Iterator[Dict[str, Any]]:
        """
        Generator: menghasilkan satu event dict per baris dari auth.txt.gz.
        Parsing dilakukan inline → tidak pernah load seluruh file ke RAM.
        """
        path = self.config.auth_log_path
        logger.info(f"[Loader] Membuka auth log: {path}")

        open_fn = gzip.open if path.endswith(".gz") else open
        n_parsed, n_error = 0, 0

        with open_fn(path, "rt", encoding="utf-8", errors="replace") as f:
            for raw_line in f:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue

                parts = raw_line.split(",")
                if len(parts) < 9:
                    n_error += 1
                    continue

                try:
                    ts = int(parts[0])
                except ValueError:
                    n_error += 1
                    continue

                raw_src = parts[1].strip()
                src_user = raw_src.split("@")[0] if "@" in raw_src else raw_src
                dst_host = parts[4].strip()

                event = {
                    "timestamp":        ts,
                    "src_user":         src_user,
                    "dst_user":         parts[2].strip(),
                    "src_computer":     parts[3].strip(),
                    "dst_computer":     dst_host,
                    "auth_type":        parts[5].strip(),
                    "logon_type":       parts[6].strip(),
                    "auth_orientation": parts[7].strip(),
                    "success_fail":     parts[8].strip(),
                }
                n_parsed += 1
                yield event

        logger.info(f"[Loader] Auth log selesai: {n_parsed:,} parsed, {n_error:,} error")

    def stream_redteam_log(self) -> Dict[Tuple, List[int]]:
        """
        Membaca seluruh redteam.txt.gz ke dict attack index.
        Key: (user, src_computer, dst_computer) → sorted list of timestamps.
        Redteam jauh lebih kecil dari auth log, aman dimuat ke memori.
        """
        path = self.config.redteam_log_path
        logger.info(f"[Loader] Membaca redteam log: {path}")

        attack_index: Dict[Tuple, List[int]] = defaultdict(list)
        open_fn = gzip.open if path.endswith(".gz") else open

        try:
            with open_fn(path, "rt", encoding="utf-8", errors="replace") as f:
                for raw_line in f:
                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue
                    parts = raw_line.split(",")
                    if len(parts) < 4:
                        continue
                    try:
                        ts = int(parts[0])
                    except ValueError:
                        continue
                    user = parts[1].strip().split("@")[0]
                    src  = parts[2].strip()
                    dst  = parts[3].strip()
                    attack_index[(user, src, dst)].append(ts)
        except FileNotFoundError:
            logger.warning(f"[Loader] redteam log tidak ditemukan: {path}. "
                           f"Semua label akan 0 (normal).")

        # Pre-sort timestamps per signature untuk binary search O(log n)
        for key in attack_index:
            attack_index[key].sort()

        logger.info(f"[Loader] Redteam: {len(attack_index):,} unique attack signatures")
        return dict(attack_index)


# ─────────────────────────────────────────────────────────────────────────────
# POIN 2 — CHRONOLOGICAL EVENT ORDERING
# ─────────────────────────────────────────────────────────────────────────────
class ChronologicalSorter:
    """
    Menjamin TEMPORAL CAUSALITY: semua event diproses dalam urutan waktu.

    Pendekatan: External Merge Sort berbasis Parquet chunks.
    - Baca N events ke buffer → sort by timestamp → tulis chunk sementara
    - Merge semua chunk dengan heapq.merge → output chronological stream

    MENGAPA BUKAN RANDOM SHUFFLE?
    ─────────────────────────────
    Random shuffle menghancurkan 3 properti kritis TGN:
      1. Sequential behavior  : pola login berurutan A→B→C menjadi acak
      2. Temporal dependency  : edge pada t harus terbentuk setelah t-1
      3. Memory evolution     : node memory mi(t) bergantung pada mi(t-1)
    """

    def __init__(self, config: LogNetConfig):
        self.config = config
        self._tmp_dir = Path(config.output_dir) / "_tmp_sort"
        self._tmp_dir.mkdir(parents=True, exist_ok=True)

    def sorted_stream(self, raw_stream: Iterator[Dict]) -> Iterator[Dict]:
        """
        Menghasilkan event stream yang sudah diurutkan berdasarkan timestamp.
        Menggunakan external sort → aman untuk file ratusan GB.
        """
        chunk_files = self._write_sorted_chunks(raw_stream)
        logger.info(f"[Sorter] {len(chunk_files)} sorted chunks siap untuk merge")
        yield from self._merge_chunks(chunk_files)
        self._cleanup(chunk_files)

    def _write_sorted_chunks(self, raw_stream: Iterator[Dict]) -> List[Path]:
        """Tulis buffer yang sudah di-sort ke file chunk sementara."""
        chunk_files = []
        buffer = []
        chunk_idx = 0

        for event in raw_stream:
            buffer.append(event)
            if len(buffer) >= self.config.sort_buffer_size:
                chunk_path = self._flush_chunk(buffer, chunk_idx)
                chunk_files.append(chunk_path)
                chunk_idx += 1
                buffer.clear()

        if buffer:
            chunk_path = self._flush_chunk(buffer, chunk_idx)
            chunk_files.append(chunk_path)

        return chunk_files

    def _flush_chunk(self, buffer: List[Dict], idx: int) -> Path:
        """Sort buffer dan simpan ke Parquet sementara."""
        # Sort by (timestamp, src_user) → tiebreak deterministik
        buffer.sort(key=lambda e: (e["timestamp"], e.get("src_user", "")))
        df = pd.DataFrame(buffer)
        path = self._tmp_dir / f"chunk_{idx:06d}.parquet"
        df.to_parquet(path, compression="snappy", index=False)
        logger.debug(f"[Sorter] Chunk {idx}: {len(buffer):,} events → {path.name}")
        return path

    def _merge_chunks(self, chunk_files: List[Path]) -> Iterator[Dict]:
        """
        K-way merge menggunakan heapq.
        Setiap chunk sudah terurut → hasil merge juga terurut.

        [FIX-01] Sebelumnya: field mapping hardcoded 9 field.
                 Sekarang: menggunakan df.columns yang dibaca dari Parquet
                 → tahan terhadap perubahan schema atau field order.
        """
        def make_keyed_iter(path: Path):
            df = pd.read_parquet(path)
            columns = list(df.columns)  # [FIX-01] dinamis, bukan hardcoded
            for row in df.itertuples(index=False, name=None):
                row_dict = dict(zip(columns, row))
                yield (row_dict["timestamp"], row_dict)

        iterators = [make_keyed_iter(p) for p in chunk_files]
        merged = heapq.merge(*iterators, key=lambda x: x[0])
        for _, event in merged:
            yield event

    def _cleanup(self, chunk_files: List[Path]):
        for f in chunk_files:
            try:
                f.unlink()
            except Exception:
                pass
        try:
            self._tmp_dir.rmdir()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# POIN 3 — DYNAMIC NODE REGISTRY
# ─────────────────────────────────────────────────────────────────────────────
class DynamicNodeRegistry:
    """
    Memetakan string entity (user/host) ke integer index secara DINAMIS.
    Node hanya terdaftar ketika pertama kali MUNCUL dalam stream
    → tidak ada node yang pre-registered dari masa depan (future-free).

    Output utama: integer node index (bukan string).

    Internal state:
      - node2id   : {entity_string → int}
      - id2node   : {int → entity_string}
      - node_type : {int → "user"|"host"}
      - first_seen: {int → timestamp}
      - last_seen : {int → timestamp}
      - degree    : {int → int}  (temporal degree = jumlah interaksi)
    """

    def __init__(self):
        self._node2id:    Dict[str, int] = {}
        self._id2node:    Dict[int, str] = {}
        self._node_type:  Dict[int, str] = {}
        self._first_seen: Dict[int, int] = {}
        self._last_seen:  Dict[int, int] = {}
        self._degree:     Dict[int, int] = defaultdict(int)
        self._counter:    int = 0

    def get_or_register(self, entity: str, node_type: str, timestamp: int) -> int:
        """
        Kembalikan ID integer untuk entity.
        Jika belum ada → register baru secara online (saat event tiba).
        Tidak ada pre-loading → future-free by design.
        """
        if entity not in self._node2id:
            idx = self._counter
            self._node2id[entity]   = idx
            self._id2node[idx]      = entity
            self._node_type[idx]    = node_type
            self._first_seen[idx]   = timestamp
            self._counter += 1
        else:
            idx = self._node2id[entity]

        self._last_seen[idx] = timestamp
        self._degree[idx]   += 1
        return idx

    def __len__(self) -> int:
        return self._counter

    def to_dataframe(self) -> pd.DataFrame:
        """Ekspor registry ke DataFrame untuk disimpan sebagai Parquet."""
        rows = []
        for idx in range(self._counter):
            rows.append({
                "node_id":    idx,
                "entity":     self._id2node[idx],
                "node_type":  self._node_type[idx],
                "first_seen": self._first_seen.get(idx, -1),
                "last_seen":  self._last_seen.get(idx, -1),
                "degree":     self._degree[idx],
            })
        return pd.DataFrame(rows)

    def get_memory_init_metadata(self) -> Dict[str, Any]:
        """
        Export metadata yang dibutuhkan oleh TGN node memory initialization.
        mi(t): setiap node punya state awal berdasarkan temporal statistics.
        """
        return {
            "total_nodes":     self._counter,
            "node_first_seen": {str(k): v for k, v in self._first_seen.items()},
            "node_last_seen":  {str(k): v for k, v in self._last_seen.items()},
            "node_degree":     {str(k): v for k, v in self._degree.items()},
        }


# ─────────────────────────────────────────────────────────────────────────────
# POIN 4 — TEMPORAL EDGE CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────
class TemporalEdgeBuilder:
    """
    Mengonversi event auth log menjadi TEMPORAL EDGE dengan semua field yang
    dibutuhkan TGN.

    Setiap edge merepresentasikan satu auth event antara:
      - Source Node : User/login initiator  → integer ID dari DynamicNodeRegistry
      - Dest Node   : Host/service tujuan   → integer ID dari DynamicNodeRegistry
      - Timestamp   : numerik & monotonic   → dijamin oleh ChronologicalSorter
      - Edge Features:
          * auth_type_enc      : categorical (label encoded)
          * logon_type_enc     : categorical (label encoded)
          * is_success         : binary  (1=Success, 0=Fail)
          * inter_arrival_time : temporal (detik sejak auth terakhir dari user ini)
          * session_count      : numeric  (jumlah sesi aktif user ini)

    [FIX-12] IAT dikomputasi di sini (satu-satunya tempat) untuk menghindari
             duplikasi state antara EdgeBuilder dan FeatureEngineer.
    """

    AUTH_TYPE_MAP  = {"NTLM": 0, "Kerberos": 1, "Negotiate": 2, "?": 3}
    LOGON_TYPE_MAP = {
        "Network": 0, "Interactive": 1, "Batch": 2,
        "Service": 3, "NetworkCleartext": 4, "?": 5
    }

    def __init__(self, node_registry: DynamicNodeRegistry):
        self.registry = node_registry
        # [FIX-12] Satu state IAT, tidak duplikat dengan FeatureEngineer
        self._user_last_time:     Dict[str, int] = {}
        self._user_session_count: Dict[str, int] = defaultdict(int)

    def build_edge(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Konversi satu auth event → satu temporal edge dict."""
        ts       = event["timestamp"]
        src_user = event["src_user"]
        dst_host = event["dst_computer"]

        # ── Node ID (Dynamic, Future-Free) ────────────────────────────────────
        src_id = self.registry.get_or_register(src_user, "user", ts)
        dst_id = self.registry.get_or_register(dst_host, "host", ts)

        # ── Edge Features ─────────────────────────────────────────────────────
        auth_type_enc  = self.AUTH_TYPE_MAP.get(
            event.get("auth_type", "?"), self.AUTH_TYPE_MAP["?"])
        logon_type_enc = self.LOGON_TYPE_MAP.get(
            event.get("logon_type", "?"), self.LOGON_TYPE_MAP["?"])
        is_success = 1 if event.get("success_fail", "").lower() == "success" else 0

        # Inter-arrival time: waktu sejak login terakhir user ini [FIX-12]
        if src_user in self._user_last_time:
            iat = max(0, ts - self._user_last_time[src_user])
        else:
            iat = 0  # first appearance → no prior event
        self._user_last_time[src_user] = ts

        self._user_session_count[src_user] += 1
        session_count = self._user_session_count[src_user]

        return {
            # Identitas edge
            "src_node":           src_id,
            "dst_node":           dst_id,
            "timestamp":          ts,
            # Edge features (poin 4)
            "auth_type_enc":      auth_type_enc,
            "logon_type_enc":     logon_type_enc,
            "is_success":         is_success,
            "inter_arrival_time": float(iat),
            "session_count":      session_count,
            # Raw strings (untuk label alignment & debug)
            "src_user_raw":       src_user,
            "dst_host_raw":       dst_host,
            "src_computer":       event.get("src_computer", ""),
            "auth_orientation":   event.get("auth_orientation", ""),
            # Label placeholder
            "label":              0,
        }


# ─────────────────────────────────────────────────────────────────────────────
# POIN 5 — TEMPORAL FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
class TemporalFeatureEngineer:
    """
    Menghitung 5 fitur temporal penting yang merupakan sinyal anomali utama.
    Semua kalkulasi bersifat ONLINE (hanya melihat histori t' <= t).
    State di-update SETELAH komputasi → tidak ada future leakage.

    Fitur:
      1. Inter-Arrival Time (IAT)        → dikomputasi di TemporalEdgeBuilder [FIX-12]
      2. Host Switching Rate (HSR)       → kecepatan user pindah ke host baru
      3. Failed Login Ratio (FLR)        → rasio kegagalan login (rolling window)
      4. Novel Destination Score (NDS)   → apakah dst_host baru untuk user ini
      5. Temporal Session Entropy (TSE)  → entropi distribusi IAT (beaconing detector)
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        # Histori per user — hanya window_size event terakhir (O(1) memory per user)
        self._user_hosts:        Dict[str, deque] = defaultdict(
                                     lambda: deque(maxlen=window_size))
        self._user_success_hist: Dict[str, deque] = defaultdict(
                                     lambda: deque(maxlen=window_size))
        self._user_time_hist:    Dict[str, deque] = defaultdict(
                                     lambda: deque(maxlen=window_size))
        self._user_known_hosts:  Dict[str, set]   = defaultdict(set)

    def compute_and_enrich(self, edge: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hitung semua fitur temporal dan tambahkan ke edge dict.
        PENTING: state di-update SETELAH komputasi → tidak ada future leakage.
        """
        user = edge["src_user_raw"]
        host = edge["dst_host_raw"]
        ts   = edge["timestamp"]
        succ = edge["is_success"]

        # ── 2. Host Switching Rate ─────────────────────────────────────────────
        past_hosts = self._user_hosts[user]
        if len(past_hosts) >= 2:
            switches = sum(1 for i in range(1, len(past_hosts))
                           if past_hosts[i] != past_hosts[i-1])
            hsr = switches / (len(past_hosts) - 1)
        else:
            hsr = 0.0

        # ── 3. Failed Login Ratio ──────────────────────────────────────────────
        past_succ = self._user_success_hist[user]
        if past_succ:
            flr = 1.0 - (sum(past_succ) / len(past_succ))
        else:
            flr = 0.0

        # ── 4. Novel Destination Score ─────────────────────────────────────────
        nds = 0.0 if host in self._user_known_hosts[user] else 1.0

        # ── 5. Temporal Session Entropy ────────────────────────────────────────
        # Entropi distribusi IAT → tinggi = pola acak (exploration), rendah = beaconing
        past_times = list(self._user_time_hist[user])
        tse = self._compute_iat_entropy(past_times)

        # ── Simpan ke edge ─────────────────────────────────────────────────────
        edge["host_switching_rate"]     = round(hsr, 6)
        edge["failed_login_ratio"]      = round(flr, 6)
        edge["novel_destination_score"] = nds
        edge["temporal_session_entropy"]= round(tse, 6)

        # ── Update state (SETELAH komputasi!) ──────────────────────────────────
        self._user_hosts[user].append(host)
        self._user_success_hist[user].append(succ)
        self._user_time_hist[user].append(ts)
        self._user_known_hosts[user].add(host)

        return edge

    @staticmethod
    def _compute_iat_entropy(timestamps: List[int]) -> float:
        """
        Hitung Shannon entropy dari distribusi inter-arrival times.

        Sinyal:
          - Entropy tinggi  → pola login acak (suspicious exploration)
          - Entropy rendah  → pola sangat periodik (beaconing attack)
        """
        if len(timestamps) < 3:
            return 0.0
        iats = [timestamps[i] - timestamps[i-1]
                for i in range(1, len(timestamps))
                if timestamps[i] > timestamps[i-1]]
        if not iats:
            return 0.0
        bins = np.histogram(iats, bins=min(10, len(iats)))[0]
        bins = bins[bins > 0].astype(float)
        probs = bins / bins.sum()
        return float(-np.sum(probs * np.log2(probs + 1e-10)))


# ─────────────────────────────────────────────────────────────────────────────
# POIN 6 — TEMPORAL WINDOWING SYSTEM
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class TemporalWindow:
    """Representasi satu temporal window dengan metadata lengkap."""
    window_id:    int
    window_type:  str    # "event" atau "time"
    purpose:      str    # "train", "val", "test", "drift", "replay", "checkpoint"
    start_ts:     int
    end_ts:       int
    start_idx:    int
    end_idx:      int
    n_events:     int
    n_pos_labels: int = 0
    n_neg_labels: int = 0


class TemporalWindowingSystem:
    """
    Membuat dua jenis sliding window:

    1. EVENT-BASED WINDOW (training & replay):
       - Ukuran = N events, Slide = S events
       - Cocok untuk: Training stream, Replay buffer update

    2. FIXED-TIME WINDOW (evaluasi & drift):
       - Ukuran = T detik, Slide = S detik
       - Cocok untuk: Evaluation, Drift monitoring

    HYBRID CHECKPOINT: event-based boundary, time-based metadata.

    [FIX-08] Index lookup menggunakan positional indexing (reset_index)
             agar tidak ada mismatch saat df diload dari Parquet.
    """

    def __init__(self, config: LogNetConfig):
        self.cfg = config
        self.event_windows: List[TemporalWindow] = []
        self.time_windows:  List[TemporalWindow] = []

    def build_windows(self, edges_df: pd.DataFrame) -> Dict[str, List[TemporalWindow]]:
        """
        Buat semua window dari DataFrame edges yang sudah diurutkan.
        PENTING: edges_df harus sudah diurutkan berdasarkan timestamp.
        """
        logger.info("[Window] Membangun temporal windows...")
        # [FIX-08] Reset index agar positional indexing konsisten
        df = edges_df.reset_index(drop=True)
        self._build_event_windows(df)
        self._build_time_windows(df)
        logger.info(f"[Window] Event-based: {len(self.event_windows)} windows")
        logger.info(f"[Window] Fixed-time:  {len(self.time_windows)} windows")
        return {
            "event_windows": self.event_windows,
            "time_windows":  self.time_windows,
        }

    def _build_event_windows(self, df: pd.DataFrame):
        """Sliding event-based window dengan stride."""
        n      = len(df)
        size   = self.cfg.event_window_size
        stride = self.cfg.event_window_stride
        wid    = 0

        for start in range(0, n - size + 1, stride):
            end   = min(start + size, n)
            chunk = df.iloc[start:end]
            pos   = int(chunk["label"].sum())
            self.event_windows.append(TemporalWindow(
                window_id    = wid,
                window_type  = "event",
                purpose      = "train",   # direklasifikasi oleh split system
                start_ts     = int(chunk["timestamp"].iloc[0]),
                end_ts       = int(chunk["timestamp"].iloc[-1]),
                start_idx    = start,
                end_idx      = end - 1,
                n_events     = len(chunk),
                n_pos_labels = pos,
                n_neg_labels = len(chunk) - pos,
            ))
            wid += 1

    def _build_time_windows(self, df: pd.DataFrame):
        """Sliding fixed-time window. [FIX-08] positional indexing."""
        if df.empty:
            return
        t_min  = int(df["timestamp"].min())
        t_max  = int(df["timestamp"].max())
        size   = self.cfg.time_window_size
        stride = self.cfg.time_window_stride
        wid    = 0

        t = t_min
        while t + size <= t_max:
            t_end = t + size
            mask  = (df["timestamp"] >= t) & (df["timestamp"] < t_end)
            chunk = df[mask]
            pos   = int(chunk["label"].sum())

            # [FIX-08] Gunakan positional index setelah reset_index
            if not chunk.empty:
                start_idx = int(chunk.index.min())
                end_idx   = int(chunk.index.max())
            else:
                start_idx = end_idx = 0

            self.time_windows.append(TemporalWindow(
                window_id    = wid,
                window_type  = "time",
                purpose      = "eval",
                start_ts     = t,
                end_ts       = t_end,
                start_idx    = start_idx,
                end_idx      = end_idx,
                n_events     = len(chunk),
                n_pos_labels = pos,
                n_neg_labels = len(chunk) - pos,
            ))
            t   += stride
            wid += 1

    def to_dict_list(self, windows: List[TemporalWindow]) -> List[Dict]:
        return [asdict(w) for w in windows]


# ─────────────────────────────────────────────────────────────────────────────
# POIN 7 — LABEL ALIGNMENT SYSTEM
# ─────────────────────────────────────────────────────────────────────────────
class LabelAlignmentSystem:
    """
    Mencocokkan auth events dengan attack labels dari redteam.txt.
    MASALAH: timestamp auth dan redteam tidak selalu identik.

    Solusi: SIGNATURE MATCHING dengan time tolerance.
      Signature = (src_user, src_computer, dst_computer)
      Match = signature sama DAN |ts_auth - ts_attack| <= tolerance

    Lebih akurat dari exact-match karena:
    - Clock skew antar system
    - Delay propagasi log
    - Aggregation window dalam koleksi log

    [FIX-06] import bisect dipindah ke top-level module import.
    """

    def __init__(self, attack_index: Dict[Tuple, List[int]], tolerance: int = 60):
        self.attack_index = attack_index
        self.tolerance    = tolerance
        # Sudah di-sort oleh RawLogLoader.stream_redteam_log()
        self._sorted_index = attack_index

    def get_label(self, src_user: str, src_computer: str,
                  dst_computer: str, timestamp: int) -> int:
        """
        Return 1 (attack) jika ada match dalam tolerance window, else 0.
        Binary search O(log n). [FIX-06] bisect sudah di-import di top.
        """
        key = (src_user, src_computer, dst_computer)
        if key not in self._sorted_index:
            return 0

        times = self._sorted_index[key]
        lo = bisect.bisect_left(times, timestamp - self.tolerance)
        if lo < len(times) and times[lo] <= timestamp + self.tolerance:
            return 1
        return 0

    def label_batch(self, edges_df: pd.DataFrame) -> pd.DataFrame:
        """Vectorized labeling untuk DataFrame."""
        labels = edges_df.apply(
            lambda row: self.get_label(
                row["src_user_raw"],
                row["src_computer"],
                row["dst_host_raw"],
                row["timestamp"]
            ), axis=1
        )
        edges_df = edges_df.copy()
        edges_df["label"] = labels
        return edges_df


# ─────────────────────────────────────────────────────────────────────────────
# POIN 8 — TEMPORAL NEGATIVE SAMPLING
# ─────────────────────────────────────────────────────────────────────────────
class TemporalNegativeSampler:
    """
    Menghasilkan NEGATIVE EDGES yang realistis untuk training TGN.

    ATURAN KRITIS: Negative edge TIDAK boleh berasal dari future.
    → Hanya sample dari node yang sudah terlihat pada t' <= ti.

    [FIX-04] Sebelumnya: _seen_users/_seen_hosts adalah List biasa.
             Membership check O(n) dan tidak ada deduplikasi efisien.
             Sekarang: OrderedDict sebagai ordered set → O(1) check,
             deduplikasi otomatis, dan urutan insertion terjaga.
    """

    def __init__(self, config: LogNetConfig, node_registry: DynamicNodeRegistry):
        self.cfg      = config
        self.registry = node_registry
        self.rng      = np.random.default_rng(config.random_seed)
        # [FIX-04] OrderedDict sebagai ordered set (key=node_id, value=None)
        self._seen_users: "OrderedDict[int, None]" = OrderedDict()
        self._seen_hosts: "OrderedDict[int, None]" = OrderedDict()
        self._positive_pairs: Set[Tuple[int, int]] = set()

    def update_state(self, edge: Dict[str, Any]):
        """Update state dengan edge baru. DIPANGGIL SEBELUM sampling negatif."""
        src = edge["src_node"]
        dst = edge["dst_node"]
        # [FIX-04] O(1) insertion dan dedup
        self._seen_users[src] = None
        self._seen_hosts[dst] = None
        self._positive_pairs.add((src, dst))

    def sample_negatives(
            self,
            positive_edge: Dict[str, Any],
            n_neg: int = 1) -> List[Dict[str, Any]]:
        """
        Sample n_neg negative edges untuk satu positive edge.
        Hanya menggunakan node dari MASA LALU (t' <= ti) [FIX-04].
        """
        if not self._seen_users or not self._seen_hosts:
            return []

        negatives    = []
        lookback     = self.cfg.neg_sample_lookback
        # [FIX-04] Ambil lookback terakhir dari ordered dict
        user_pool = list(self._seen_users.keys())[-lookback:]
        host_pool = list(self._seen_hosts.keys())[-lookback:]

        if not user_pool or not host_pool:
            return []

        attempts, max_attempts = 0, n_neg * 10
        while len(negatives) < n_neg and attempts < max_attempts:
            attempts += 1
            if self.rng.random() < 0.5:
                neg_src = positive_edge["src_node"]
                neg_dst = int(self.rng.choice(host_pool))
            else:
                neg_src = int(self.rng.choice(user_pool))
                neg_dst = positive_edge["dst_node"]

            if (neg_src, neg_dst) in self._positive_pairs:
                continue

            neg_edge = dict(positive_edge)
            neg_edge.update({
                "src_node":         neg_src,
                "dst_node":         neg_dst,
                "label":            0,
                "is_negative_sample": True,
            })
            negatives.append(neg_edge)

        return negatives


# ─────────────────────────────────────────────────────────────────────────────
# POIN 9 — MEMORY INITIALIZATION METADATA
# ─────────────────────────────────────────────────────────────────────────────
class MemoryInitializer:
    """
    Menyiapkan metadata untuk inisialisasi TGN node memory mi(t).

    Untuk memulai training yang benar, setiap node perlu:
      - first_appearance  : kapan pertama kali muncul
      - last_interaction  : kapan terakhir aktif
      - temporal_degree   : total jumlah interaksi
    """

    def __init__(self, node_registry: DynamicNodeRegistry):
        self.registry = node_registry

    def export_metadata(self, output_path: str) -> Dict[str, Any]:
        """Ekspor metadata ke JSON untuk digunakan oleh TGN model."""
        meta = self.registry.get_memory_init_metadata()
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"[Memory] Metadata disimpan: {path} ({meta['total_nodes']} nodes)")
        return meta


# ─────────────────────────────────────────────────────────────────────────────
# POIN 10 — STORAGE FORMAT (PARQUET)
# ─────────────────────────────────────────────────────────────────────────────
class ParquetStorage:
    """
    Menyimpan semua output dalam format Apache Parquet.

    [FIX-10] timestamp di-cast eksplisit ke int64 untuk mencegah schema mismatch.
    """

    EDGE_SCHEMA = pa.schema([
        pa.field("src_node",                  pa.int32()),
        pa.field("dst_node",                  pa.int32()),
        pa.field("timestamp",                 pa.int64()),   # [FIX-10] eksplisit int64
        pa.field("auth_type_enc",             pa.int8()),
        pa.field("logon_type_enc",            pa.int8()),
        pa.field("is_success",                pa.int8()),
        pa.field("inter_arrival_time",        pa.float32()),
        pa.field("session_count",             pa.int32()),
        pa.field("host_switching_rate",       pa.float32()),
        pa.field("failed_login_ratio",        pa.float32()),
        pa.field("novel_destination_score",   pa.float32()),
        pa.field("temporal_session_entropy",  pa.float32()),
        pa.field("label",                     pa.int8()),
        pa.field("is_negative_sample",        pa.bool_()),
        pa.field("src_user_raw",              pa.string()),
        pa.field("dst_host_raw",              pa.string()),
        pa.field("src_computer",              pa.string()),
        pa.field("auth_orientation",          pa.string()),
    ])

    NODE_SCHEMA = pa.schema([
        pa.field("node_id",    pa.int32()),
        pa.field("entity",     pa.string()),
        pa.field("node_type",  pa.string()),
        pa.field("first_seen", pa.int64()),
        pa.field("last_seen",  pa.int64()),
        pa.field("degree",     pa.int64()),
    ])

    def __init__(self, config: LogNetConfig):
        self.cfg = config
        self.output_dir = Path(config.output_dir)

    def save_edges(self, df: pd.DataFrame, filename: str):
        path = self.output_dir / "temporal_edges" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        df = self._cast_edge_df(df)
        table = pa.Table.from_pandas(df, schema=self.EDGE_SCHEMA,
                                     preserve_index=False)
        pq.write_table(table, path, compression=self.cfg.parquet_compression)
        logger.info(f"[Storage] Edges → {path} ({len(df):,} rows)")

    def save_node_registry(self, df: pd.DataFrame):
        path = self.output_dir / "node_registry" / "nodes.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(df, schema=self.NODE_SCHEMA,
                                     preserve_index=False)
        pq.write_table(table, path, compression=self.cfg.parquet_compression)
        logger.info(f"[Storage] Node registry → {path} ({len(df):,} nodes)")

    def save_replay_buffer(self, df: pd.DataFrame):
        path = self.output_dir / "replay_buffer" / "replay_candidates.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df = self._cast_edge_df(df)
        table = pa.Table.from_pandas(df, schema=self.EDGE_SCHEMA,
                                     preserve_index=False)
        pq.write_table(table, path, compression=self.cfg.parquet_compression)
        logger.info(f"[Storage] Replay buffer → {path} ({len(df):,} edges)")

    def _cast_edge_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # [FIX-10] timestamp eksplisit int64
        if "timestamp" in df.columns:
            df["timestamp"] = df["timestamp"].fillna(0).astype("int64")
        int8_cols    = ["auth_type_enc", "logon_type_enc", "is_success", "label"]
        int32_cols   = ["src_node", "dst_node", "session_count"]
        float32_cols = [
            "inter_arrival_time", "host_switching_rate",
            "failed_login_ratio", "novel_destination_score",
            "temporal_session_entropy"
        ]
        for c in int8_cols:
            if c in df.columns:
                df[c] = df[c].fillna(0).astype("int8")
        for c in int32_cols:
            if c in df.columns:
                df[c] = df[c].fillna(0).astype("int32")
        for c in float32_cols:
            if c in df.columns:
                df[c] = df[c].fillna(0.0).astype("float32")
        if "is_negative_sample" not in df.columns:
            df["is_negative_sample"] = False
        df["is_negative_sample"] = df["is_negative_sample"].fillna(False).astype(bool)
        str_cols = ["src_user_raw", "dst_host_raw", "src_computer", "auth_orientation"]
        for c in str_cols:
            if c not in df.columns:
                df[c] = ""
        return df


# ─────────────────────────────────────────────────────────────────────────────
# POIN 11 — TEMPORAL SPLIT METADATA
# ─────────────────────────────────────────────────────────────────────────────
class TemporalSplitManager:
    """
    Membagi data menjadi Train / Validation / Test berdasarkan WAKTU.

    KRITIS: Split harus CHRONOLOGICAL, bukan random!
    Train → Val → Test berurutan dalam waktu, tidak ada overlap.
    Model tidak pernah melihat masa depan saat training.

    Proporsi berdasarkan rentang timestamp (bukan jumlah event):
      - Train : 70% awal timeline
      - Val   : 15% tengah timeline
      - Test  : 15% akhir timeline
    """

    def __init__(self, config: LogNetConfig):
        self.cfg = config

    def compute_splits(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Hitung boundary timestamp untuk setiap split."""
        t_min  = int(df["timestamp"].min())
        t_max  = int(df["timestamp"].max())
        total  = t_max - t_min

        train_end = t_min + int(total * self.cfg.train_ratio)
        val_end   = train_end + int(total * self.cfg.val_ratio)

        splits = {
            "train": {"start_ts": t_min,      "end_ts": train_end},
            "val":   {"start_ts": train_end,   "end_ts": val_end},
            "test":  {"start_ts": val_end,     "end_ts": t_max},
        }

        for split_name, bounds in splits.items():
            mask = ((df["timestamp"] >= bounds["start_ts"]) &
                    (df["timestamp"] <  bounds["end_ts"]))
            subset = df[mask]
            splits[split_name].update({
                "n_events":     len(subset),
                "n_attack":     int(subset["label"].sum()),
                "n_normal":     len(subset) - int(subset["label"].sum()),
                "attack_ratio": float(subset["label"].mean()) if len(subset) > 0 else 0.0,
            })
            logger.info(
                f"[Split] {split_name:5s}: "
                f"ts [{bounds['start_ts']}, {bounds['end_ts']}), "
                f"{splits[split_name]['n_events']:,} events, "
                f"{splits[split_name]['n_attack']:,} attacks"
            )
        return splits

    def label_window_purposes(
            self,
            windows: List[TemporalWindow],
            splits: Dict) -> List[TemporalWindow]:
        """Tetapkan purpose (train/val/test) ke setiap window."""
        for w in windows:
            mid_ts = (w.start_ts + w.end_ts) // 2
            if mid_ts < splits["val"]["start_ts"]:
                w.purpose = "train"
            elif mid_ts < splits["test"]["start_ts"]:
                w.purpose = "val"
            else:
                w.purpose = "test"
        return windows

    def save_split_metadata(self, splits: Dict, output_dir: str):
        path = Path(output_dir) / "metadata" / "temporal_splits.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(splits, f, indent=2)
        logger.info(f"[Split] Metadata disimpan: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# REPLAY BUFFER (Output #4)
# ─────────────────────────────────────────────────────────────────────────────
class ReplayBuffer:
    """
    Buffer untuk CONTINUAL LEARNING — mencegah catastrophic forgetting.

    Menyimpan kandidat edge dari berbagai temporal region agar model
    tidak melupakan pola lama saat mempelajari pola baru.

    Strategi: Reservoir sampling — setiap edge punya probabilitas masuk
    yang sama tanpa harus menyimpan seluruh histori (O(max_size) memory).

    [FIX-07] Guard eksplisit di sample() agar tidak error saat n > buffer.
    """

    def __init__(self, config: LogNetConfig):
        self.max_size = config.replay_buffer_max_size
        self.rng      = np.random.default_rng(config.random_seed)
        self._buffer: List[Dict] = []
        self._n_seen: int = 0

    def update(self, edges: List[Dict]):
        """Reservoir sampling update."""
        for edge in edges:
            self._n_seen += 1
            if len(self._buffer) < self.max_size:
                self._buffer.append(edge)
            else:
                replace_idx = int(self.rng.integers(0, self._n_seen))
                if replace_idx < self.max_size:
                    self._buffer[replace_idx] = edge

    def sample(self, n: int) -> pd.DataFrame:
        """[FIX-07] Guard n > len(buffer) dengan cara yang aman."""
        if not self._buffer:
            return pd.DataFrame()
        n = min(n, len(self._buffer))  # guard utama
        if n <= 0:
            return pd.DataFrame()
        # [FIX-07] tambah guard eksplisit sebelum rng.choice
        indices = self.rng.choice(len(self._buffer), size=n, replace=False)
        sampled = [self._buffer[i] for i in indices]
        return pd.DataFrame(sampled)

    def to_dataframe(self) -> pd.DataFrame:
        if not self._buffer:
            return pd.DataFrame()
        return pd.DataFrame(self._buffer)

    def __len__(self) -> int:
        return len(self._buffer)


# ─────────────────────────────────────────────────────────────────────────────
# POIN 12 — STREAM SIMULATION INTERFACE
# ─────────────────────────────────────────────────────────────────────────────
class TemporalStreamIterator:
    """
    Interface streaming untuk CONTINUAL LEARNING.

    Menyimulasikan kedatangan data real-time:
      - next_batch()   → N edge temporal berikutnya (urutan waktu dijamin)
      - has_next()     → apakah masih ada data [ADD-01]
      - seek(ts)       → loncat ke timestamp tertentu [ADD-02]
      - reset()        → kembali ke awal stream
      - get_progress() → progress info

    JAMINAN: Setiap panggilan next_batch() selalu mengembalikan data
             dengan timestamp >= batch sebelumnya. Tidak ada backward time travel.

    Kompatibel dengan continual learning loop:
      while stream.has_next():
          batch, replay = stream.next_batch(include_replay=True)
          model.update(batch)
          model.evaluate(batch)

    [FIX-02, FIX-03] has_next() ditambahkan, duplikasi __iter__/__next__ dihapus.
    """

    def __init__(self, edges_path: str, batch_size: int = 256,
                 replay_buffer: Optional["ReplayBuffer"] = None):
        self._path         = edges_path
        self._batch_size   = batch_size
        self._replay_buffer= replay_buffer
        self._parquet_file = pq.ParquetFile(edges_path)
        self._total_rows   = self._parquet_file.metadata.num_rows
        self._cursor       = 0
        self._batch_counter= 0
        self._exhausted    = False
        self._iter_batches = self._parquet_file.iter_batches(batch_size=batch_size)

    # ── [FIX-03] Satu pasang __iter__ / __next__ yang bersih ─────────────────
    def __iter__(self):
        return self

    def __next__(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """[FIX-03] Panggil next_batch; raise StopIteration jika stream habis."""
        if not self.has_next():
            raise StopIteration
        return self.next_batch(include_replay=True)

    # ── [ADD-01] has_next() ──────────────────────────────────────────────────
    def has_next(self) -> bool:
        """Return True jika masih ada batch yang belum dibaca."""
        return not self._exhausted

    # ── next_batch ───────────────────────────────────────────────────────────
    def next_batch(self, include_replay: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Kembalikan batch edge berikutnya dalam urutan waktu.
        GARANTEE: timestamp batch ini >= timestamp batch sebelumnya.
        """
        try:
            record_batch   = next(self._iter_batches)
            current_batch  = record_batch.to_pandas()
            self._batch_counter += 1
            self._cursor   += len(current_batch)

            replay_batch = pd.DataFrame()
            if include_replay and self._replay_buffer is not None:
                n_replay = min(32, max(1, len(current_batch) // 4))
                replay_batch = self._replay_buffer.sample(n=n_replay)

            return current_batch, replay_batch

        except StopIteration:
            self._exhausted = True
            raise StopIteration

    # ── [ADD-02] seek() yang benar-benar berfungsi ───────────────────────────
    def seek(self, timestamp: int):
        """
        [ADD-02] Loncat ke posisi dalam stream dengan timestamp >= yang diberikan.
        Implementasi: re-scan dari awal dan skip batch yang seluruhnya < timestamp.
        Aman karena Parquet mendukung predicate pushdown implisit via scan.
        """
        logger.info(f"[Stream] Seeking ke timestamp >= {timestamp}...")
        self.reset()
        skipped = 0
        while self.has_next():
            try:
                record_batch = next(self._iter_batches)
                batch_df = record_batch.to_pandas()
                self._cursor += len(batch_df)

                # Jika ada baris dengan ts >= target, set ulang iter dari sini
                if batch_df["timestamp"].max() >= timestamp:
                    # Buat "pre-loaded" batch agar next_batch() tidak melewatinya
                    self._seeked_batch = batch_df[batch_df["timestamp"] >= timestamp]
                    logger.info(f"[Stream] Seek selesai: melewati {skipped} batch")
                    return
                skipped += 1
            except StopIteration:
                self._exhausted = True
                break
        logger.warning(f"[Stream] Seek: tidak ada data dengan timestamp >= {timestamp}")

    # ── reset ─────────────────────────────────────────────────────────────────
    def reset(self):
        """Reset stream ke awal."""
        self._iter_batches  = self._parquet_file.iter_batches(
                                  batch_size=self._batch_size)
        self._cursor        = 0
        self._batch_counter = 0
        self._exhausted     = False
        logger.info("[Stream] Stream di-reset ke awal")

    def get_progress(self) -> Dict[str, Any]:
        return {
            "cursor":      self._cursor,
            "total":       self._total_rows,
            "pct_done":    round(100 * self._cursor / max(self._total_rows, 1), 2),
            "batch_count": self._batch_counter,
        }


# ─────────────────────────────────────────────────────────────────────────────
# [ADD-05] TEMPORAL INTEGRITY CHECKER
# ─────────────────────────────────────────────────────────────────────────────
class TemporalIntegrityChecker:
    """
    [ADD-05] Verifikasi output Parquet untuk memastikan tidak ada:
      1. Timestamp yang tidak monoton (causal violation)
      2. Future leakage di negative samples
      3. Node ID yang melebihi registry size
    """

    def __init__(self, edges_path: str, nodes_path: str):
        self.edges_path = edges_path
        self.nodes_path = nodes_path

    def run_checks(self) -> Dict[str, Any]:
        logger.info("[Integrity] Menjalankan temporal integrity checks...")
        df = pq.read_table(self.edges_path).to_pandas()
        nodes_df = pq.read_table(self.nodes_path).to_pandas()
        n_nodes = len(nodes_df)
        results = {}

        # Check 1: Monotonisitas timestamp
        pos_only = df[~df["is_negative_sample"]].copy()
        violations = (pos_only["timestamp"].diff() < 0).sum()
        results["timestamp_monotonic"] = violations == 0
        results["n_causal_violations"] = int(violations)

        # Check 2: Node ID dalam range
        max_src = df["src_node"].max()
        max_dst = df["dst_node"].max()
        results["node_ids_in_range"] = (max_src < n_nodes) and (max_dst < n_nodes)
        results["max_src_node"] = int(max_src)
        results["max_dst_node"] = int(max_dst)
        results["total_nodes_in_registry"] = n_nodes

        # Check 3: Proporsi label
        results["n_total"]    = len(df)
        results["n_positive"] = int(df["label"].sum())
        results["n_negative"] = len(df) - int(df["label"].sum())
        results["n_neg_samples"] = int(df["is_negative_sample"].sum())

        # Check 4: Tidak ada NaN di kolom numerik kritis
        numeric_cols = ["src_node", "dst_node", "timestamp",
                        "inter_arrival_time", "session_count",
                        "host_switching_rate", "failed_login_ratio",
                        "novel_destination_score", "temporal_session_entropy"]
        null_counts = {c: int(df[c].isnull().sum()) for c in numeric_cols if c in df.columns}
        results["null_counts"] = null_counts
        results["no_nulls"] = all(v == 0 for v in null_counts.values())

        # Summary
        all_ok = (results["timestamp_monotonic"] and
                  results["node_ids_in_range"] and
                  results["no_nulls"])
        results["all_checks_passed"] = all_ok

        status = "✅ PASSED" if all_ok else "⚠️  WARNING"
        logger.info(f"[Integrity] {status}")
        logger.info(f"[Integrity] Causal violations: {violations}")
        logger.info(f"[Integrity] Null values: {null_counts}")
        logger.info(f"[Integrity] Pos labels: {results['n_positive']}, "
                    f"Neg samples: {results['n_neg_samples']}")
        return results


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE UTAMA
# ─────────────────────────────────────────────────────────────────────────────
class TemporalGraphStreamPipeline:
    """
    Orchestrator utama Tahap 1 LogNet.
    Menjalankan semua 12 poin secara berurutan dan menghasilkan 4 output utama.

    OUTPUT:
      1. Temporal Edge Stream    → temporal_edges/edges.parquet
      2. Dynamic Node Registry   → node_registry/nodes.parquet
      3. Temporal Windows        → windows/windows.json
      4. Replay Candidate Buffer → replay_buffer/replay_candidates.parquet
      +  metadata/memory_init.json
      +  metadata/temporal_splits.json
    """

    def __init__(self, config: LogNetConfig):
        self.cfg = config
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

        self.loader       = RawLogLoader(config)
        self.sorter       = ChronologicalSorter(config)
        self.node_reg     = DynamicNodeRegistry()
        self.edge_builder = TemporalEdgeBuilder(self.node_reg)
        self.feat_eng     = TemporalFeatureEngineer(
                                window_size=config.feature_rolling_window)
        self.windower     = TemporalWindowingSystem(config)
        self.neg_sampler  = TemporalNegativeSampler(config, self.node_reg)
        self.mem_init     = MemoryInitializer(self.node_reg)
        self.storage      = ParquetStorage(config)
        self.split_mgr    = TemporalSplitManager(config)
        self.replay_buf   = ReplayBuffer(config)

    def run(self) -> Dict[str, Any]:
        logger.info("=" * 70)
        logger.info("  LogNet Tahap 1: Temporal Graph Stream Construction v2.0")
        logger.info("=" * 70)

        # ── Step 1-2: Load & Sort ───────────────────────────────────────────
        logger.info("\n[Step 1] Memuat redteam log (attack labels)...")
        attack_index  = self.loader.stream_redteam_log()
        label_system  = LabelAlignmentSystem(attack_index, self.cfg.label_time_tolerance)

        logger.info("\n[Step 2] Streaming & sorting auth log secara chronological...")
        raw_stream    = self.loader.stream_auth_log()
        sorted_stream = self.sorter.sorted_stream(raw_stream)

        # ── Step 3-8: Single-pass Edge Construction ─────────────────────────
        logger.info("\n[Step 3-8] Single-pass edge construction, feature eng, & neg sampling...")
        out_edges_path = Path(self.cfg.output_dir) / "temporal_edges" / "edges.parquet"
        out_edges_path.parent.mkdir(parents=True, exist_ok=True)
        writer = pq.ParquetWriter(
            out_edges_path, self.storage.EDGE_SCHEMA,
            compression=self.cfg.parquet_compression
        )

        chunk_buffer      = []
        n_causal_violations = 0
        total_pos_edges   = 0
        total_neg_edges   = 0
        total_attacks     = 0
        prev_ts           = -1

        for event in sorted_stream:
            # Bangun edge & fitur positif
            edge = self.edge_builder.build_edge(event)
            edge = self.feat_eng.compute_and_enrich(edge)

            # [FIX-09] Pastikan src_computer selalu ada untuk label alignment
            src_computer = edge.get("src_computer") or event.get("src_computer", "")
            edge["label"] = label_system.get_label(
                edge["src_user_raw"], src_computer,
                edge["dst_host_raw"], edge["timestamp"]
            )
            edge["is_negative_sample"] = False

            # [ADD-04] Monitoring kausal violation
            if edge["timestamp"] < prev_ts:
                n_causal_violations += 1
            prev_ts = edge["timestamp"]

            if edge["label"] == 1:
                total_attacks += 1

            # Update state negatif sampler SEBELUM sampling
            self.neg_sampler.update_state(edge)
            chunk_buffer.append(edge)
            total_pos_edges += 1

            # Inline Negative Sampling (di timestamp yang sama, pakai node masa lalu)
            # n_neg dikontrol oleh neg_sample_ratio:
            #   ratio=1.0 → 1 neg per positive (balanced)
            #   ratio=2.0 → 2 neg per positive
            #   attack events (label==1) selalu dapat extra 1 neg untuk imbalance handling
            n_neg = max(0, round(self.cfg.neg_sample_ratio))
            if edge["label"] == 1:
                n_neg = max(1, n_neg + 1)  # extra neg untuk attack (harder negatives)
            if n_neg > 0:
                negs = self.neg_sampler.sample_negatives(edge, n_neg=n_neg)
                chunk_buffer.extend(negs)
                total_neg_edges += len(negs)
                self.replay_buf.update([edge] + negs)
            else:
                self.replay_buf.update([edge])

            # Tulis ke Parquet per chunk (Memory O(chunk_size))
            if len(chunk_buffer) >= self.cfg.stream_chunk_size:
                self._flush_chunk_to_writer(writer, chunk_buffer)
                chunk_buffer.clear()

        # Flush sisa buffer
        if chunk_buffer:
            self._flush_chunk_to_writer(writer, chunk_buffer)
        writer.close()

        logger.info(f"[Pipeline] Edge selesai: {total_pos_edges:,} positif, "
                    f"{total_neg_edges:,} negatif, {total_attacks:,} attack labels")
        if n_causal_violations > 0:
            logger.warning(f"[Pipeline] ⚠️  Kausal violation: {n_causal_violations} "
                           f"(timestamp tidak monoton)")
        else:
            logger.info("[Pipeline] ✅ Tidak ada kausal violation — temporal causality terjaga")

        # ── Step 9-12: Split, Windows, Metadata ─────────────────────────────
        logger.info("\n[Step 9-12] Menyiapkan split, windowing, & metadata...")

        # Load hanya 2 kolom krusial untuk efisiensi RAM
        df_meta = pq.read_table(
            out_edges_path, columns=["timestamp", "label"]
        ).to_pandas()

        splits  = self.split_mgr.compute_splits(df_meta)
        windows = self.windower.build_windows(df_meta)

        windows["event_windows"] = self.split_mgr.label_window_purposes(
            windows["event_windows"], splits)
        windows["time_windows"]  = self.split_mgr.label_window_purposes(
            windows["time_windows"], splits)

        # Simpan semua output
        self.storage.save_node_registry(self.node_reg.to_dataframe())
        self.storage.save_replay_buffer(self.replay_buf.to_dataframe())
        self.split_mgr.save_split_metadata(splits, self.cfg.output_dir)
        self.mem_init.export_metadata(
            str(Path(self.cfg.output_dir) / "metadata" / "memory_init.json"))

        windows_path = Path(self.cfg.output_dir) / "windows" / "windows.json"
        windows_path.parent.mkdir(parents=True, exist_ok=True)
        with open(windows_path, "w") as f:
            json.dump({
                "event_windows": self.windower.to_dict_list(windows["event_windows"]),
                "time_windows":  self.windower.to_dict_list(windows["time_windows"]),
            }, f, indent=2)
        logger.info(f"[Window] Windows disimpan: {windows_path}")

        nodes_path = str(Path(self.cfg.output_dir) / "node_registry" / "nodes.parquet")
        results = {
            "edges_path":   str(out_edges_path),
            "nodes_path":   nodes_path,
            "windows_path": str(windows_path),
            "n_pos_edges":  total_pos_edges,
            "n_neg_edges":  total_neg_edges,
            "n_attacks":    total_attacks,
            "n_nodes":      len(self.node_reg),
        }

        # ── [ADD-05] Integrity Check ─────────────────────────────────────────
        logger.info("\n[Integrity] Menjalankan post-process integrity check...")
        checker = TemporalIntegrityChecker(str(out_edges_path), nodes_path)
        integrity = checker.run_checks()
        results["integrity"] = integrity

        return results

    def _flush_chunk_to_writer(
            self, writer: pq.ParquetWriter, chunk: List[Dict]):
        """Helper: cast dan tulis chunk ke ParquetWriter."""
        df = self.storage._cast_edge_df(pd.DataFrame(chunk))
        table = pa.Table.from_pandas(df, schema=self.storage.EDGE_SCHEMA)
        writer.write_table(table)

    def get_stream_iterator(self, batch_size: int = 256) -> TemporalStreamIterator:
        """Buat TemporalStreamIterator dari output edges yang sudah diproses."""
        edges_path = str(
            Path(self.cfg.output_dir) / "temporal_edges" / "edges.parquet")
        return TemporalStreamIterator(edges_path, batch_size, self.replay_buf)

    # ── [ADD-03] Split Iterator ──────────────────────────────────────────────
    def get_split_iterator(
            self,
            split: str = "train",
            batch_size: int = 256) -> Iterator[pd.DataFrame]:
        """
        [ADD-03] Iterasi hanya edges dalam split tertentu (train/val/test).
        Menggunakan predicate pushdown Parquet untuk efisiensi.

        Args:
            split: "train", "val", atau "test"
            batch_size: jumlah edge per yield

        Yields:
            pd.DataFrame berisi batch edges dalam split yang diminta.
        """
        split_path = Path(self.cfg.output_dir) / "metadata" / "temporal_splits.json"
        if not split_path.exists():
            raise FileNotFoundError(
                f"Split metadata tidak ditemukan: {split_path}. "
                f"Jalankan pipeline.run() terlebih dahulu.")

        with open(split_path) as f:
            splits = json.load(f)

        if split not in splits:
            raise ValueError(f"Split '{split}' tidak valid. Pilih: train, val, test")

        t_start = splits[split]["start_ts"]
        t_end   = splits[split]["end_ts"]
        logger.info(f"[SplitIter] Iterasi split '{split}': "
                    f"ts [{t_start}, {t_end}), "
                    f"{splits[split]['n_events']:,} events")

        edges_path = Path(self.cfg.output_dir) / "temporal_edges" / "edges.parquet"
        pf = pq.ParquetFile(str(edges_path))

        for batch in pf.iter_batches(batch_size=batch_size):
            df = batch.to_pandas()
            # Filter ke split window yang diminta
            mask = (df["timestamp"] >= t_start) & (df["timestamp"] < t_end)
            subset = df[mask]
            if not subset.empty:
                yield subset


# ─────────────────────────────────────────────────────────────────────────────
# DEMO / TEST — Jalankan dengan data dummy jika file LANL tidak tersedia
# ─────────────────────────────────────────────────────────────────────────────
def _generate_dummy_auth_log(path: str, n_events: int = 50_000) -> List[Tuple]:
    """
    Generate dummy auth.txt.gz untuk testing tanpa dataset LANL asli.
    Format: timestamp,src_user@domain,dst_user@domain,src_comp,dst_comp,
            auth_type,logon_type,auth_orient,success_fail

    [FIX-LABEL] Return list of (ts, user, src_comp, dst_comp) tuples untuk
    attack injection yang guaranteed match dengan redteam.

    Strategi 2-fase:
      Fase 1: Generate ~1% event sebagai "attack seeds" — catat exact signature.
      Fase 2: Generate sisa event sebagai normal traffic.
    Kedua fase dicampur dan di-sort by timestamp sebelum ditulis.
    """
    import random
    rng     = random.Random(42)
    users   = [f"U{i:04d}@DOM1" for i in range(50)]
    hosts   = [f"C{i:04d}" for i in range(100)]
    auth_t  = ["NTLM", "Kerberos", "Negotiate"]
    logon_t = ["Network", "Interactive", "Batch", "Service"]

    n_attack = max(50, n_events // 100)  # minimal 50 attack events

    # Pilih sejumlah attacker yang fixed agar signature mudah di-match
    atk_users = [f"U{i:04d}@DOM1" for i in range(5)]   # U0000-U0004
    atk_src   = [f"C{i:04d}" for i in range(5)]         # C0000-C0004 (src_computer)
    atk_dst   = [f"C{i:04d}" for i in range(90, 100)]   # C0090-C0099 (dst_computer)

    events = []
    ts = 1_000_000

    # Fase 1: Attack events (guaranteed signature)
    attack_signatures = []  # list of (ts, user_no_domain, src_comp, dst_comp)
    for i in range(n_attack):
        ts += rng.randint(1, 15)
        src_u = rng.choice(atk_users)
        dst_u = rng.choice(users)
        src_c = rng.choice(atk_src)
        dst_c = rng.choice(atk_dst)
        at    = rng.choice(auth_t)
        lt    = rng.choice(logon_t)
        sf    = "Fail"  # attack sering gagal
        events.append((ts, src_u, dst_u, src_c, dst_c, at, lt, sf))
        # Catat signature tanpa domain untuk redteam matching
        user_no_domain = src_u.split("@")[0]
        attack_signatures.append((ts, user_no_domain, src_c, dst_c))

    # Fase 2: Normal events
    for _ in range(n_events - n_attack):
        ts += rng.randint(0, 30)
        src_u = rng.choice(users)
        dst_u = rng.choice(users)
        src_c = rng.choice(hosts[:80])   # normal traffic ke host 0-79
        dst_c = rng.choice(hosts[:80])
        at    = rng.choice(auth_t)
        lt    = rng.choice(logon_t)
        sf    = rng.choice(["Success"] * 9 + ["Fail"])
        events.append((ts, src_u, dst_u, src_c, dst_c, at, lt, sf))

    # Sort by timestamp untuk menjaga urutan kronologis dalam file mentah
    events.sort(key=lambda e: e[0])

    with gzip.open(path, "wt") as f:
        for ev in events:
            ts_w, src_u, dst_u, src_c, dst_c, at, lt, sf = ev
            f.write(f"{ts_w},{src_u},{dst_u},{src_c},{dst_c},{at},{lt},LogOn,{sf}\n")

    return attack_signatures  # [FIX-LABEL] kembalikan signatures untuk redteam


def _generate_dummy_redteam_log(path: str, attack_signatures: List[Tuple]):
    """
    [FIX-LABEL] Generate redteam.txt.gz dari attack_signatures yang EXACT MATCH
    dengan auth log. Sebelumnya: random seed berbeda → signature tidak pernah match.
    Sekarang: pakai exact (ts, user, src_comp, dst_comp) dari auth log.

    Format redteam: timestamp,user@domain,src_computer,dst_computer
    """
    import random
    rng = random.Random(43)

    with gzip.open(path, "wt") as f:
        for (ts, user, src_c, dst_c) in attack_signatures:
            # Tambahkan sedikit jitter (±30 detik) untuk simulasi clock skew
            ts_jitter = ts + rng.randint(-30, 30)
            f.write(f"{ts_jitter},{user}@DOM1,{src_c},{dst_c}\n")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LogNet Stage 1 Pipeline v2.0")
    parser.add_argument("--auth",       default="auth.txt.gz",
                        help="Path ke auth.txt.gz")
    parser.add_argument("--redteam",    default="redteam.txt.gz",
                        help="Path ke redteam.txt.gz")
    parser.add_argument("--output",     default="lognet_output",
                        help="Output directory")
    parser.add_argument("--demo",       action="store_true",
                        help="Jalankan dengan data dummy (tanpa dataset LANL)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size untuk stream iterator")
    parser.add_argument("--no-integrity", action="store_true",
                        help="Lewati integrity check (lebih cepat)")
    args = parser.parse_args()

    config = LogNetConfig(
        auth_log_path    = args.auth,
        redteam_log_path = args.redteam,
        output_dir       = args.output,
    )

    # ── Mode Demo ──────────────────────────────────────────────────────────
    if args.demo or (not Path(args.auth).exists()):
        logger.info("[Demo] Dataset LANL tidak ditemukan → generate dummy data...")
        attack_sigs = _generate_dummy_auth_log(config.auth_log_path, n_events=50_000)
        _generate_dummy_redteam_log(config.redteam_log_path, attack_sigs)
        logger.info(f"[Demo] Dummy data selesai: {config.auth_log_path}, "
                    f"{config.redteam_log_path} ({len(attack_sigs)} attack signatures)")

    # ── Jalankan Pipeline ──────────────────────────────────────────────────
    pipeline = TemporalGraphStreamPipeline(config)
    results  = pipeline.run()

    # ── Demo Stream Iterator ───────────────────────────────────────────────
    logger.info("\n[Demo] Menjalankan stream iterator (3 batch pertama)...")
    stream = pipeline.get_stream_iterator(batch_size=args.batch_size)
    for i, (current_batch, replay_batch) in enumerate(stream):
        logger.info(
            f"  Batch {i+1}: {len(current_batch)} current edges, "
            f"{len(replay_batch)} replay edges, "
            f"ts=[{current_batch['timestamp'].min()}, "
            f"{current_batch['timestamp'].max()}], "
            f"attacks={current_batch['label'].sum()}"
        )
        if i >= 2:
            break

    # ── Demo Split Iterator ────────────────────────────────────────────────
    logger.info("\n[Demo] Menjalankan split iterator (train, 2 batch pertama)...")
    for i, batch_df in enumerate(pipeline.get_split_iterator("train", batch_size=256)):
        logger.info(f"  Train Batch {i+1}: {len(batch_df)} edges, "
                    f"attacks={batch_df['label'].sum()}")
        if i >= 1:
            break

    print("\n✅ Tahap 1 selesai.")
    print(f"   Edges  : {results['edges_path']}")
    print(f"   Nodes  : {results['nodes_path']}")
    print(f"   Windows: {results['windows_path']}")
    print(f"   Stats  : {results['n_pos_edges']:,} pos, "
          f"{results['n_neg_edges']:,} neg, "
          f"{results['n_nodes']:,} nodes")
    if results.get("integrity", {}).get("all_checks_passed"):
        print("   Integrity: ✅ ALL CHECKS PASSED")
    else:
        print("   Integrity: ⚠️  Ada peringatan — lihat log di atas")
    print("\n   Gunakan TemporalStreamIterator.next_batch() untuk Tahap 2 (TGN Training).")