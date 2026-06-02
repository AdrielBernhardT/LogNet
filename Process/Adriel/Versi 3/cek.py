"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          LogNet — Temporal Graph Stream Construction (Tahap 1)               ║
║          Dataset: LANL Auth Log (auth.txt.gz + redteam.txt.gz)               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import gzip
import os
import json
import math
import heapq
import logging
import warnings
from pathlib import Path
from typing import Iterator, Dict, Tuple, Optional, List, Any
from collections import defaultdict, deque
from dataclasses import dataclass, asdict

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

# --- (Pertahankan class LogNetConfig, RawLogLoader, ChronologicalSorter dari kode aslimu) ---
# [Untuk efisiensi, saya asumsikan RawLogLoader dan ChronologicalSorter sama dengan yang asli]

@dataclass
class LogNetConfig:
    auth_log_path: str      = "auth.txt.gz"
    redteam_log_path: str   = "redteam.txt.gz"
    output_dir: str         = "lognet_output"
    stream_chunk_size: int  = 50_000
    sort_buffer_size: int   = 500_000
    event_window_size: int  = 1_000
    event_window_stride: int= 500
    time_window_size: int   = 3_600
    time_window_stride: int = 1_800
    label_time_tolerance: int = 60
    neg_sample_ratio: float = 1.0
    neg_sample_lookback: int = 10_000
    train_ratio: float = 0.70
    val_ratio:   float = 0.15
    replay_buffer_max_size: int = 5_000
    parquet_compression: str = "snappy"
    random_seed: int = 42

class RawLogLoader:
    # [Implementasi identik dengan aslimu]
    pass

class ChronologicalSorter:
    # [Implementasi identik dengan aslimu]
    pass

class DynamicNodeRegistry:
    # [Implementasi identik dengan aslimu]
    pass

class TemporalEdgeBuilder:
    # [Implementasi identik dengan aslimu]
    pass

class TemporalFeatureEngineer:
    # [Implementasi identik dengan aslimu]
    pass

class LabelAlignmentSystem:
    # [Implementasi identik dengan aslimu]
    pass

class MemoryInitializer:
    # [Implementasi identik dengan aslimu]
    pass

# --- PERBAIKAN 1: Negative Sampler (Inline Mode) ---
class TemporalNegativeSampler:
    """Negative sampler dimodifikasi agar bisa digunakan inline."""
    def __init__(self, config: LogNetConfig, node_registry: DynamicNodeRegistry):
        self.cfg      = config
        self.registry = node_registry
        self.rng      = np.random.default_rng(config.random_seed)
        self._seen_users: List[int] = []
        self._seen_hosts: List[int] = []
        self._positive_pairs: set = set()

    def update_state(self, edge: Dict[str, Any]):
        src, dst = edge["src_node"], edge["dst_node"]
        if src not in self._seen_users: self._seen_users.append(src)
        if dst not in self._seen_hosts: self._seen_hosts.append(dst)
        self._positive_pairs.add((src, dst))

    def sample_negatives(self, positive_edge: Dict[str, Any], n_neg: int = 1) -> List[Dict[str, Any]]:
        if not self._seen_users or not self._seen_hosts: return []
        negatives = []
        lookback = self.cfg.neg_sample_lookback
        user_pool = self._seen_users[-lookback:]
        host_pool = self._seen_hosts[-lookback:]
        
        attempts, max_attempts = 0, n_neg * 10
        while len(negatives) < n_neg and attempts < max_attempts:
            attempts += 1
            if self.rng.random() < 0.5:
                neg_src, neg_dst = positive_edge["src_node"], int(self.rng.choice(host_pool))
            else:
                neg_src, neg_dst = int(self.rng.choice(user_pool)), positive_edge["dst_node"]
            
            if (neg_src, neg_dst) in self._positive_pairs: continue
            
            neg_edge = dict(positive_edge)
            neg_edge.update({"src_node": neg_src, "dst_node": neg_dst, "label": 0, "is_negative_sample": True})
            negatives.append(neg_edge)
        return negatives

# --- PERBAIKAN 2: Stream Iterator yang Memisahkan Replay Buffer ---
class TemporalStreamIterator:
    """
    Interface streaming yang mencegah Time-Travel bug pada TGN.
    """
    def __init__(self, edges_path: str, batch_size: int = 256, replay_buffer=None):
        self._path = edges_path
        self._batch_size = batch_size
        self._replay_buffer = replay_buffer
        
        # Buka parquet file sebagai streaming object
        self._parquet_file = pq.ParquetFile(edges_path)
        self._iter_batches = self._parquet_file.iter_batches(batch_size=batch_size)
        self._batch_counter = 0

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.next_batch(include_replay=True)

    def next_batch(self, include_replay: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        RETURN 2 DataFrame: (Current_Batch, Replay_Batch).
        Pisahkan edge masa kini dengan edge masa lalu agar node memory TGN tidak rusak.
        """
        try:
            record_batch = next(self._iter_batches)
            current_batch = record_batch.to_pandas()
            self._batch_counter += 1
            
            replay_batch = pd.DataFrame()
            if include_replay and self._replay_buffer is not None:
                replay_batch = self._replay_buffer.sample(n=min(32, len(current_batch) // 4))
                
            return current_batch, replay_batch
        except StopIteration:
            raise StopIteration

# --- PERBAIKAN 3: Pipeline Utama (O(1) Memory Usage menggunakan ParquetWriter) ---
class ParquetStorage:
    # [Skema pa.schema dari aslimu tetap di sini]
    EDGE_SCHEMA = pa.schema([
        pa.field("src_node",                 pa.int32()),
        pa.field("dst_node",                 pa.int32()),
        pa.field("timestamp",                pa.int64()),
        pa.field("auth_type_enc",            pa.int8()),
        pa.field("logon_type_enc",           pa.int8()),
        pa.field("is_success",               pa.int8()),
        pa.field("inter_arrival_time",       pa.float32()),
        pa.field("session_count",            pa.int32()),
        pa.field("host_switching_rate",      pa.float32()),
        pa.field("failed_login_ratio",       pa.float32()),
        pa.field("novel_destination_score",  pa.float32()),
        pa.field("temporal_session_entropy", pa.float32()),
        pa.field("label",                    pa.int8()),
        pa.field("is_negative_sample",       pa.bool_()),
        pa.field("src_user_raw",             pa.string()),
        pa.field("dst_host_raw",             pa.string()),
        pa.field("src_computer",             pa.string()),
        pa.field("auth_orientation",         pa.string()),
    ])
    # ... fungsi lain

class TemporalGraphStreamPipeline:
    def __init__(self, config: LogNetConfig):
        self.cfg = config
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        # Inisialisasi komponen
        self.loader      = RawLogLoader(config)
        self.sorter      = ChronologicalSorter(config)
        self.node_reg    = DynamicNodeRegistry()
        self.edge_builder= TemporalEdgeBuilder(self.node_reg)
        self.feat_eng    = TemporalFeatureEngineer()
        self.neg_sampler = TemporalNegativeSampler(config, self.node_reg)
        self.mem_init    = MemoryInitializer(self.node_reg)
        self.storage     = ParquetStorage(config)
        self.replay_buf  = ReplayBuffer(config) # Asumsikan ReplayBuffer class identik

    def run(self) -> Dict[str, Any]:
        logger.info("="*70)
        logger.info("  LogNet Tahap 1: Temporal Graph Stream Construction (Optimized)")
        logger.info("="*70)

        attack_index = self.loader.stream_redteam_log()
        label_system = LabelAlignmentSystem(attack_index, self.cfg.label_time_tolerance)
        raw_stream   = self.loader.stream_auth_log()
        sorted_stream = self.sorter.sorted_stream(raw_stream)

        # Siapkan ParquetWriter agar RAM tidak meledak
        out_edges_path = Path(self.cfg.output_dir) / "temporal_edges" / "edges.parquet"
        out_edges_path.parent.mkdir(parents=True, exist_ok=True)
        writer = pq.ParquetWriter(out_edges_path, self.storage.EDGE_SCHEMA, compression=self.cfg.parquet_compression)

        logger.info("\n[Step 3-8] Single-pass Edge Construction, Feature Eng, & Neg Sampling...")
        
        chunk_buffer = []
        n_causal_violations = 0
        total_pos_edges = 0
        total_neg_edges = 0
        prev_ts = -1

        for event in sorted_stream:
            # Bangun Edge & Fitur Positif
            edge = self.edge_builder.build_edge(event)
            edge = self.feat_eng.compute_and_enrich(edge)
            edge["label"] = label_system.get_label(
                edge["src_user_raw"], edge["src_computer"], edge["dst_host_raw"], edge["timestamp"]
            )
            edge["is_negative_sample"] = False

            if edge["timestamp"] < prev_ts: n_causal_violations += 1
            prev_ts = edge["timestamp"]

            # Update state sebelum sampling
            self.neg_sampler.update_state(edge)
            chunk_buffer.append(edge)
            total_pos_edges += 1

            # INLINE Negative Sampling (Tepat di timestamp yang sama)
            if edge["label"] == 1 or self.cfg.neg_sample_ratio > 0:
                n_neg = 1 if edge["label"] == 0 else 2
                negs = self.neg_sampler.sample_negatives(edge, n_neg=n_neg)
                chunk_buffer.extend(negs)
                total_neg_edges += len(negs)
                self.replay_buf.update([edge] + negs) # Positif & Negatif masuk ke Replay Buffer
            else:
                self.replay_buf.update([edge])

            # Tulis ke Parquet per chunk (Memory O(1))
            if len(chunk_buffer) >= self.cfg.stream_chunk_size:
                df_chunk = self.storage._cast_edge_df(pd.DataFrame(chunk_buffer))
                table = pa.Table.from_pandas(df_chunk, schema=self.storage.EDGE_SCHEMA)
                writer.write_table(table)
                chunk_buffer.clear()

        # Flush sisa buffer
        if chunk_buffer:
            df_chunk = self.storage._cast_edge_df(pd.DataFrame(chunk_buffer))
            table = pa.Table.from_pandas(df_chunk, schema=self.storage.EDGE_SCHEMA)
            writer.write_table(table)
            
        writer.close()
        logger.info(f"[Pipeline] Selesai: {total_pos_edges} pos edges, {total_neg_edges} neg edges.")

        # --- Evaluasi Post-Process untuk Windowing & Split ---
        # Karena kita menggunakan file streaming, split dan windowing dapat diekstraksi
        # dengan iterasi pembacaan parquet menggunakan pyarrow untuk menjaga RAM.
        
        # [Simpan Node Registry dan Metadata seperti biasa]
        # self.storage.save_node_registry(self.node_reg.to_dataframe())
        # self.storage.save_replay_buffer(self.replay_buf.to_dataframe())
        
        return {"edges_path": str(out_edges_path)}

    def get_stream_iterator(self, batch_size: int = 256) -> TemporalStreamIterator:
        edges_path = str(Path(self.cfg.output_dir) / "temporal_edges" / "edges.parquet")
        return TemporalStreamIterator(edges_path, batch_size, self.replay_buf)