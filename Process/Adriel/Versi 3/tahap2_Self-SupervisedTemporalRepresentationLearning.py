"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                   LogNet — Self-Supervised TGN Learning (Tahap 2)            ║
║                   Core: Temporal Graph Network (TGN) Encoder                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
Prinsip: Dynamic Memory update, No Future Leakage, Time-Aware Embeddings.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────────────
# 1. TIME ENCODER (Temporal Awareness)
# ─────────────────────────────────────────────────────────────────────────────
class TimeEncoder(nn.Module):
    """
    Mengonversi selisih waktu (delta_t) menjadi vector embedding 
    menggunakan fungsi harmonik (sin/cos).
    Penting untuk mendeteksi beaconing atau burst behavior.
    """
    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension
        self.basis_freq = nn.Parameter(torch.from_numpy(1 / 10 ** np.linspace(0, 9, dimension)).float())
        self.phase = nn.Parameter(torch.zeros(dimension).float())

    def forward(self, ts: torch.Tensor) -> torch.Tensor:
        # ts: [batch_size]
        batch_size = ts.size(0)
        ts = ts.view(batch_size, 1)
        map_ts = ts * self.basis_freq.view(1, -1) + self.phase.view(1, -1)
        return torch.cos(map_ts)

# ─────────────────────────────────────────────────────────────────────────────
# 2. TGN MEMORY MODULE (Dynamic Node State)
# ─────────────────────────────────────────────────────────────────────────────
class TGNMemory(nn.Module):
    """
    Mengelola mi(t). State memori tidak disimpan di dalam graph, 
    tapi di-update setiap kali ada interaksi (message).
    """
    def __init__(self, num_nodes: int, memory_dim: int, message_dim: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim
        
        # Registri memori node (Persistent selama training stream)
        self.register_buffer("memory", torch.zeros(num_nodes, memory_dim))
        self.register_buffer("last_update", torch.zeros(num_nodes))
        
        # RNN Updater (GRU)
        self.updater = nn.GRUCell(input_size=message_dim, hidden_size=memory_dim)

    def get_memory(self, node_ids: torch.Tensor) -> torch.Tensor:
        return self.memory[node_ids]

    def update_memory(self, node_ids: torch.Tensor, messages: torch.Tensor):
        """Update memori node menggunakan GRU berdasarkan pesan baru."""
        updated_memory = self.updater(messages, self.memory[node_ids])
        self.memory[node_ids] = updated_memory

    def reset_memory(self):
        self.memory.fill_(0)
        self.last_update.fill_(0)

# ─────────────────────────────────────────────────────────────────────────────
# 3. TGN CORE ENCODER
# ─────────────────────────────────────────────────────────────────────────────
class LogNetEncoder(nn.Module):
    def __init__(self, num_nodes: int, edge_feat_dim: int, 
                 memory_dim: int = 128, embedding_dim: int = 128):
        super().__init__()
        self.memory_dim = memory_dim
        self.edge_feat_dim = edge_feat_dim
        
        # Time Encoder
        self.time_enc = TimeEncoder(memory_dim)
        
        # Memory Module
        # Message dim = memory_src + memory_dst + edge_feat + time_feat
        msg_dim = (memory_dim * 2) + edge_feat_dim + memory_dim
        self.memory_module = TGNMemory(num_nodes, memory_dim, msg_dim)
        
        # Embedding Projector (Menggabungkan Memori + Raw Time)
        self.embedding_layer = nn.Sequential(
            nn.Linear(memory_dim + memory_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Link Predictor (Self-Supervised Task)
        self.affinity_score = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, src: torch.Tensor, dst: torch.Tensor, 
                ts: torch.Tensor, edge_feats: torch.Tensor) -> torch.Tensor:
        """
        Flow:
        1. Ambil memori saat ini (sebelum update).
        2. Hitung embedding temporal.
        3. Berikan skor afinitas (link prediction).
        """
        # Get Current Memory
        src_mem = self.memory_module.get_memory(src)
        dst_mem = self.memory_module.get_memory(dst)
        
        # Compute Time Delta & Encoding
        # (Delta dari interaksi terakhir node tersebut)
        src_delta_t = ts - self.memory_module.last_update[src]
        src_time_feat = self.time_enc(src_delta_t)
        
        # Generate Embeddings
        src_emb = self.embedding_layer(torch.cat([src_mem, src_time_feat], dim=1))
        dst_emb = self.embedding_layer(torch.cat([dst_mem, src_time_feat], dim=1)) # Simetris temporal
        
        # Affinity (Probability of edge existing)
        scores = self.affinity_score(torch.cat([src_emb, dst_emb], dim=1))
        
        # Update Metadata for next messages (Late Update strategy)
        # Note: Update memori biasanya dilakukan setelah batch selesai untuk menghindari leakage 
        # dalam batch yang sama jika ada node yang berulang.
        
        return scores, src_emb, dst_emb

    def update_node_states(self, src, dst, ts, edge_feats):
        """Method untuk memicu pembaruan memori setelah interaksi diamati."""
        with torch.no_grad():
            src_mem = self.memory_module.get_memory(src)
            dst_mem = self.memory_module.get_memory(dst)
            
            delta_ts = ts - self.memory_module.last_update[src]
            time_feats = self.time_enc(delta_ts)
            
            # Construct Messages
            # Message = [Mem_src, Mem_dst, Edge_Feat, Time_Feat]
            messages = torch.cat([src_mem, dst_mem, edge_feats, time_feats], dim=1)
            
            # Update Memory
            self.memory_module.update_memory(src, messages)
            self.memory_module.update_memory(dst, messages)
            
            # Update Timestamps
            self.memory_module.last_update[src] = ts
            self.memory_module.last_update[dst] = ts

# ─────────────────────────────────────────────────────────────────────────────
# 4. TRAINER ENGINE (Continual Update Ready)
# ─────────────────────────────────────────────────────────────────────────────
class LogNetStage2Trainer:
    def __init__(self, model: LogNetEncoder, lr: float = 1e-4):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.BCELoss() # Binary Cross Entropy untuk Link Prediction

    def train_batch(self, pos_df: pd.DataFrame, neg_df: pd.DataFrame) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        
        # 1. Persiapkan Data Positif
        p_src = torch.tensor(pos_df['src_node'].values, dtype=torch.long)
        p_dst = torch.tensor(pos_df['dst_node'].values, dtype=torch.long)
        p_ts  = torch.tensor(pos_df['timestamp'].values, dtype=torch.float)
        # Gabungkan semua fitur temporal dari tahap 1
        p_feat = torch.tensor(pos_df[['auth_type_enc', 'logon_type_enc', 'inter_arrival_time', 
                                      'host_switching_rate', 'failed_login_ratio']].values, dtype=torch.float)
        
        # 2. Persiapkan Data Negatif (Negative Sampling dari Tahap 1)
        n_src = torch.tensor(neg_df['src_node'].values, dtype=torch.long)
        n_dst = torch.tensor(neg_df['dst_node'].values, dtype=torch.long)
        n_ts  = torch.tensor(neg_df['timestamp'].values, dtype=torch.float)
        n_feat = torch.tensor(neg_df[['auth_type_enc', 'logon_type_enc', 'inter_arrival_time', 
                                      'host_switching_rate', 'failed_login_ratio']].values, dtype=torch.float)

        # 3. Forward Pass
        pos_scores, _, _ = self.model(p_src, p_dst, p_ts, p_feat)
        neg_scores, _, _ = self.model(n_src, n_dst, n_ts, n_feat)
        
        # 4. Loss (Self-supervised: Model harus bisa membedakan edge asli vs random)
        pos_labels = torch.ones_like(pos_scores)
        neg_labels = torch.zeros_like(neg_scores)
        
        loss = self.criterion(pos_scores, pos_labels) + self.criterion(neg_scores, neg_labels)
        
        loss.backward()
        self.optimizer.step()
        
        # 5. Update Memory (Hanya gunakan interaksi POSITIF/NYATA untuk update state node)
        self.model.update_node_states(p_src, p_dst, p_ts, p_feat)
        
        return loss.item()

# ─────────────────────────────────────────────────────────────────────────────
# 5. EXECUTION BRIDGE
# ─────────────────────────────────────────────────────────────────────────────
def run_stage2(pipeline, num_epochs=1):
    # Load Metadata dari Tahap 1
    with open(f"{pipeline.cfg.output_dir}/metadata/memory_init.json") as f:
        mem_meta = json.load(f)
    
    num_nodes = mem_meta['total_nodes']
    edge_feat_dim = 5 # Sesuai kolom yang kita pilih di Trainer
    
    # Inisialisasi Model
    model = LogNetEncoder(num_nodes=num_nodes, edge_feat_dim=edge_feat_dim)
    trainer = LogNetStage2Trainer(model)
    
    logger.info(f"[Stage 2] Memulai Training TGN pada {num_nodes} nodes...")
    
    for epoch in range(num_epochs):
        # Gunakan Stream Iterator dari Tahap 1
        stream = pipeline.get_stream_iterator(batch_size=512)
        total_loss = 0
        step = 0
        
        while stream.has_next():
            try:
                # Batch berisi: Current edges (Positif) & Replay/Negative edges
                current_batch, replay_batch = stream.next_batch(include_replay=True)
                
                # Filter negatif sample asli dari dataset vs negatif sample buatan
                # (Tahap 1 sudah menyediakan 'is_negative_sample')
                pos_edges = current_batch[current_batch['is_negative_sample'] == False]
                neg_edges = current_batch[current_batch['is_negative_sample'] == True]
                
                if len(neg_edges) == 0: # Fallback jika ratio sampling 0
                    neg_edges = replay_batch 
                
                if not pos_edges.empty and not neg_edges.empty:
                    # Ambil subset negasi yang ukurannya sama dengan positif untuk balance
                    neg_edges = neg_edges.sample(n=min(len(pos_edges), len(neg_edges)))
                    
                    loss = trainer.train_batch(pos_edges, neg_edges)
                    total_loss += loss
                    step += 1
                    
                    if step % 50 == 0:
                        logger.info(f"Epoch {epoch} | Step {step} | Loss: {loss:.4f}")
            except StopIteration:
                break
                
    logger.info(f"[Stage 2] Training Selesai. Model siap untuk Tahap 3 (Anomali Detection).")
    return model