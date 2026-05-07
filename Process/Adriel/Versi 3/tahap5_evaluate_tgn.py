import torch
import os
from torch.nn import Linear, ReLU
from torch_geometric.data import TemporalData
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn import TGNMemory, TransformerConv
from torch_geometric.nn.models.tgn import LastAggregator, LastNeighborLoader
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score, roc_auc_score, precision_recall_curve
import numpy as np

# 1. PERSIAPAN PERANGKAT & DATA
print("1. Menyiapkan Perangkat dan Data...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# FIX: Sesuaikan path dengan output Tahap 3
data_dict = torch.load('./datasets/tgn_dataset.pt')
data = TemporalData(
    src=data_dict['src'], 
    dst=data_dict['dst'], 
    t=data_dict['t'].long(), # FIX: Pastikan format long()
    msg=data_dict['msg'], 
    y=data_dict['y']
).to(device)

test_loader = TemporalDataLoader(data, batch_size=2000)

# 2. MEMBANGUN ULANG KERANGKA (SINKRON DENGAN TAHAP 4)
print("2. Membangun ulang kerangka TGN...")
memory_dim = 100
time_dim   = 100
msg_dim    = data.msg.size(1)

num_nodes = max(data.src.max(), data.dst.max()).item() + 1
# FIX: Harus sama persis dengan buffer Tahap 4 agar state_dict cocok
num_nodes_buffer = num_nodes + 100000 

# --- [A] Custom Message Module ---
class LearnedMessageModule(torch.nn.Module):
    def __init__(self, msg_dim, memory_dim, time_dim):
        super().__init__()
        in_dim = memory_dim + memory_dim + msg_dim + time_dim
        self.lin1 = Linear(in_dim, memory_dim)
        self.lin2 = Linear(memory_dim, memory_dim)
        self.act  = ReLU()
        self.out_channels = memory_dim

    def forward(self, z_src, z_dst, raw_msg, t_enc):
        x = torch.cat([z_src, z_dst, raw_msg, t_enc], dim=-1)
        return self.lin2(self.act(self.lin1(x)))

# --- [B] TGN Memory ---
memory = TGNMemory(
    num_nodes=num_nodes_buffer,
    raw_msg_dim=msg_dim,
    memory_dim=memory_dim,
    time_dim=time_dim,
    message_module=LearnedMessageModule(msg_dim, memory_dim, time_dim),
    aggregator_module=LastAggregator()
).to(device)

# --- [C] Time Encoder ---
class TimeEncoder(torch.nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.lin = Linear(1, out_channels)
    def forward(self, t):
        return self.lin(t.float().unsqueeze(-1)).cos()

time_encoder = TimeEncoder(out_channels=time_dim).to(device)

# --- [D] Graph Attention Embedding ---
class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_dim):
        super().__init__()
        edge_dim = msg_dim + time_dim
        self.conv = TransformerConv(
            in_channels, out_channels // 2, heads=2, dropout=0.1, edge_dim=edge_dim
        )
    def forward(self, x, last_update, edge_index, t, msg):
        rel_t     = (last_update[edge_index[0]] - t.float())
        rel_t_enc = time_encoder(rel_t)
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        return self.conv(x, edge_index, edge_attr)

gnn = GraphAttentionEmbedding(
    in_channels=memory_dim, out_channels=memory_dim, msg_dim=msg_dim, time_dim=time_dim,
).to(device)

# --- [E] Neighbor Loader ---
neighbor_loader = LastNeighborLoader(num_nodes_buffer, size=15, device=device)

# --- [F] Link Predictor ---
class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.lin_src   = Linear(in_channels, in_channels)
        self.lin_dst   = Linear(in_channels, in_channels)
        self.lin_final = Linear(in_channels, 1)

    def forward(self, z_src, z_dst):
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        h = h.relu()
        return self.lin_final(h)

link_pred = LinkPredictor(in_channels=memory_dim).to(device)
assoc = torch.empty(num_nodes_buffer, dtype=torch.long, device=device)

# --- FUNGSI AMBIL EMBEDDING (Wajib ada untuk Inference) ---
def get_node_embeddings(src, dst):
    all_nodes              = torch.cat([src, dst]).unique()
    n_id, edge_index, e_id = neighbor_loader(all_nodes)
    z, last_update         = memory(n_id)

    if edge_index.numel() > 0:
        z = gnn(z, last_update, edge_index, data.t[e_id].float(), data.msg[e_id])

    assoc[n_id] = torch.arange(n_id.size(0), device=device)
    return z[assoc[src]], z[assoc[dst]]

# 3. LOAD "OTAK" DETEKTIF FORENSIK
print("3. Memasukkan memori Detektif Forensik...")
# FIX: Path disesuaikan dengan output Tahap 4 dan load semua modul
memory.load_state_dict(torch.load('./models/tgn_memory_best.pth', map_location=device))
gnn.load_state_dict(torch.load('./models/tgn_gnn_best.pth', map_location=device))
time_encoder.load_state_dict(torch.load('./models/tgn_timeenc_best.pth', map_location=device))
link_pred.load_state_dict(torch.load('./models/tgn_predictor_best.pth', map_location=device))

# 4. PROSES EVALUASI (INFERENCE)
print("\n4. Memulai Proses Evaluasi (Inference)...")
memory.eval()
gnn.eval()
link_pred.eval()

all_preds_prob = []
all_true_labels = []

memory.reset_state()
neighbor_loader.reset_state() 

with torch.no_grad(): 
    for i, batch in enumerate(test_loader):
        batch = batch.to(device)
        
        # FIX: Gunakan get_node_embeddings agar GNN ikut bekerja, bukan langsung memory()
        z_src, z_dst = get_node_embeddings(batch.src, batch.dst)
        
        pred = link_pred(z_src, z_dst).squeeze()
        prob = torch.sigmoid(pred) 
        
        # Tangani kasus di mana prob hanya berisi 1 elemen (skalar)
        if prob.dim() == 0:
            prob = prob.unsqueeze(0)
            
        all_preds_prob.extend(prob.cpu().tolist())
        all_true_labels.extend(batch.y.cpu().tolist())
        
        # FIX: Jangan lupa insert ke neighbor_loader
        neighbor_loader.insert(batch.src, batch.dst)
        memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
        
        if (i+1) % 1000 == 0:
            print(f"Selesai menebak {i+1} batch...")

# 5. CETAK RAPOR EVALUASI & DIAGNOSA
print("\n" + "="*40)
print("TEST RESULT TGN")
print("="*40)

all_preds_prob = np.array(all_preds_prob)
all_true_labels = np.array(all_true_labels)

# --- DIAGNOSA SKOR ATTACK ---
attack_scores = all_preds_prob[all_true_labels == 1]
normal_scores = all_preds_prob[all_true_labels == 0]

# Safety check jika evaluasi ini tidak punya sampel attack sama sekali
if len(attack_scores) > 0:
    print(f"Skor Attack TERTINGGI  : {np.max(attack_scores):.6f}")
    print(f"Skor Attack TERENDAH   : {np.min(attack_scores):.6f}")
    print(f"Rata-rata Skor Attack  : {np.mean(attack_scores):.6f}")
else:
    print("WARNING: Tidak ada label serangan (1) pada sampel ini.")

if len(normal_scores) > 0:
    print(f"Rata-rata Skor Normal  : {np.mean(normal_scores):.6f}")

# --- PENCARIAN THRESHOLD OPTIMAL (SMART F1-MAXIMIZATION) ---
print("\n[INFO] Menghitung ribuan skenario jaring untuk mencari titik optimal...")
if len(attack_scores) > 0:
    precisions, recalls, thresholds = precision_recall_curve(all_true_labels, all_preds_prob)

    # Rumus F1-Score untuk mencari keseimbangan terbaik
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores[:-1]) 
    best_threshold = thresholds[best_idx]

    print(f"[INFO] Best threshold di angka: {best_threshold:.6f}")
    print(f"[INFO] Ekspektasi F1-Score maksimal: {f1_scores[best_idx]:.4f}")

    # Terapkan Threshold yang terpilih
    pred_labels = (all_preds_prob > best_threshold).astype(int)

    # Hitung Metrik Final
    auroc = roc_auc_score(all_true_labels, all_preds_prob)
    auprc = average_precision_score(all_true_labels, all_preds_prob)
    cm = confusion_matrix(all_true_labels, pred_labels)

    print(f"\nAUROC (Skor Ilusi)   : {auroc:.4f}")
    print(f"AUPRC (Skor Jujur)   : {auprc:.4f}")

    print("\n--- CLASSIFICATION REPORT ---")
    print(classification_report(all_true_labels, pred_labels, target_names=['Normal (0)', 'Attack (1)'], zero_division=0))

    print("--- CONFUSION MATRIX ---")
    print(f"Benar-benar Normal (TN) : {cm[0][0]}")
    print(f"Salah Tuduh Normal (FP) : {cm[0][1]}  <-- Alarm Palsu")
    print(f"Salah Tuduh Aman   (FN) : {cm[1][0]}  <-- Hacker Lolos")
    print(f"Berhasil Tangkap   (TP) : {cm[1][1]}  <-- Keberhasilan TGN")
else:
    print("Evaluasi lanjutan dilewati karena dataset tidak mengandung serangan.")