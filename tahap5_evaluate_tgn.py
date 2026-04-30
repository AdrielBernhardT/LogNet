import torch
from torch.nn import Linear
from torch_geometric.data import TemporalData
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn import TGNMemory
from torch_geometric.nn.models.tgn import IdentityMessage, LastAggregator
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score, roc_auc_score, precision_recall_curve
import numpy as np

# ==========================================
# 1. PERSIAPAN PERANGKAT & DATA
# ==========================================
print("1. Menyiapkan Perangkat dan Data...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_dict = torch.load('tgn_dataset.pt')
data = TemporalData(
    src=data_dict['src'], dst=data_dict['dst'], t=data_dict['t'],
    msg=data_dict['msg'], y=data_dict['y']
).to(device)

test_loader = TemporalDataLoader(data, batch_size=2000)

# ==========================================
# 2. MEMBANGUN ULANG KERANGKA
# ==========================================
print("2. Membangun ulang kerangka TGN...")
memory_dim = 100
time_dim = 100
num_nodes = max(data.src.max(), data.dst.max()).item() + 1

memory = TGNMemory(
    num_nodes=num_nodes, raw_msg_dim=data.msg.size(1),
    memory_dim=memory_dim, time_dim=time_dim,
    message_module=IdentityMessage(data.msg.size(1), memory_dim, time_dim),
    aggregator_module=LastAggregator()
).to(device)

class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.lin_src = Linear(in_channels, in_channels)
        self.lin_dst = Linear(in_channels, in_channels)
        self.lin_final = Linear(in_channels, 1)

    def forward(self, z_src, z_dst):
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        h = h.relu()
        return self.lin_final(h)

link_pred = LinkPredictor(in_channels=memory_dim).to(device)

# ==========================================
# 3. LOAD "OTAK" DETEKTIF FORENSIK
# ==========================================
print("3. Memasukkan memori Detektif Forensik...")
memory.load_state_dict(torch.load('tgn_memory_model.pth', map_location=device))
link_pred.load_state_dict(torch.load('tgn_predictor_model.pth', map_location=device))

# ==========================================
# 4. PROSES EVALUASI (INFERENCE)
# ==========================================
print("\n4. Memulai Proses Evaluasi (Inference)...")
memory.eval() 
link_pred.eval()

all_preds_prob = []
all_true_labels = []

memory.reset_state() 

with torch.no_grad(): 
    for i, batch in enumerate(test_loader):
        batch = batch.to(device)
        z_src, _ = memory(batch.src)
        z_dst, _ = memory(batch.dst)
        
        pred = link_pred(z_src, z_dst).squeeze()
        prob = torch.sigmoid(pred) 
        
        all_preds_prob.extend(prob.cpu().tolist())
        all_true_labels.extend(batch.y.cpu().tolist())
        
        memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
        
        if (i+1) % 1000 == 0:
            print(f"Selesai menebak {i+1} batch...")

# ==========================================
# 5. CETAK RAPOR EVALUASI & DIAGNOSA
# ==========================================
print("\n" + "="*40)
print("HASIL UJIAN TGN (V4 - DETEKTIF FORENSIK)")
print("="*40)

all_preds_prob = np.array(all_preds_prob)
all_true_labels = np.array(all_true_labels)

# --- DIAGNOSA SKOR ATTACK ---
attack_scores = all_preds_prob[all_true_labels == 1]
normal_scores = all_preds_prob[all_true_labels == 0]

print(f"Skor Attack TERTINGGI  : {np.max(attack_scores):.6f}")
print(f"Skor Attack TERENDAH   : {np.min(attack_scores):.6f}")
print(f"Rata-rata Skor Attack  : {np.mean(attack_scores):.6f}")
print(f"Rata-rata Skor Normal  : {np.mean(normal_scores):.6f}")

# --- PENCARIAN THRESHOLD OPTIMAL (SMART F1-MAXIMIZATION) ---
print("\n[INFO] Menghitung ribuan skenario jaring untuk mencari titik optimal...")
precisions, recalls, thresholds = precision_recall_curve(all_true_labels, all_preds_prob)

# Rumus F1-Score untuk mencari keseimbangan terbaik
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
best_idx = np.argmax(f1_scores[:-1]) 
best_threshold = thresholds[best_idx]

print(f"[INFO] Titik Emas (Sniper Threshold) ditemukan di angka: {best_threshold:.6f}")
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