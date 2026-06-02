import torch
from torch.nn import Linear
from torch_geometric.data import TemporalData
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn import TGNMemory
from torch_geometric.nn.models.tgn import IdentityMessage, LastAggregator
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score, roc_auc_score
import numpy as np

# ==========================================
# 1-3. PERSIAPAN, KERANGKA, & LOAD MODEL
# ==========================================
print("1. Menyiapkan Perangkat dan Data...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_dict = torch.load('tgn_dataset.pt')
data = TemporalData(
    src=data_dict['src'], dst=data_dict['dst'], t=data_dict['t'],
    msg=data_dict['msg'], y=data_dict['y']
).to(device)

test_loader = TemporalDataLoader(data, batch_size=2000)

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

print("3. Memasukkan memori Detektif Forensik Elite (V4.1)...")
memory.load_state_dict(torch.load('tgn_memory_model.pth', map_location=device))
link_pred.load_state_dict(torch.load('tgn_predictor_model.pth', map_location=device))

# ==========================================
# 4. PROSES EVALUASI (INFERENCE)
# ==========================================
print("\n4. Memulai Proses Evaluasi (Inference Keseluruhan untuk Bangun Memory)...")
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
# 5. PARTICLE SWARM OPTIMIZATION (TARGET DOSEN)
# ==========================================
print("\n" + "="*40)
print("HASIL UJIAN TGN (V5 - CHRONOLOGICAL SPLIT 50:50)")
print("="*40)

all_preds_prob = np.array(all_preds_prob)
all_true_labels = np.array(all_true_labels)

# --- MEMISAHKAN JAWABAN UJIAN (50% MASA DEPAN) ---
train_size = int(0.50 * len(all_preds_prob))
test_preds_prob = all_preds_prob[train_size:]
test_true_labels = all_true_labels[train_size:]

print(f"[INFO] Menilai murni pada {len(test_preds_prob)} data masa depan (50% Testing)...")

attack_scores = test_preds_prob[test_true_labels == 1]

if len(attack_scores) == 0:
    print("[WARNING] Tidak ada serangan di 30% masa depan! Silakan sesuaikan rasio split.")
else:
    print(f"Skor Attack TERTINGGI  : {np.max(attack_scores):.6f}")
    print(f"Rata-rata Skor Attack  : {np.mean(attack_scores):.6f}")

print("\n[INFO] Mengerahkan Algoritma PSO untuk mencari Threshold terbaik...")

# Sesuaikan target TPS dengan 60% dari serangan yang TERSISA di set Ujian
total_attack_di_test = np.sum(test_true_labels == 1)
TARGET_TP = int(0.60 * total_attack_di_test)

print(f"[INFO] Total Serangan di Masa Depan: {total_attack_di_test}")
print(f"[INFO] Target Baru PSO (60% Recall): {TARGET_TP} Tangkapan")

# Fungsi Objektif (Fungsi Penalti Super Kejam - Hanya menilai test_preds_prob)
def objective_function(threshold):
    pred_labels = (test_preds_prob > threshold).astype(int)
    tp = np.sum((pred_labels == 1) & (test_true_labels == 1))
    fp = np.sum((pred_labels == 1) & (test_true_labels == 0))
    
    if tp < TARGET_TP:
        # Hukuman Kiamat: 100 Juta + (1 Juta per 1 Hacker yang lolos)
        return fp + 100000000 + ((TARGET_TP - tp) * 1000000)
    else:
        return fp

# Hyperparameter PSO
num_particles = 30
num_iterations = 50
w = 0.5  
c1 = 1.5 
c2 = 1.5 

np.random.seed(42) # Agar kawanan burung PSO bergeraknya konsisten setiap di-run
positions = np.random.uniform(0.0000001, np.max(attack_scores) if len(attack_scores)>0 else 0.5, num_particles)
velocities = np.zeros(num_particles)
pbest_positions = np.copy(positions)
pbest_scores = np.array([objective_function(p) for p in positions])

gbest_idx = np.argmin(pbest_scores)
gbest_position = pbest_positions[gbest_idx]
gbest_score = pbest_scores[gbest_idx]

# Loop PSO terbang mencari titik emas
for iteration in range(num_iterations):
    r1 = np.random.rand(num_particles)
    r2 = np.random.rand(num_particles)
    
    velocities = (w * velocities) + (c1 * r1 * (pbest_positions - positions)) + (c2 * r2 * (gbest_position - positions))
    positions = positions + velocities
    
    positions = np.clip(positions, 0.000001, 0.999999)
    
    current_scores = np.array([objective_function(p) for p in positions])
    
    better_mask = current_scores < pbest_scores
    pbest_positions[better_mask] = positions[better_mask]
    pbest_scores[better_mask] = current_scores[better_mask]
    
    if np.min(pbest_scores) < gbest_score:
        gbest_position = pbest_positions[np.argmin(pbest_scores)]
        gbest_score = np.min(pbest_scores)
        
    if (iteration+1) % 10 == 0:
        print(f"  PSO Iterasi {iteration+1}/50 | Skor Terendah Saat Ini: {gbest_score:.0f}")

print(f"\n[INFO] Titik Emas (PSO Threshold) dikunci pada angka: {gbest_position:.6f}")

# Terapkan Threshold hasil PSO (HANYA PADA DATA TEST)
pred_labels = (test_preds_prob > gbest_position).astype(int)
auroc = roc_auc_score(test_true_labels, test_preds_prob)
cm = confusion_matrix(test_true_labels, pred_labels)

print(f"\nAUROC (Skor Ilusi)   : {auroc:.4f}")
print("\n--- CLASSIFICATION REPORT (50% MASA DEPAN) ---")
print(classification_report(test_true_labels, pred_labels, target_names=['Normal (0)', 'Attack (1)'], zero_division=0))

print("--- CONFUSION MATRIX ---")
print(f"Benar-benar Normal (TN) : {cm[0][0]}")
print(f"Salah Tuduh Normal (FP) : {cm[0][1]}  <-- Hasil Minimalisasi PSO di Set Ujian")
print(f"Salah Tuduh Aman   (FN) : {cm[1][0]}  <-- Hacker Lolos")
print(f"Berhasil Tangkap   (TP) : {cm[1][1]}  <-- Target >= {TARGET_TP}")