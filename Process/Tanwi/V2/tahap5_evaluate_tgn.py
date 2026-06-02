import torch
from torch.nn import Linear
from torch_geometric.data import TemporalData
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn import TGNMemory
from torch_geometric.nn.models.tgn import IdentityMessage, LastAggregator
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
import numpy as np

# ==========================================
# 1-3. INITIATION & LOAD MODEL
# ==========================================
print("1. Menyiapkan Perangkat dan Data...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_dict = torch.load('tgn_dataset.pt')
data = TemporalData(
    src=data_dict['src'], dst=data_dict['dst'], t=data_dict['t'],
    msg=data_dict['msg'], y=data_dict['y']
).to(device)

test_loader = TemporalDataLoader(data, batch_size=2000)

print("2. Membangun ulang struktur TGN...")
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

memory.load_state_dict(torch.load('tgn_memory_model.pth', map_location=device))
link_pred.load_state_dict(torch.load('tgn_predictor_model.pth', map_location=device))

# ==========================================
# 4. FULL INFERENCE FOR MEMORY BUILDING
# ==========================================
print("\n4. Memulai Inference Global untuk Sinkronisasi State...")
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

all_preds_prob = np.array(all_preds_prob)
all_true_labels = np.array(all_true_labels)

# ==========================================
# 5. PEMISAHAN VALIDASI & TESTING (STRATIFIED AMAN UNTUK JURNAL)
# ==========================================
# Sisa 50% data masa depan diambil
train_size = int(0.50 * len(all_preds_prob))
future_preds_prob = all_preds_prob[train_size:]
future_true_labels = all_true_labels[train_size:]

indices = np.arange(len(future_preds_prob))

# Lakukan stratified split berdasarkan label asli (y) agar kelas serangan terbagi rata
if np.sum(future_true_labels == 1) > 1:
    idx_val, idx_test = train_test_split(
        indices, test_size=0.50, random_state=42, stratify=future_true_labels
    )
else:
    idx_val, idx_test = train_test_split(indices, test_size=0.50, random_state=42)

# Susun kembali subset array berdasarkan indeks hasil stratified split
val_preds_prob = future_preds_prob[idx_val]
val_true_labels = future_true_labels[idx_val]

test_preds_prob = future_preds_prob[idx_test]
test_true_labels = future_true_labels[idx_test]

print(f"[OK] Validasi Terbagi Stratified -> Validasi PSO: {len(val_preds_prob)} | Ujian Akhir (Test): {len(test_preds_prob)}")

total_attack_di_val = np.sum(val_true_labels == 1)
total_attack_di_test = np.sum(test_true_labels == 1)
print(f"[INFO] Jumlah Serangan -> Di Validasi: {total_attack_di_val} | Di Test Set: {total_attack_di_test}")

TARGET_TP_VAL = int(0.60 * total_attack_di_val) if total_attack_di_val > 0 else 1
print(f"[INFO] Target Tangkapan PSO di Set Validasi (60% Recall): {TARGET_TP_VAL} Serangan")

# ==========================================
# EXTRA: ALGORITMA OPTIMASI PSO (MENCARI THRESHOLD)
# ==========================================
# Fungsi Objektif PSO hanya mengakses Data Validasi untuk menghindari data leakage
def objective_function(threshold):
    pred_labels = (val_preds_prob > threshold).astype(int)
    tp = np.sum((pred_labels == 1) & (val_true_labels == 1))
    fp = np.sum((pred_labels == 1) & (val_true_labels == 0))
    
    if tp < TARGET_TP_VAL:
        # Hukuman berat jika target recall minimal tidak tercapai di set validasi
        return fp + 100000000 + ((TARGET_TP_VAL - tp) * 1000000)
    else:
        return fp

# Hyperparameter PSO
num_particles = 30
num_iterations = 50
w, c1, c2 = 0.5, 1.5, 1.5

positions = np.random.uniform(0.0000001, np.max(val_preds_prob) if total_attack_di_val > 0 else 0.5, num_particles)
velocities = np.zeros(num_particles)
pbest_positions = np.copy(positions)
pbest_scores = np.array([objective_function(p) for p in positions])

gbest_idx = np.argmin(pbest_scores)
gbest_position = pbest_positions[gbest_idx]
gbest_score = pbest_scores[gbest_idx]

print("\n[PSO] Menjalankan Swarm Intelligence di Set Validasi...")
for iteration in range(num_iterations):
    r1, r2 = np.random.rand(num_particles), np.random.rand(num_particles)
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

print(f"[PSO] Menemukan Threshold Emas di Set Validasi: {gbest_position:.6f}")

# ==========================================
# 6. PENILAIAN MUTLAK DI DATA TEST (EVALUASI PAPERS)
# ==========================================
print("\n" + "="*40)
print("HASIL EVALUASI AKHIR (MURNI DATA TEST SEJATI)")
print("="*40)

# Terapkan Threshold hasil Validasi murni ke dalam DATA TEST
final_test_preds = (test_preds_prob > gbest_position).astype(int)
auroc_test = roc_auc_score(test_true_labels, test_preds_prob)
cm_test = confusion_matrix(test_true_labels, final_test_preds)

print(f"AUROC Sejati di Test Set : {auroc_test:.4f}")
print("\n--- CLASSIFICATION REPORT (UNSEEN FUTURE TEST SET) ---")
print(classification_report(test_true_labels, final_test_preds, target_names=['Normal (0)', 'Attack (1)'], zero_division=0))

print("--- CONFUSION MATRIX ---")
print(f"True Negative  (TN) : {cm_test[0][0]}")
print(f"False Positive (FP) : {cm_test[0][1]}  <-- Angka Salah Tuduh Riil")
print(f"False Negative (FN) : {cm_test[1][0]}  <-- Hacker yang Lolos Riil")
print(f"True Positive  (TP) : {cm_test[1][1]}  <-- Berhasil Ditangkap")