import torch
from torch.nn import Linear
from torch_geometric.data import TemporalData
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn import TGNMemory
from torch_geometric.nn.models.tgn import IdentityMessage, LastAggregator
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.model_selection import train_test_split
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

is_auth_edge_full = (data.msg[:, 0].cpu().numpy() == 1)

# ==========================================
# 5. PEMISAHAN VALIDASI & TESTING (STRATIFIED AMAN UNTUK JURNAL)
# ==========================================
# Sisa 50% data masa depan diambil
train_size = int(0.50 * len(all_preds_prob))
future_preds_prob = all_preds_prob[train_size:]
future_true_labels = all_true_labels[train_size:]
future_is_auth = is_auth_edge_full[train_size:]

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
val_is_auth = future_is_auth[idx_val]

test_preds_prob = future_preds_prob[idx_test]
test_true_labels = future_true_labels[idx_test]
test_is_auth = future_is_auth[idx_test]

print(f"[OK] Validasi Terbagi Stratified -> Validasi: {len(val_preds_prob)} | Ujian Akhir (Test): {len(test_preds_prob)}")

total_attack_di_val = np.sum(val_true_labels == 1)
total_attack_di_test = np.sum(test_true_labels == 1)
print(f"[INFO] Jumlah Serangan -> Di Validasi: {total_attack_di_val} | Di Test Set: {total_attack_di_test}")

TARGET_RECALL = 0.60
print(f"[INFO] Target Recall di Set Validasi: {TARGET_RECALL*100:.0f}%")

# ==========================================
# 6. PENCARIAN THRESHOLD - METODE DETERMINISTIK (PENGGANTI PSO SEBAGAI METODE UTAMA)
# ==========================================
print("\n[DETERMINISTIK] Menghitung precision-recall curve di Data Validasi...")
precisions_arr, recalls_arr, thresholds_arr = precision_recall_curve(val_true_labels, val_preds_prob)


def cari_threshold_deterministik(target_recall):
    """
    Cari threshold TERBESAR (paling ketat / FP paling sedikit) yang tetap memenuhi target_recall.
    precisions_arr & recalls_arr punya 1 elemen lebih banyak dari thresholds_arr (elemen terakhir
    tidak punya threshold berpasangan, itu untuk kasus "prediksi semua negatif"), jadi elemen
    terakhir di-exclude saat mencari indeks yang valid.
    """
    valid_idx = np.where(recalls_arr[:-1] >= target_recall)[0]
    if len(valid_idx) == 0:
        return thresholds_arr[0], recalls_arr[0], precisions_arr[0]
    best_idx = valid_idx[np.argmax(thresholds_arr[valid_idx])]
    return thresholds_arr[best_idx], recalls_arr[best_idx], precisions_arr[best_idx]


threshold_deterministik, recall_tercapai, precision_val = cari_threshold_deterministik(TARGET_RECALL)
print(f"[DETERMINISTIK] Threshold utk target recall {TARGET_RECALL*100:.0f}% : {threshold_deterministik:.6f}")
print(f"[DETERMINISTIK] Recall tercapai di Validasi         : {recall_tercapai*100:.2f}%")
print(f"[DETERMINISTIK] Precision di Validasi                : {precision_val*100:.2f}%")

print("\n[TABEL TRADE-OFF] Threshold pada berbagai target recall (dihitung di Data Validasi):")
print(f"{'Target Recall':>15} | {'Threshold':>12} | {'Recall Aktual':>14} | {'Precision':>10}")
for target in [0.50, 0.60, 0.70, 0.80, 0.90]:
    thr, rec, prec = cari_threshold_deterministik(target)
    print(f"{target*100:>14.0f}% | {thr:>12.6f} | {rec*100:>13.2f}% | {prec*100:>9.2f}%")
print(
    "Catatan: target 60% yang dipakai sebagai threshold utama BUKAN satu-satunya pilihan yang "
    "mungkin -- tabel di atas menunjukkan trade-off di target lain, supaya pemilihan 60% "
    "transparan dan bisa dijustifikasi (bukan angka yang 'kebetulan kelihatan bagus')."
)

plt.figure(figsize=(7, 5))
plt.plot(recalls_arr, precisions_arr, marker='.', linewidth=1, label='Precision-Recall (Data Validasi)')
plt.axvline(x=TARGET_RECALL, color='red', linestyle='--', label=f'Target Recall {TARGET_RECALL*100:.0f}%')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Kurva Precision-Recall di Data Validasi')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('precision_recall_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("[PLOT] Kurva precision-recall disimpan di: precision_recall_curve.png")

np.random.seed(42)


def objective_function(threshold):
    pred_labels = (val_preds_prob > threshold).astype(int)
    tp = np.sum((pred_labels == 1) & (val_true_labels == 1))
    fp = np.sum((pred_labels == 1) & (val_true_labels == 0))

    target_tp_val = int(TARGET_RECALL * total_attack_di_val) if total_attack_di_val > 0 else 1
    if tp < target_tp_val:
        return fp + 100000000 + ((target_tp_val - tp) * 1000000)
    else:
        return fp


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

print("\n[PSO - validasi silang] Menjalankan Swarm Intelligence di Set Validasi...")
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

print(f"[PSO - validasi silang] Threshold hasil PSO (seeded) : {gbest_position:.6f}")
print(f"[BANDINGKAN] Deterministik: {threshold_deterministik:.6f}  vs  PSO: {gbest_position:.6f}")
selisih_pct = abs(threshold_deterministik - gbest_position) / threshold_deterministik * 100 if threshold_deterministik > 0 else 0
print(f"[BANDINGKAN] Selisih relatif: {selisih_pct:.2f}% (kalau kecil, ini memperkuat validitas threshold deterministik)")

# Threshold yang benar-benar dipakai untuk evaluasi akhir = hasil DETERMINISTIK (bukan PSO)
threshold_final = threshold_deterministik

# ==========================================
# 7. PENILAIAN MUTLAK DI DATA TEST (EVALUASI PAPERS)
# ==========================================
print("\n" + "=" * 40)
print("HASIL EVALUASI AKHIR (MURNI DATA TEST SEJATI, PROPORSI TERUNDERSAMPLE)")
print("=" * 40)

final_test_preds = (test_preds_prob > threshold_final).astype(int)
auroc_test = roc_auc_score(test_true_labels, test_preds_prob)
cm_test = confusion_matrix(test_true_labels, final_test_preds)

print(f"AUROC Sejati di Test Set : {auroc_test:.4f}")
print("\n--- CLASSIFICATION REPORT (UNSEEN FUTURE TEST SET, DATA SUDAH DI-UNDERSAMPLE) ---")
print(classification_report(test_true_labels, final_test_preds, target_names=['Normal (0)', 'Attack (1)'], zero_division=0))

print("--- CONFUSION MATRIX (OBSERVED, DATA TERUNDERSAMPLE) ---")
print(f"True Negative  (TN) : {cm_test[0][0]}")
print(f"False Positive (FP) : {cm_test[0][1]}  <-- Angka Salah Tuduh di data yg sudah di-undersample")
print(f"False Negative (FN) : {cm_test[1][0]}  <-- Hacker yang Lolos Riil")
print(f"True Positive  (TP) : {cm_test[1][1]}  <-- Berhasil Ditangkap")

print("\n" + "=" * 40)
print("ESTIMASI PERFORMA DI PROPORSI KELAS ASLI (TANPA UNDERSAMPLING)")
print("=" * 40)
print(
    "Confusion matrix di atas dihitung pada data yang sudah di-undersampling sejak Tahap 1 & 2\n"
    "(hanya sebagian kecil baris normal auth & flow yang disimpan, semua serangan disimpan penuh).\n"
    "Bagian ini mengoreksi angkanya dengan inverse probability weighting: setiap baris normal yang\n"
    "diamati di-'timbang ulang' sesuai kebalikan dari peluang dia disimpan saat sampling, sehingga\n"
    "hasilnya adalah ESTIMASI performa model jika diuji pada proporsi kelas ASLI di dunia nyata."
)

with open('sampling_metadata.json') as f:
    sampling_meta = json.load(f)

auth_frac = sampling_meta['actual_sampling_fraction']
flow_frac = sampling_meta.get('actual_flow_sampling_fraction', 0.005)
print(f"\n[METADATA] Fraksi sampling auth-normal : {auth_frac:.6f} (dari Tahap 1)")
print(f"[METADATA] Fraksi sampling flow         : {flow_frac:.6f} (dari Tahap 2)")


def hitung_confusion_terkoreksi(true_labels, pred_labels, is_auth_mask, auth_frac, flow_frac):
    auth_mask = is_auth_mask
    flow_mask = ~is_auth_mask

    tp = int(np.sum((pred_labels == 1) & (true_labels == 1) & auth_mask))
    fn = int(np.sum((pred_labels == 0) & (true_labels == 1) & auth_mask))

    tn_auth = int(np.sum((pred_labels == 0) & (true_labels == 0) & auth_mask))
    fp_auth = int(np.sum((pred_labels == 1) & (true_labels == 0) & auth_mask))

    tn_flow = int(np.sum((pred_labels == 0) & flow_mask))
    fp_flow = int(np.sum((pred_labels == 1) & flow_mask))

    tn_estimasi = (tn_auth / auth_frac) + (tn_flow / flow_frac)
    fp_estimasi = (fp_auth / auth_frac) + (fp_flow / flow_frac)

    return {
        'TP': tp, 'FN': fn,
        'TN_estimasi': tn_estimasi, 'FP_estimasi': fp_estimasi,
        'TN_auth_observed': tn_auth, 'FP_auth_observed': fp_auth,
        'TN_flow_observed': tn_flow, 'FP_flow_observed': fp_flow,
    }


hasil_koreksi = hitung_confusion_terkoreksi(
    test_true_labels, final_test_preds, test_is_auth, auth_frac, flow_frac
)

tp = hasil_koreksi['TP']
fn = hasil_koreksi['FN']
fp_est = hasil_koreksi['FP_estimasi']
tn_est = hasil_koreksi['TN_estimasi']

precision_est = tp / (tp + fp_est) if (tp + fp_est) > 0 else 0.0
recall_est = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1_est = (2 * precision_est * recall_est / (precision_est + recall_est)) if (precision_est + recall_est) > 0 else 0.0

print(f"\n[OBSERVED - data sudah di-undersampling]")
print(f"  TP={tp}  FN={fn}  TN(auth)={hasil_koreksi['TN_auth_observed']}  FP(auth)={hasil_koreksi['FP_auth_observed']}")
print(f"  TN(flow)={hasil_koreksi['TN_flow_observed']}  FP(flow)={hasil_koreksi['FP_flow_observed']}")

print(f"\n[ESTIMASI DUNIA NYATA - dikoreksi dengan inverse probability weighting]")
print(f"  TP (tidak berubah)        : {tp:,}")
print(f"  FN (tidak berubah)        : {fn:,}")
print(f"  TN (estimasi)             : {tn_est:,.0f}")
print(f"  FP (estimasi)             : {fp_est:,.0f}")
print(f"  Precision (estimasi)      : {precision_est:.6f}  ({precision_est*100:.4f}%)")
print(f"  Recall (sama seperti di atas) : {recall_est:.4f}  ({recall_est*100:.2f}%)")
print(f"  F1-score (estimasi)       : {f1_est:.6f}")

print(
    "\n[CATATAN KETERBATASAN] Angka 'estimasi dunia nyata' di atas adalah PENDEKATAN STATISTIK\n"
    "(inverse probability weighting berdasarkan fraksi sampling yang diketahui secara pasti dari\n"
    "Tahap 1 & 2), BUKAN hasil menjalankan ulang model pada data mentah yang sepenuhnya utuh.\n"
    "Model tetap dilatih & membangun memory state dari urutan data yang sudah di-undersampling,\n"
    "sehingga estimasi ini mengasumsikan pola false-positive rate yang teramati bersifat\n"
    "representatif/konstan terhadap populasi yang lebih besar. Untuk validasi yang lebih kuat,\n"
    "idealnya model dijalankan ulang pada aliran data mentah asli tanpa undersampling sama sekali\n"
    "(lebih berat secara komputasi, tapi bisa dilakukan sebagai pekerjaan lanjutan)."
)