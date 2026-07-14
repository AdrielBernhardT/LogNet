import os
import csv
import numpy as np
import torch
from torch.nn import Linear, Dropout
from torch_geometric.data import TemporalData
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn import TGNMemory
from torch_geometric.nn.models.tgn import IdentityMessage, LastAggregator
from sklearn.metrics import average_precision_score, roc_auc_score

# ==========================================
# 0. REPRODUCIBILITY (V4.2 - TAMBAHAN)
# ==========================================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("1. Menyiapkan Perangkat dan Data...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Menggunakan Device: {device}")

data_dict = torch.load('tgn_dataset.pt')
data = TemporalData(
    src=data_dict['src'], dst=data_dict['dst'], t=data_dict['t'],
    msg=data_dict['msg'], y=data_dict['y']
)

# ==========================================
# CHRONOLOGICAL SPLIT: 45% TRAIN + 5% VAL-INTERNAL (V4.2 - TAMBAHAN)
# ==========================================
# PENTING: batas 50% di bawah ini HARUS tetap sinkron dengan
# tahap5_evaluate_tgn.py, yang menganggap 50% pertama dari tgn_dataset.pt
# sebagai "masa lalu" (pernah dilihat pipeline training ini secara causal)
# dan 50% sisanya sebagai "masa depan" murni (val/test evaluasi akhir).
# Di dalam 50% pertama itu, sekarang dipecah lagi jadi:
#   0%  - 45% (TRAIN_FRAC) -> dipakai untuk gradient update (bobot berubah)
#   45% - 50% (VAL_FRAC)   -> val-internal, HANYA untuk monitoring & pemilihan
#                             checkpoint terbaik, TIDAK ADA gradient update
TRAIN_FRAC = 0.45
VAL_FRAC = 0.05

total_rows = data.msg.size(0)
train_size = int(TRAIN_FRAC * total_rows)
val_end = int((TRAIN_FRAC + VAL_FRAC) * total_rows)

print(f"\n[INFO] Total Baris Data: {total_rows}")
print(f"[INFO] Train (gradient update)   : baris 0 - {train_size} ({TRAIN_FRAC*100:.0f}%)")
print(f"[INFO] Val-internal (monitoring) : baris {train_size} - {val_end} ({VAL_FRAC*100:.0f}%)")
print(f"[INFO] Baris {val_end} - {total_rows} DIBIARKAN UTUH untuk Tahap 5 (val/test evaluasi akhir)")


def slice_temporal(src_data, start, end):
    # Gunakan .clone() untuk memutus dependensi memori GPU/CPU ke tensor asli
    return TemporalData(
        src=src_data.src[start:end].clone(),
        dst=src_data.dst[start:end].clone(),
        t=src_data.t[start:end].clone(),
        msg=src_data.msg[start:end].clone(),
        y=src_data.y[start:end].clone()
    )


train_data = slice_temporal(data, 0, train_size).to(device)
val_data = slice_temporal(data, train_size, val_end).to(device)

train_loader = TemporalDataLoader(train_data, batch_size=2000)
val_loader = TemporalDataLoader(val_data, batch_size=2000)

# ==========================================
# 2. WEIGHTED LOSS (MENGATASI IMBALANCE)
# ==========================================
jumlah_normal = (train_data.y == 0).sum().item()
jumlah_serangan = (train_data.y == 1).sum().item()

if jumlah_serangan == 0:
    jumlah_serangan = 1

pos_weight_value = torch.tensor([jumlah_normal / jumlah_serangan], dtype=torch.float, device=device)
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_value)
print(f"-> Bobot Penalti Serangan: {pos_weight_value.item():.2f}x lipat lebih berat")

# ==========================================
# 3. MERAKIT MODEL TGN
# ==========================================
print("\n3. Merakit Model TGN...")
memory_dim = 100
time_dim = 100
num_nodes = max(data.src.max(), data.dst.max()).item() + 1

memory = TGNMemory(
    num_nodes=num_nodes,
    raw_msg_dim=data.msg.size(1),
    memory_dim=memory_dim, time_dim=time_dim,
    message_module=IdentityMessage(data.msg.size(1), memory_dim, time_dim),
    aggregator_module=LastAggregator()
).to(device)


class LinkPredictor(torch.nn.Module):
    # V4.2 - TAMBAHAN: dropout di hidden layer buat mitigasi overfitting,
    # relevan sekarang karena jumlah epoch dinaikkan (lihat bagian training loop)
    def __init__(self, in_channels, dropout_p=0.2):
        super().__init__()
        self.lin_src = Linear(in_channels, in_channels)
        self.lin_dst = Linear(in_channels, in_channels)
        self.dropout = Dropout(p=dropout_p)
        self.lin_final = Linear(in_channels, 1)

    def forward(self, z_src, z_dst):
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        h = h.relu()
        h = self.dropout(h)
        return self.lin_final(h)


link_pred = LinkPredictor(in_channels=memory_dim, dropout_p=0.2).to(device)
optimizer = torch.optim.Adam(set(memory.parameters()) | set(link_pred.parameters()), lr=0.0001)

# V4.2 - TAMBAHAN: LR scheduler, turunkan LR kalau val AP mandek
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=2
)

# ==========================================
# 4. TRAINING LOOP + VALIDASI INTERNAL + CHECKPOINT (V4.2)
# ==========================================
NUM_EPOCHS = 15          # dinaikkan dari 5 -> aman krn ada early stopping + checkpoint terbaik
EARLY_STOP_PATIENCE = 5  # stop kalau val AP gak membaik N epoch berturut-turut

LOG_FILE = 'training_log.csv'
MEMORY_CKPT = 'tgn_memory_model.pth'
PREDICTOR_CKPT = 'tgn_predictor_model.pth'

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'w', newline='') as f:
        csv.writer(f).writerow(
            ['epoch', 'train_avg_loss', 'val_avg_precision_ap', 'val_auroc', 'lr', 'is_best']
        )

print(f"\n4. Memulai Training ({NUM_EPOCHS} EPOCH, early stopping patience={EARLY_STOP_PATIENCE})...")

best_val_ap = -1.0
best_epoch = -1
epochs_no_improve = 0
last_epoch_run = 0

for epoch in range(NUM_EPOCHS):
    last_epoch_run = epoch + 1

    # ---- TRAIN (0% - 45%): update bobot ----
    memory.train()
    link_pred.train()
    memory.reset_state()
    total_loss = 0

    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        batch = batch.to(device)

        memory.detach()

        z_src, _ = memory(batch.src)
        z_dst, _ = memory(batch.dst)

        pred = link_pred(z_src, z_dst).squeeze()
        loss = criterion(pred, batch.y)
        loss.backward()
        optimizer.step()

        memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
        total_loss += loss.item()

        if (i + 1) % 100 == 0:
            print(f"Epoch {epoch+1} | Batch {i+1} | Loss Sementara: {loss.item():.4f}")

    avg_train_loss = total_loss / len(train_loader)
    print(f"=== Selesai Epoch {epoch+1} | Rata-rata Train Loss: {avg_train_loss:.4f} ===")

    # ---- VALIDASI INTERNAL (45% - 50%): lanjut causal, TANPA gradient ----
    # memory TIDAK di-reset di sini supaya val-internal dievaluasi dengan state
    # yang benar-benar melanjutkan (causal) dari akhir batch training terakhir.
    memory.eval()
    link_pred.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            z_src, _ = memory(batch.src)
            z_dst, _ = memory(batch.dst)
            pred = link_pred(z_src, z_dst).squeeze()

            val_preds.extend(torch.sigmoid(pred).cpu().tolist())
            val_labels.extend(batch.y.cpu().tolist())

            # tetap update memory secara causal walau lagi eval, supaya
            # state yang dipakai konsisten dgn cara Tahap 5 melakukan replay
            memory.update_state(batch.src, batch.dst, batch.t, batch.msg)

    val_labels_np = np.array(val_labels)
    val_preds_np = np.array(val_preds)

    if val_labels_np.sum() > 0:
        val_ap = average_precision_score(val_labels_np, val_preds_np)
        val_auroc = roc_auc_score(val_labels_np, val_preds_np)
    else:
        # kalau kebetulan gak ada serangan di window val-internal, AP/AUROC gak terdefinisi
        val_ap, val_auroc = float('nan'), float('nan')

    print(f"    [VAL-INTERNAL] AP: {val_ap:.4f} | AUROC: {val_auroc:.4f} | Serangan di val: {int(val_labels_np.sum())}")

    if not np.isnan(val_ap):
        scheduler.step(val_ap)

    is_best = (not np.isnan(val_ap)) and (val_ap > best_val_ap)
    if is_best:
        best_val_ap = val_ap
        best_epoch = epoch + 1
        epochs_no_improve = 0
        torch.save(memory.state_dict(), MEMORY_CKPT)
        torch.save(link_pred.state_dict(), PREDICTOR_CKPT)
        print(f"    [CHECKPOINT] Val AP membaik ({best_val_ap:.4f}) -> model disimpan (epoch {best_epoch})")
    else:
        epochs_no_improve += 1
        print(f"    [CHECKPOINT] Val AP tidak membaik ({epochs_no_improve}/{EARLY_STOP_PATIENCE})")

    current_lr = optimizer.param_groups[0]['lr']
    with open(LOG_FILE, 'a', newline='') as f:
        csv.writer(f).writerow([
            epoch + 1, f"{avg_train_loss:.6f}", f"{val_ap:.6f}",
            f"{val_auroc:.6f}", f"{current_lr:.8f}", int(is_best)
        ])

    if epochs_no_improve >= EARLY_STOP_PATIENCE:
        print(f"\n[EARLY STOPPING] Val AP tidak membaik {EARLY_STOP_PATIENCE} epoch berturut-turut. Stop di epoch {epoch+1}.")
        break

if best_epoch == -1:
    # fallback: val AP selalu NaN (kemungkinan kecil, misal tidak ada serangan
    # sama sekali di window val-internal) -> tetap simpan bobot epoch terakhir
    # supaya Tahap 5 selalu punya checkpoint untuk di-load.
    print("[WARNING] Tidak ada epoch yang tercatat 'membaik' (val AP selalu NaN). "
          "Menyimpan bobot epoch terakhir sebagai fallback.")
    torch.save(memory.state_dict(), MEMORY_CKPT)
    torch.save(link_pred.state_dict(), PREDICTOR_CKPT)
    best_epoch = last_epoch_run

print(f"\nModel Forensik Terbaik Disimpan (epoch {best_epoch}, Val AP={best_val_ap:.4f})!")
print(f"Log training lengkap (per-epoch loss & val metric) ada di: {LOG_FILE}")
