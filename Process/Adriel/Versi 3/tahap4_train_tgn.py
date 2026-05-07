import torch
import os
from torch_geometric.data import TemporalData
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn import TGNMemory, TransformerConv
from torch.nn import Linear, ReLU
from torch_geometric.nn.models.tgn import LastAggregator, LastNeighborLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score

os.makedirs('./models', exist_ok=True)

# 1. PERSIAPAN PERANGKAT & DATA
print("1. Menyiapkan Perangkat dan Data...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Menggunakan Device: {device}")

data_dict = torch.load('./datasets/tgn_dataset.pt')
data = TemporalData(
    src=data_dict['src'],
    dst=data_dict['dst'],
    t=data_dict['t'].long(),
    msg=data_dict['msg'],
    y=data_dict['y']
)

# 2. TEMPORAL SPLIT (INDEX-BASED)
print("\n2. Membuat Temporal Train/Val/Test Split (index-based)...")

t_vals = data.t
print(f"   Timestamp — min: {t_vals.min().item()}, "
      f"max: {t_vals.max().item()}, "
      f"unique: {t_vals.unique().numel():,} dari {t_vals.numel():,} events")

perm = data.t.argsort(stable=True)
data.src = data.src[perm]
data.dst = data.dst[perm]
data.t   = data.t[perm]
data.msg = data.msg[perm]
data.y   = data.y[perm]

n_total = data.t.size(0)
n_train = int(n_total * 0.35)
n_val   = int(n_total * 0.20)

idx_train_end = n_train
idx_val_end   = n_train + n_val

def slice_temporal(data, start, end):
    return TemporalData(
        src=data.src[start:end], dst=data.dst[start:end],
        t=data.t[start:end], msg=data.msg[start:end], y=data.y[start:end],
    )

train_data = slice_temporal(data, 0, idx_train_end)
val_data   = slice_temporal(data, idx_train_end, idx_val_end)
test_data  = slice_temporal(data, idx_val_end,   n_total)

print(f"-> Train : {train_data.num_events:,} events")
print(f"-> Val   : {val_data.num_events:,} events")
print(f"-> Test  : {test_data.num_events:,} events")

train_data = train_data.to(device)
val_data   = val_data.to(device)
test_data  = test_data.to(device)
data       = data.to(device)

train_loader  = TemporalDataLoader(train_data, batch_size=2000)
val_loader    = TemporalDataLoader(val_data,   batch_size=2000)
test_loader   = TemporalDataLoader(test_data,  batch_size=2000)
warmup_loader = TemporalDataLoader(train_data, batch_size=5000)

# 3. WEIGHTED LOSS (Dari TRAIN saja)
print("\n3. Menghitung Class Weight dari Data Training...")
jumlah_normal   = (train_data.y == 0).sum().item()
jumlah_serangan = (train_data.y == 1).sum().item()

if jumlah_serangan == 0:
    raise ValueError("Tidak ada label serangan (y=1) di training set.")

pos_weight_val = torch.tensor([jumlah_normal / jumlah_serangan], device=device)
criterion      = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_val)
print(f"-> Normal / Serangan: {jumlah_normal:,} / {jumlah_serangan:,}")
print(f"-> Bobot serangan   : {pos_weight_val.item():.2f}x")

# 4. DEFINISI MODEL
print("\n4. Merakit Model TGN...")
memory_dim = 100
time_dim   = 100
num_nodes = max(data.src.max(), data.dst.max()).item() + 1
# SUDAH BENAR: Menyiapkan ruang kosong (buffer) untuk node masa depan
num_nodes_buffer = num_nodes + 100000
msg_dim    = data.msg.size(1)

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

msg_module = LearnedMessageModule(msg_dim, memory_dim, time_dim)

memory = TGNMemory(
    num_nodes=num_nodes_buffer,
    raw_msg_dim=msg_dim,
    memory_dim=memory_dim,
    time_dim=time_dim,
    message_module=msg_module,
    aggregator_module=LastAggregator(),
).to(device)

class TimeEncoder(torch.nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.lin = Linear(1, out_channels)
    def forward(self, t):
        return self.lin(t.float().unsqueeze(-1)).cos()

time_encoder = TimeEncoder(out_channels=time_dim).to(device)

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

neighbor_loader = LastNeighborLoader(num_nodes_buffer, size=15, device=device)

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

all_params = (
    set(memory.parameters())       |
    set(gnn.parameters())          |
    set(time_encoder.parameters()) |
    set(link_pred.parameters())
)
optimizer = torch.optim.Adam(all_params, lr=0.0001)

# 5. FUNGSI: AMBIL EMBEDDING
def get_node_embeddings(src, dst):
    all_nodes              = torch.cat([src, dst]).unique()
    n_id, edge_index, e_id = neighbor_loader(all_nodes)
    z, last_update         = memory(n_id)

    if edge_index.numel() > 0:
        z = gnn(z, last_update, edge_index,
                data.t[e_id].float(),
                data.msg[e_id])

    assoc[n_id] = torch.arange(n_id.size(0), device=device)
    return z[assoc[src]], z[assoc[dst]]

# 6. WARM-UP STATE DARI TRAIN DATA
@torch.no_grad()
def warmup_state_from_train():
    memory.eval()
    gnn.eval()
    link_pred.eval()
    memory.reset_state()
    neighbor_loader.reset_state()

    for batch in warmup_loader:
        _ = get_node_embeddings(batch.src, batch.dst)
        neighbor_loader.insert(batch.src, batch.dst)
        memory.update_state(batch.src, batch.dst, batch.t, batch.msg)

# 7. FUNGSI: EVALUASI
@torch.no_grad()
def evaluate(loader, loader_name=""):
    memory.eval()
    gnn.eval()
    link_pred.eval()

    all_probs, all_labels = [], []

    for batch in loader:
        z_src, z_dst = get_node_embeddings(batch.src, batch.dst)
        prob = torch.sigmoid(link_pred(z_src, z_dst).squeeze())

        all_probs.append(prob.cpu())
        all_labels.append(batch.y.cpu())

        neighbor_loader.insert(batch.src, batch.dst)
        memory.update_state(batch.src, batch.dst, batch.t, batch.msg)

    all_probs  = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()

    if all_labels.sum() == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.5

    prec_arr, rec_arr, thresholds = precision_recall_curve(all_labels, all_probs)
    f1_arr      = 2 * prec_arr * rec_arr / (prec_arr + rec_arr + 1e-8)
    best_idx    = f1_arr[:-1].argmax()
    best_thresh = float(thresholds[best_idx])

    preds = (all_probs >= best_thresh).astype(float)
    tp = ((preds == 1) & (all_labels == 1)).sum()
    fp = ((preds == 1) & (all_labels == 0)).sum()
    fn = ((preds == 0) & (all_labels == 1)).sum()

    prec   = tp / (tp + fp + 1e-8)
    rec    = tp / (tp + fn + 1e-8)
    f1     = 2 * prec * rec / (prec + rec + 1e-8)
    auc    = roc_auc_score(all_labels, all_probs)
    pr_auc = average_precision_score(all_labels, all_probs)

    return prec, rec, f1, auc, pr_auc, best_thresh

# 8. TRAINING LOOP (Diperbaiki: Tanpa Negative Sampling)
print("\n8. Memulai Training (5 Epoch)...")
best_val_f1 = 0.0

for epoch in range(5):
    memory.train()
    gnn.train()
    link_pred.train()
    memory.reset_state()
    neighbor_loader.reset_state()
    total_loss = 0

    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        memory.detach()

        # ---- AMBIL EMBEDDING & PREDIKSI KONEKSI ASLI ----
        z_src, z_dst = get_node_embeddings(batch.src, batch.dst)
        pred = link_pred(z_src, z_dst).squeeze()

        # ---- LOSS KLASIFIKASI (Normal vs Serangan) ----
        # Hanya menggunakan label asli untuk mendeteksi anomali
        loss = criterion(pred, batch.y.float())

        loss.backward()
        optimizer.step()

        # Update state SETELAH backward
        neighbor_loader.insert(batch.src, batch.dst)
        memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
        total_loss += loss.item()

        if (i + 1) % 500 == 0:
            print(f"  Epoch {epoch+1} | Batch {i+1}/{len(train_loader)} "
                  f"| Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    print(f"\n=== Epoch {epoch+1} Selesai | Avg Loss: {avg_loss:.4f} ===")

    print("   [WARMUP] Mengisi state dari train data...")
    warmup_state_from_train()
    prec, rec, f1, auc, pr_auc, thresh = evaluate(val_loader, "VAL")
    print(f"   [VAL] Precision: {prec:.4f} | Recall: {rec:.4f} | "
          f"F1: {f1:.4f} | AUC-ROC: {auc:.4f} | "
          f"PR-AUC: {pr_auc:.4f} | Threshold: {thresh:.3f}")

    if f1 > best_val_f1:
        best_val_f1 = f1
        torch.save(memory.state_dict(),       './models/tgn_memory_best.pth')
        torch.save(gnn.state_dict(),          './models/tgn_gnn_best.pth')
        torch.save(time_encoder.state_dict(), './models/tgn_timeenc_best.pth')
        torch.save(link_pred.state_dict(),    './models/tgn_predictor_best.pth')
        # FIX: Simpan state optimizer untuk Continuous Training!
        torch.save(optimizer.state_dict(),    './models/tgn_optimizer_best.pth')
        print(f"   ✅ Model terbaik disimpan! (Val F1: {best_val_f1:.4f})")

# 9. EVALUASI FINAL DI TEST SET
print("\n9. Memuat model terbaik dan evaluasi di Test Set...")
memory.load_state_dict(torch.load('./models/tgn_memory_best.pth'))
gnn.load_state_dict(torch.load('./models/tgn_gnn_best.pth'))
time_encoder.load_state_dict(torch.load('./models/tgn_timeenc_best.pth'))
link_pred.load_state_dict(torch.load('./models/tgn_predictor_best.pth'))

print("   [WARMUP] Mengisi state: train...")
warmup_state_from_train()
print("   [WARMUP] Melanjutkan state: val...")
with torch.no_grad():
    for batch in val_loader:
        _ = get_node_embeddings(batch.src, batch.dst)
        neighbor_loader.insert(batch.src, batch.dst)
        memory.update_state(batch.src, batch.dst, batch.t, batch.msg)

prec, rec, f1, auc, pr_auc, thresh = evaluate(test_loader, "TEST")
print(f"\n{'='*60}")
print(f"  [TEST FINAL] Precision : {prec:.4f}")
print(f"  [TEST FINAL] Recall    : {rec:.4f}")
print(f"  [TEST FINAL] F1-Score  : {f1:.4f}")
print(f"  [TEST FINAL] AUC-ROC   : {auc:.4f}")
print(f"  [TEST FINAL] PR-AUC    : {pr_auc:.4f}  ← metrik utama untuk imbalanced")
print(f"  [TEST FINAL] Threshold : {thresh:.3f}")
print(f"{'='*60}")
print("\nTraining selesai. Semua model tersimpan di ./models/")