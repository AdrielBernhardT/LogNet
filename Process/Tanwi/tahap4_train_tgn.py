import torch
from torch.nn import Linear
from torch_geometric.data import TemporalData
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn import TGNMemory
from torch_geometric.nn.models.tgn import IdentityMessage, LastAggregator

# ==========================================
# 1. PERSIAPAN PERANGKAT & DATA
# ==========================================
print("1. Menyiapkan Perangkat dan Data...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Menggunakan Device: {device}")

data_dict = torch.load('tgn_dataset.pt')
data = TemporalData(
    src=data_dict['src'], dst=data_dict['dst'], t=data_dict['t'],
    msg=data_dict['msg'], y=data_dict['y']
)

# --- CHRONOLOGICAL SPLIT (50% TRAIN) ---
total_rows = data.msg.size(0)
train_size = int(0.50 * total_rows)
print(f"\n[INFO] Total Baris Data: {total_rows}")
print(f"[INFO] Memotong {train_size} baris (50% masa lalu) murni untuk Training!")

# OPTIMASI 1: Tambahkan .clone() agar memori training terputus dari file raksasa 11 juta baris
train_data = TemporalData(
    src=data.src[:train_size].clone(), 
    dst=data.dst[:train_size].clone(), 
    t=data.t[:train_size].clone(),
    msg=data.msg[:train_size].clone(), 
    y=data.y[:train_size].clone()
).to(device)

train_loader = TemporalDataLoader(train_data, batch_size=2000)

# ==========================================
# 2. WEIGHTED LOSS (MENGATASI IMBALANCE DI DATA TRAIN)
# ==========================================
jumlah_normal = (train_data.y == 0).sum().item()
jumlah_serangan = (train_data.y == 1).sum().item()

# Cegah error pembagian nol jika kebetulan serangan numpuk di akhir
if jumlah_serangan == 0: jumlah_serangan = 1 

# OPTIMASI 2: Tambahkan dtype=torch.float untuk kestabilan kalkulasi Loss
pos_weight_value = torch.tensor([jumlah_normal / jumlah_serangan], dtype=torch.float, device=device)
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_value)

print(f"-> Bobot Hukuman untuk Serangan: {pos_weight_value.item():.2f}x lipat lebih berat")

# ==========================================
# 3. MERAKIT MODEL TGN
# ==========================================
print("\n3. Merakit Model TGN...")
memory_dim = 100
time_dim = 100
num_nodes = max(data.src.max(), data.dst.max()).item() + 1

# raw_msg_dim otomatis membaca kolom fitur
memory = TGNMemory(
    num_nodes=num_nodes, 
    raw_msg_dim=data.msg.size(1), 
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
optimizer = torch.optim.Adam(set(memory.parameters()) | set(link_pred.parameters()), lr=0.0001)

# ==========================================
# 4. TRAINING LOOP (5 EPOCH)
# ==========================================
print("\n4. Memulai Training (5 EPOCH - Mode 50% Chronological Split)...")
memory.train()
link_pred.train()

for epoch in range(5): 
    memory.reset_state() 
    total_loss = 0
    
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        batch = batch.to(device)
        
        # Mencegah error backward graph
        memory.detach() 
        
        z_src, _ = memory(batch.src)
        z_dst, _ = memory(batch.dst)
        
        pred = link_pred(z_src, z_dst).squeeze()
        loss = criterion(pred, batch.y)
        loss.backward()
        optimizer.step()
        
        memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
        total_loss += loss.item()
        
        # OPTIMASI 3: Update tampilan terminal menjadi per 100 batch supaya kamu tahu prosesnya berjalan
        if (i+1) % 100 == 0:
            print(f"Epoch {epoch+1} | Batch {i+1} | Loss Sementara: {loss.item():.4f}")

    print(f"=== Selesai Epoch {epoch+1} | Rata-rata Loss: {total_loss / len(train_loader):.4f} ===")

torch.save(memory.state_dict(), 'tgn_memory_model.pth')
torch.save(link_pred.state_dict(), 'tgn_predictor_model.pth')
print("Model Detektif Forensik berhasil disimpan!")