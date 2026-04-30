import torch
from torch.nn import Linear
from torch_geometric.data import TemporalData
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn import TGNMemory
from torch_geometric.nn.models.tgn import IdentityMessage, LastAggregator

print("1. Menyiapkan Perangkat dan Data...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Menggunakan Device: {device}")

data_dict = torch.load('tgn_dataset.pt')
data = TemporalData(
    src=data_dict['src'], dst=data_dict['dst'], t=data_dict['t'],
    msg=data_dict['msg'], y=data_dict['y']
).to(device)

train_loader = TemporalDataLoader(data, batch_size=2000)

jumlah_normal = (data.y == 0).sum().item()
jumlah_serangan = (data.y == 1).sum().item()
pos_weight_value = torch.tensor([jumlah_normal / jumlah_serangan], device=device)
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_value)

print("\n3. Merakit Model TGN...")
memory_dim = 100
time_dim = 100
num_nodes = max(data.src.max(), data.dst.max()).item() + 1

memory = TGNMemory(
    num_nodes=num_nodes, 
    raw_msg_dim=data.msg.size(1), # Sekarang otomatis menyesuaikan jadi 3
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

print("\n4. Memulai Training (3 EPOCH)...")
memory.train()
link_pred.train()

for epoch in range(3): 
    memory.reset_state() 
    total_loss = 0
    
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        batch = batch.to(device)
        
        memory.detach() # Mencegah Error Autograd
        
        z_src, _ = memory(batch.src)
        z_dst, _ = memory(batch.dst)
        
        pred = link_pred(z_src, z_dst).squeeze()
        loss = criterion(pred, batch.y)
        loss.backward()
        optimizer.step()
        
        memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
        total_loss += loss.item()
        
        if (i+1) % 500 == 0:
            print(f"Epoch {epoch+1} | Batch {i+1} | Loss Sementara: {loss.item():.4f}")

    print(f"=== Selesai Epoch {epoch+1} | Rata-rata Loss: {total_loss / len(train_loader):.4f} ===")

torch.save(memory.state_dict(), 'tgn_memory_model.pth')
torch.save(link_pred.state_dict(), 'tgn_predictor_model.pth')
print("Model Sakti berhasil disimpan!")