import pandas as pd
import torch
import gc

GRAPH_EDGES_FILE = 'graph_edges.parquet'
OUTPUT_PYTORCH_FILE = 'tgn_dataset.pt'

print("1. Membaca Graf dari Parquet...")
df = pd.read_parquet(GRAPH_EDGES_FILE)

print("2. Mengubah Data Menjadi PyTorch Tensors...")
src = torch.tensor(df['src_id'].values, dtype=torch.long)
dst = torch.tensor(df['dst_id'].values, dtype=torch.long)
t = torch.tensor(df['time'].values, dtype=torch.long)
y = torch.tensor(df['label'].values, dtype=torch.float)

print("3. Membuat Edge Features (V4.1 - One-Hot Encoding)...")
edge_types = df['edge_type'].values
logon_types = df['logon_type'].values

msg = torch.zeros((len(df), 8), dtype=torch.float)

msg[edge_types == 0, 0] = 1.0  
msg[edge_types == 1, 1] = 1.0  
msg[:, 2] = torch.tensor(df['is_success'].values, dtype=torch.float) 
msg[:, 3] = torch.tensor(df['is_machine'].values, dtype=torch.float) 

# One-Hot Encoding untuk Logon Type
msg[logon_types == 2.0, 4] = 1.0   
msg[logon_types == 3.0, 5] = 1.0   
msg[logon_types == 10.0, 6] = 1.0  
msg[(logon_types != 2.0) & (logon_types != 3.0) & (logon_types != 10.0), 7] = 1.0 

del df, edge_types, logon_types
gc.collect()

print("4. Menyimpan PyTorch Dataset...")
torch.save({'src': src, 'dst': dst, 't': t, 'msg': msg, 'y': y}, OUTPUT_PYTORCH_FILE)

print("\n=== RINGKASAN TENSOR BARU ===")
print(f"Shape MSG  : {msg.shape} -> (8 Fitur Utama)")
print(f"File berhasil disimpan di: {OUTPUT_PYTORCH_FILE}")