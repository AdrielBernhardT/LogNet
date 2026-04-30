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

print("3. Membuat Edge Features (Fitur Dukun Siber)...")
edge_types = df['edge_type'].values
is_success = df['is_success'].values

# Tensor kita sekarang ukurannya 3 kolom!
msg = torch.zeros((len(df), 3), dtype=torch.float)

# Mengisi kecerdasan fitur
msg[edge_types == 0, 0] = 1.0  # Kolom 0: Tanda kalau ini Auth
msg[edge_types == 1, 1] = 1.0  # Kolom 1: Tanda kalau ini Flow
msg[:, 2] = torch.tensor(is_success, dtype=torch.float) # Kolom 2: Status Success/Fail

del df, edge_types, is_success
gc.collect()

print("4. Menyimpan PyTorch Dataset...")
torch.save({'src': src, 'dst': dst, 't': t, 'msg': msg, 'y': y}, OUTPUT_PYTORCH_FILE)

print("\n=== RINGKASAN TENSOR BARU ===")
print(f"Shape MSG  : {msg.shape} -> (Makin Sakti dengan 3 Fitur!)")
print(f"File berhasil disimpan di: {OUTPUT_PYTORCH_FILE}")