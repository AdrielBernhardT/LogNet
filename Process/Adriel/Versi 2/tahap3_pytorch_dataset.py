import pandas as pd
import torch
import gc
import os

GRAPH_EDGES_FILE = './graphs/graph_edges.parquet'
OUTPUT_PYTORCH_FILE = './datasets/tgn_dataset.pt'

os.makedirs(os.path.dirname(OUTPUT_PYTORCH_FILE), exist_ok=True)

print("1. Membaca Graf dari Parquet...")
df = pd.read_parquet(GRAPH_EDGES_FILE)

print("2. Mengubah Data Menjadi PyTorch Tensors...")
src = torch.tensor(df['src_id'].values, dtype=torch.long)
dst = torch.tensor(df['dst_id'].values, dtype=torch.long)
t = torch.tensor(df['time_norm'].values, dtype=torch.float)
y = torch.tensor(df['label'].values, dtype=torch.float)

print("3. Membuat Edge Features (V4.1 - One-Hot Encoding)...")
edge_types = df['edge_type'].values
logon_types = df['logon_type'].values

auth_type_map = {'NTLM': 0, 'Kerberos': 1, 'N/A': 2}
auth_types = df['auth_type'].map(auth_type_map).fillna(3).values.astype(int)

# Tensor kita berevolusi jadi 8 Kolom!
msg = torch.zeros((len(df), 8), dtype=torch.float)

# Fitur Dasar
msg[edge_types == 0, 0] = 1.0  # Kolom 0: Is Auth
msg[edge_types == 1, 1] = 1.0  # Kolom 1: Is Flow
msg[:, 2] = torch.tensor(df['is_success'].values, dtype=torch.float) # Kolom 2: is_success
msg[:, 3] = torch.tensor(df['is_machine'].values, dtype=torch.float) # Kolom 3: is_machine

# Fitur One-Hot Encoding untuk Logon Type (Penyembuh Dosa ML)
msg[logon_types == 2, 4] = 1.0   # Kolom 4: Is Logon Type 2 (Interactive)
msg[logon_types == 3, 5] = 1.0   # Kolom 5: Is Logon Type 3 (Network)
msg[logon_types == 10, 6] = 1.0  # Kolom 6: Is Logon Type 10 (Remote)
# Logon type lainnya atau flow (-1) biarkan masuk ke Kolom 7 (Other)
msg[(logon_types != 2) & (logon_types != 3) & (logon_types != 10), 7] = 1.0 

msg[auth_types == 0, 8] = 1.0   # Kolom 8: Is NTLM
msg[auth_types == 1, 9] = 1.0   # Kolom 9: Is Kerberos
msg[auth_types >= 2, 10] = 1.0  # Kolom 10: Is Other/N/A

del df, edge_types, logon_types
gc.collect()

print("4. Menyimpan PyTorch Dataset...")
torch.save({'src': src, 'dst': dst, 't': t, 'msg': msg, 'y': y}, OUTPUT_PYTORCH_FILE)

print("\n=== RINGKASAN TENSOR BARU ===")
print(f"Shape MSG  : {msg.shape} -> (8 Kolom: Telah di-One-Hot Encode!)")
print(f"File berhasil disimpan di: {OUTPUT_PYTORCH_FILE}")