import pandas as pd
import torch
import gc
import os

GRAPH_EDGES_FILE = './graphs/graph_edges.parquet'
OUTPUT_PYTORCH_FILE = './datasets/tgn_dataset.pt'

os.makedirs(os.path.dirname(OUTPUT_PYTORCH_FILE), exist_ok=True)

# ==========================================
# 1. BACA DATA
# ==========================================
print("1. Membaca Graf dari Parquet...")
df = pd.read_parquet(GRAPH_EDGES_FILE)

print(f"\n   Kolom tersedia: {df.columns.tolist()}")
print(f"   Sample 5 baris:")
print(df.head())

# ==========================================
# 2. TENTUKAN KOLOM TIMESTAMP ASLI
#
# FIX: 't' harus berisi timestamp ASLI, bukan
# time_norm. TGN menggunakan 't' untuk menghitung
# delta waktu antar event (rel_t). Jika semua
# nilai t = 0 atau 1, semua event dianggap
# terjadi "bersamaan" — memory module tidak bisa
# belajar pola temporal apapun.
#
# Prioritas pencarian kolom timestamp asli:
#   1. 'time'       — detik sejak epoch (LANL)
#   2. 'timestamp'  — nama umum lainnya
#   3. 'time_unix'  — kadang dipakai di preprocessing
#   4. 'second'     — format LANL asli
#
# time_norm TETAP disimpan sebagai fitur di msg
# (kolom ke-8) agar informasi posisi relatif
# dalam dataset bisa dimanfaatkan model.
# ==========================================
TIMESTAMP_CANDIDATES = ['time', 'timestamp', 'time_unix', 'second', 'ts']
time_col = None
for col in TIMESTAMP_CANDIDATES:
    if col in df.columns:
        time_col = col
        break

if time_col is None:
    num_cols = df.select_dtypes(include='number').columns.tolist()
    raise ValueError(
        f"Tidak ditemukan kolom timestamp asli.\n"
        f"Kolom numerik yang tersedia: {num_cols}\n"
        f"Ganti TIMESTAMP_CANDIDATES di atas dengan nama kolom yang benar."
    )

print(f"\n   Menggunakan kolom timestamp: '{time_col}'")
print(f"   Range nilai: {df[time_col].min()} – {df[time_col].max()}")
print(f"   Unique values: {df[time_col].nunique():,}")

t_raw = df[time_col].values
if pd.api.types.is_float_dtype(t_raw):
    import numpy as np
    t_raw = np.floor(t_raw).astype('int64')

# ==========================================
# 3. BUAT TENSOR
# ==========================================
print("\n2. Mengubah Data Menjadi PyTorch Tensors...")
src = torch.tensor(df['src_id'].values,  dtype=torch.long)
dst = torch.tensor(df['dst_id'].values,  dtype=torch.long)
t   = torch.tensor(t_raw,                dtype=torch.long)
y   = torch.tensor(df['label'].values,   dtype=torch.float)

print(f"   t — min: {t.min().item()}, max: {t.max().item()}, "
      f"unique: {t.unique().numel():,} dari {t.numel():,} events")

# ==========================================
# 4. BUAT EDGE FEATURES (9 KOLOM)
#
# Kolom 0–7: sama seperti sebelumnya
# Kolom 8  : time_norm (posisi relatif dalam dataset)
#            Berguna sebagai fitur kontekstual di msg,
#            tapi TIDAK digunakan sebagai 't'.
# ==========================================
print("\n3. Membuat Edge Features (V4.2 - 9 kolom)...")
msg = torch.zeros((len(df), 9), dtype=torch.float)

# Kolom 0-1: edge type
msg[:, 0] = torch.tensor((df['edge_type'] == 0).values, dtype=torch.float)
msg[:, 1] = torch.tensor((df['edge_type'] == 1).values, dtype=torch.float)

# Kolom 2-3: fitur dasar
msg[:, 2] = torch.tensor(df['is_success'].values, dtype=torch.float)
msg[:, 3] = torch.tensor(df['is_machine'].values, dtype=torch.float)

# Kolom 4-7: logon type one-hot (Sesuaikan angka dengan dataset aslimu jika beda)
msg[:, 4] = torch.tensor((df['logon_type'] == 2).values, dtype=torch.float)
msg[:, 5] = torch.tensor((df['logon_type'] == 3).values, dtype=torch.float)
msg[:, 6] = torch.tensor((df['logon_type'] == 7).values, dtype=torch.float)
msg[:, 7] = torch.tensor((df['logon_type'] == 10).values, dtype=torch.float)

# Kolom 8: time_norm sebagai fitur
if 'time_norm' in df.columns:
    msg[:, 8] = torch.tensor(df['time_norm'].values, dtype=torch.float)

del df
gc.collect()

# ==========================================
# 5. SIMPAN
# ==========================================
print("\n4. Menyimpan PyTorch Dataset...")
torch.save({'src': src, 'dst': dst, 't': t, 'msg': msg, 'y': y},
           OUTPUT_PYTORCH_FILE)

print("\n=== RINGKASAN TENSOR ===")
print(f"src shape  : {src.shape}")
print(f"dst shape  : {dst.shape}")
print(f"t shape    : {t.shape}  (timestamp ASLI, bukan time_norm)")
print(f"msg shape  : {msg.shape}  (9 kolom: 2 edge_type + 2 dasar + 4 logon + 1 time_norm)")
print(f"y shape    : {y.shape}")
label_counts = y.unique(return_counts=True)
print(f"y distribusi: {dict(zip(label_counts[0].tolist(), label_counts[1].tolist()))}")
print(f"\nFile disimpan di: {OUTPUT_PYTORCH_FILE}")