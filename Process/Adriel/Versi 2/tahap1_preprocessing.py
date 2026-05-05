import pandas as pd
import gc
import os

# ==========================================
# KONFIGURASI FILE
# ==========================================
AUTH_FILE    = '../../../Dataset RM/Pick/auth.txt.gz'
REDTEAM_FILE = '../../../Dataset RM/Pick/redteam.txt.gz'
OUTPUT_FILE  = './parsed/auth_sampled_balanced.parquet'
CHUNK_SIZE   = 2_000_000

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

print("1. Membaca data Redteam (Kunci Jawaban)...")
 
redteam_cols = ['time', 'user_domain', 'src_comp', 'dst_comp']
df_redteam = pd.read_csv(REDTEAM_FILE, header=None, names=redteam_cols)
 
redteam_signatures = set(
    zip(df_redteam['time'], df_redteam['user_domain'], df_redteam['src_comp'], df_redteam['dst_comp'])
)
del df_redteam
gc.collect()
 
print(f"   Total signature serangan unik: {len(redteam_signatures):,}")

print("\n2. Memproses Log Auth...")
 
auth_cols = [
    'time', 'src_user', 'dst_user', 'src_comp', 'dst_comp',
    'auth_type', 'logon_type', 'orientation', 'success'
]
 
chunks_processed    = 0
total_rows_read     = 0  # FIX 6: tracking baris aktual, bukan estimasi
all_sampled_chunks  = []

for chunk in pd.read_csv(AUTH_FILE, header=None, names=auth_cols, chunksize=CHUNK_SIZE):
    chunk_signatures = list(
        zip(chunk['time'], chunk['src_user'], chunk['src_comp'], chunk['dst_comp'])
    )
    
    chunk['label'] = [1 if sig in redteam_signatures else 0 for sig in chunk_signatures]

    chunk['is_success'] = (chunk['success'] == 'Success').astype(int)

    chunk['is_machine'] = chunk['src_user'].astype(str).str.endswith('$').astype(int)

    chunk['logon_type'] = (
        pd.to_numeric(chunk['logon_type'], errors='coerce')
        .fillna(-1)
        .astype(int)
    )

    attack_data = chunk[chunk['label'] == 1].copy()
    normal_data = chunk[chunk['label'] == 0].copy()

    if len(normal_data) > 0:
        sampled_normal = normal_data.sample(frac=0.01, random_state=42)
    else:
        sampled_normal = normal_data.copy()
 
    combined_chunk = pd.concat([attack_data, sampled_normal], ignore_index=True)

    combined_chunk = combined_chunk[[
        'time', 'src_user', 'src_comp', 'dst_comp',
        'is_success', 'is_machine', 'logon_type', 'label'
    ]]

    all_sampled_chunks.append(combined_chunk)

    total_rows_read += len(chunk)
    chunks_processed += 1
    print(
        f"   Chunk {chunks_processed:>3} selesai | "
        f"Dibaca: {total_rows_read:>12,} baris | "
        f"Attack: {len(attack_data):>5,} | "
        f"Normal sampled: {len(sampled_normal):>7,}"
    )
 
    del chunk, attack_data, normal_data, sampled_normal, combined_chunk
    gc.collect()

print("\n3. Menyimpan hasil akhir...")
if not all_sampled_chunks:
    print("ERROR: Tidak ada data yang berhasil diproses.")
    print("       Periksa path file:")
    print(f"       AUTH_FILE    = {AUTH_FILE}")
    print(f"       REDTEAM_FILE = {REDTEAM_FILE}")
else:
    final_df = pd.concat(all_sampled_chunks, ignore_index=True)

    total         = len(final_df)
    total_attack  = (final_df['label'] == 1).sum()
    total_normal  = (final_df['label'] == 0).sum()
    total_machine = (final_df['is_machine'] == 1).sum()
    total_human   = (final_df['is_machine'] == 0).sum()
 
    print("\n=== RINGKASAN DATASET BARU (V4) ===")
    print(f"Total baris              : {total:>12,}")
    print(f"--- Label ---")
    print(f"  Serangan  (label=1)    : {total_attack:>12,}  ({total_attack/total*100:.2f}%)")
    print(f"  Normal    (label=0)    : {total_normal:>12,}  ({total_normal/total*100:.2f}%)")
    print(f"--- Tipe Akun ---")
    print(f"  Mesin     (is_machine=1): {total_machine:>11,}  ({total_machine/total*100:.2f}%)")
    print(f"  Manusia   (is_machine=0): {total_human:>11,}  ({total_human/total*100:.2f}%)")
 
    final_df.to_parquet(OUTPUT_FILE, index=False)
    print(f"\nFile berhasil disimpan di: {OUTPUT_FILE}")