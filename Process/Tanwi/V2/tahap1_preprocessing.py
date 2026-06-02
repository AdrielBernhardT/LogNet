import pandas as pd
import gzip
import gc

# ==========================================
# KONFIGURASI FILE
# ==========================================
AUTH_FILE = 'Resources/auth.txt.gz'
REDTEAM_FILE = 'Resources/redteam.txt.gz'
OUTPUT_FILE = 'auth_sampled_balanced.parquet'
CHUNK_SIZE = 2000000

print("1. Membaca data Redteam (Kunci Jawaban)...")
redteam_cols = ['time', 'user_domain', 'src_comp', 'dst_comp']
df_redteam = pd.read_csv(REDTEAM_FILE, header=None, names=redteam_cols)

redteam_signatures = set(
    zip(df_redteam['time'], df_redteam['user_domain'], df_redteam['src_comp'], df_redteam['dst_comp'])
)
del df_redteam
gc.collect()

print(f"Total signature serangan unik: {len(redteam_signatures)}")
print("\n2. Memproses Log Auth (Mode Validasi Ketat)...")

auth_cols = ['time', 'src_user', 'dst_user', 'src_comp', 'dst_comp', 'auth_type', 'logon_type', 'orientation', 'success']

chunks_processed = 0
all_sampled_chunks = []

for chunk in pd.read_csv(AUTH_FILE, header=None, names=auth_cols, chunksize=CHUNK_SIZE):
    chunk_signatures = list(zip(chunk['time'], chunk['src_user'], chunk['src_comp'], chunk['dst_comp']))
    chunk['label'] = [1 if sig in redteam_signatures else 0 for sig in chunk_signatures]
    
    # EKSTRAKSI FITUR SAKTI
    chunk['is_success'] = (chunk['success'] == 'Success').astype(int)
    chunk['is_machine'] = chunk['src_user'].astype(str).str.contains('$', regex=False).astype(int)
    chunk['logon_type'] = pd.to_numeric(chunk['logon_type'], errors='coerce').fillna(-1).astype(float)
    
    attack_data = chunk[chunk['label'] == 1]
    normal_data = chunk[chunk['label'] == 0]
    
    # Undersampling Normal (1% Tetap Dipertahankan)
    sampled_normal = normal_data.sample(frac=0.01, random_state=42)
    
    combined_chunk = pd.concat([attack_data, sampled_normal])
    combined_chunk = combined_chunk[['time', 'src_user', 'dst_comp', 'is_success', 'is_machine', 'logon_type', 'label']]
    all_sampled_chunks.append(combined_chunk)
    
    chunks_processed += 1
    if chunks_processed % 5 == 0:
        print(f"Selesai memproses chunk ke-{chunks_processed} ({chunks_processed * 2} Juta baris)")
    
    del chunk, attack_data, normal_data, sampled_normal, combined_chunk
    gc.collect()

print("\n3. Menyimpan hasil akhir...")
final_df = pd.concat(all_sampled_chunks, ignore_index=True)

print("\n=== RINGKASAN DATASET AUTH ===")
print(f"Total Serangan (Label 1) : {len(final_df[final_df['label'] == 1])}")
print(f"Total Mesin              : {len(final_df[final_df['is_machine'] == 1])}")

final_df.to_parquet(OUTPUT_FILE, index=False)
print(f"File berhasil disimpan di: {OUTPUT_FILE}")