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

# Membuat "Signature" serangan agar pencarian sangat cepat
redteam_signatures = set(
    zip(df_redteam['time'], df_redteam['user_domain'], df_redteam['src_comp'], df_redteam['dst_comp'])
)
del df_redteam
gc.collect()

print(f"Total signature serangan unik: {len(redteam_signatures)}")
print("\n2. Memproses Log Auth (Mencari jarum di tumpukan jerami)...")

# LANL Auth Format: time, src_user, dst_user, src_comp, dst_comp, auth_type, logon_type, orientation, success
auth_cols = ['time', 'src_user', 'dst_user', 'src_comp', 'dst_comp', 'auth_type', 'logon_type', 'orientation', 'success']

chunks_processed = 0
all_sampled_chunks = []

for chunk in pd.read_csv(AUTH_FILE, header=None, names=auth_cols, chunksize=CHUNK_SIZE):
    # Buat signature yang sama untuk log auth
    chunk_signatures = list(zip(chunk['time'], chunk['src_user'], chunk['src_comp'], chunk['dst_comp']))
    
    # Labeling seketika (O(1) lookup)
    chunk['label'] = [1 if sig in redteam_signatures else 0 for sig in chunk_signatures]
    
    # EKSTRAKSI FITUR SAKTI: 1 jika Success, 0 jika Fail/Lainnya
    chunk['is_success'] = (chunk['success'] == 'Success').astype(int)
    
    # Pisahkan Attack dan Normal
    attack_data = chunk[chunk['label'] == 1]
    normal_data = chunk[chunk['label'] == 0]
    
    # Undersampling ekstrim untuk Normal (Ambil 1% saja)
    sampled_normal = normal_data.sample(frac=0.01, random_state=42)
    
    # Gabungkan kembali dan simpan
    combined_chunk = pd.concat([attack_data, sampled_normal])
    
    # Kita hanya simpan kolom yang krusial untuk graf
    combined_chunk = combined_chunk[['time', 'src_user', 'dst_comp', 'is_success', 'label']]
    all_sampled_chunks.append(combined_chunk)
    
    chunks_processed += 1
    print(f"Selesai memproses chunk ke-{chunks_processed} ({(chunks_processed * 2)} Juta baris)")
    
    del chunk, attack_data, normal_data, sampled_normal, combined_chunk
    gc.collect()

print("\n3. Menyimpan hasil akhir...")
final_df = pd.concat(all_sampled_chunks, ignore_index=True)

print("\n=== RINGKASAN DATASET BARU (DUKUN SIBER) ===")
print(f"Total Serangan (Label 1) : {len(final_df[final_df['label'] == 1])}")
print(f"Total Normal (Label 0)   : {len(final_df[final_df['label'] == 0])}")
print(f"Total Login GAGAL        : {len(final_df[final_df['is_success'] == 0])}")

final_df.to_parquet(OUTPUT_FILE, index=False)
print(f"File berhasil disimpan di: {OUTPUT_FILE}")
