import pandas as pd
import gzip
import gc
import json  # === TAMBAHAN: untuk menyimpan metadata sampling ===

# ==========================================
# KONFIGURASI FILE
# ==========================================
AUTH_FILE = '/home/adriel/Desktop/Coding/LogNet/Dataset RM/Pick/auth.txt.gz'
REDTEAM_FILE = '/home/adriel/Desktop/Coding/LogNet/Dataset RM/Pick/redteam.txt.gz'
OUTPUT_FILE = 'auth_sampled_balanced.parquet'
METADATA_FILE = 'sampling_metadata.json'
CHUNK_SIZE = 2000000
NORMAL_SAMPLE_FRAC = 0.01

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

total_normal_seen = 0
total_attack_seen = 0
total_normal_kept = 0

for chunk in pd.read_csv(AUTH_FILE, header=None, names=auth_cols, chunksize=CHUNK_SIZE):
    chunk_signatures = list(zip(chunk['time'], chunk['src_user'], chunk['src_comp'], chunk['dst_comp']))
    chunk['label'] = [1 if sig in redteam_signatures else 0 for sig in chunk_signatures]

    # EKSTRAKSI FITUR SAKTI
    chunk['is_success'] = (chunk['success'] == 'Success').astype(int)
    chunk['is_machine'] = chunk['src_user'].astype(str).str.contains('$', regex=False).astype(int)
    chunk['logon_type'] = pd.to_numeric(chunk['logon_type'], errors='coerce').fillna(-1).astype(float)

    attack_data = chunk[chunk['label'] == 1]
    normal_data = chunk[chunk['label'] == 0]

    total_normal_seen += len(normal_data)
    total_attack_seen += len(attack_data)

    # Undersampling Normal
    sampled_normal = normal_data.sample(frac=NORMAL_SAMPLE_FRAC, random_state=42)

    total_normal_kept += len(sampled_normal)

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

actual_normal_sampling_fraction = (
    total_normal_kept / total_normal_seen if total_normal_seen > 0 else NORMAL_SAMPLE_FRAC
)

metadata = {
    "total_normal_seen_di_auth_asli": int(total_normal_seen),
    "total_attack_seen_di_auth_asli": int(total_attack_seen),
    "total_normal_disimpan_setelah_undersampling": int(total_normal_kept),
    "target_sampling_fraction": NORMAL_SAMPLE_FRAC,
    "actual_sampling_fraction": actual_normal_sampling_fraction,
    "catatan": (
        "actual_sampling_fraction dipakai di Tahap 5 untuk mengoreksi Confusion Matrix "
        "(inverse probability weighting) supaya angka precision/FP yang dilaporkan "
        "mencerminkan estimasi proporsi kelas ASLI di dunia nyata, bukan proporsi "
        "hasil undersampling yang dipakai untuk training."
    ),
}

with open(METADATA_FILE, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n[METADATA] Total normal asli terlihat  : {total_normal_seen:,}")
print(f"[METADATA] Total normal disimpan (~1%)  : {total_normal_kept:,}")
print(f"[METADATA] Fraksi sampling aktual        : {actual_normal_sampling_fraction:.6f}")
print(f"[METADATA] Disimpan di                   : {METADATA_FILE}")