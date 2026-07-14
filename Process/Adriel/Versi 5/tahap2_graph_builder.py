import pandas as pd
import json
import gc
import os  # === TAMBAHAN: untuk cek keberadaan file metadata dari Tahap 1 ===

AUTH_PARQUET = 'auth_sampled_balanced.parquet'
FLOWS_FILE = '/home/adriel/Desktop/Coding/LogNet/Dataset RM/Pick/flows.txt.gz'
OUT_EDGES = 'graph_edges.parquet'
OUT_NODEMAP = 'node_map.json'
METADATA_FILE = 'sampling_metadata.json'  # === TAMBAHAN: file yang sama dengan Tahap 1, akan ditambah isinya ===
FLOW_SAMPLE_FRAC = 0.005  # === TAMBAHAN: ditarik jadi konstanta bernama (sebelumnya angka literal di .sample()) ===

print("1. Memuat data Auth...")
df_auth = pd.read_parquet(AUTH_PARQUET)

df_auth_graph = pd.DataFrame({
    'time': df_auth['time'],
    'src_node': df_auth['src_user'],
    'dst_node': df_auth['dst_comp'],
    'edge_type': 0,
    'is_success': df_auth['is_success'],
    'is_machine': df_auth['is_machine'],
    'logon_type': df_auth['logon_type'],
    'label': df_auth['label']
})

# Ambil min dan max time dari Auth untuk membatasi rentang waktu secara konsisten
min_time = df_auth['time'].min()
max_time = df_auth['time'].max()

del df_auth
gc.collect()

print("2. Memuat dan Memotong data Flows (Mencegah Kebocoran Temporal)...")
flow_cols = ['time', 'duration', 'src_comp', 'src_port', 'dst_comp', 'dst_port', 'protocol', 'packet_count', 'byte_count']

chunk_list = []

total_flow_seen = 0
total_flow_kept = 0

# Memproses secara bertahap
for flow_chunk in pd.read_csv(FLOWS_FILE, header=None, names=flow_cols, chunksize=5000000):
    valid_flow = flow_chunk[(flow_chunk['time'] >= min_time) & (flow_chunk['time'] <= max_time)]

    if not valid_flow.empty:
        total_flow_seen += len(valid_flow)

        sampled_flow = valid_flow.sample(frac=FLOW_SAMPLE_FRAC, random_state=42)

        total_flow_kept += len(sampled_flow)

        chunk_list.append(sampled_flow)

df_flow_raw = pd.concat(chunk_list, ignore_index=True)
print(f"Total baris Flow yang lolos kriteria temporal: {len(df_flow_raw)}")

df_flow_graph = pd.DataFrame({
    'time': df_flow_raw['time'],
    'src_node': df_flow_raw['src_comp'],
    'dst_node': df_flow_raw['dst_comp'],
    'edge_type': 1,
    'is_success': 1,
    'is_machine': 1,
    'logon_type': -1,
    'label': 0
})
del df_flow_raw, chunk_list
gc.collect()

print("3. Menggabungkan Auth dan Flows...")
df_graph = pd.concat([df_auth_graph, df_flow_graph], ignore_index=True)
df_graph = df_graph.sort_values('time').reset_index(drop=True)

print("4. Membuat Node Mapping...")
semua_node = pd.concat([df_graph['src_node'], df_graph['dst_node']]).unique()
node_map = {node_name: i for i, node_name in enumerate(semua_node)}

with open(OUT_NODEMAP, 'w') as f:
    json.dump(node_map, f)

print("5. Menerapkan Node Mapping ke Graf...")
df_graph['src_id'] = df_graph['src_node'].map(node_map)
df_graph['dst_id'] = df_graph['dst_node'].map(node_map)
df_graph = df_graph.drop(columns=['src_node', 'dst_node'])

df_graph = df_graph[['time', 'src_id', 'dst_id', 'edge_type', 'is_success', 'is_machine', 'logon_type', 'label']]

print("6. Menyimpan struktur Graf Akhir...")
df_graph.to_parquet(OUT_EDGES, index=False)

actual_flow_sampling_fraction = (
    total_flow_kept / total_flow_seen if total_flow_seen > 0 else FLOW_SAMPLE_FRAC
)

if os.path.exists(METADATA_FILE):
    with open(METADATA_FILE) as f:
        metadata = json.load(f)
else:
    metadata = {}

metadata.update({
    "total_flow_seen_dalam_rentang_waktu_auth": int(total_flow_seen),
    "total_flow_disimpan_setelah_sampling": int(total_flow_kept),
    "target_flow_sampling_fraction": FLOW_SAMPLE_FRAC,
    "actual_flow_sampling_fraction": actual_flow_sampling_fraction,
})

with open(METADATA_FILE, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n[METADATA] Flow terlihat (dalam rentang waktu) : {total_flow_seen:,}")
print(f"[METADATA] Flow disimpan (sampling)             : {total_flow_kept:,}")
print(f"[METADATA] Fraksi sampling aktual flow          : {actual_flow_sampling_fraction:.6f}")
print(f"[METADATA] Ditambahkan ke                       : {METADATA_FILE}")

print("Selesai! Lanjut ke Tahap 3.")