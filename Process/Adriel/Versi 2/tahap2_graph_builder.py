import pandas as pd
import gc
import json
import os

# ==========================================
# KONFIGURASI FILE
# ==========================================
AUTH_PARQUET = '../../parsed/auth_sampled_balanced.parquet' 
FLOWS_FILE   = '../../Dataset RM/pick/flows.txt.gz'
REDTEAM_FILE = '../../Dataset RM/pick/redteam.txt.gz'
OUT_EDGES    = '../../graphs/graph_edges.parquet'
OUT_NODEMAP  = '../../graphs/node_map.json'

os.makedirs(os.path.dirname(OUT_EDGES), exist_ok=True)

print("1. Memuat data Auth...")
df_auth = pd.read_parquet(AUTH_PARQUET)

df_auth_graph = pd.DataFrame({
    'time'       : df_auth['time'],
    'src_node'   : df_auth['src_user'],
    'dst_node'   : df_auth['dst_comp'],
    'edge_type'  : 0,                        # 0 = auth event (penanda tipe edge)
    'auth_type'  : df_auth['auth_type'],      # FIX: nilai asli auth_type (NTLM, Kerberos, dll.)
    'is_success' : df_auth['is_success'],
    'is_machine' : df_auth['is_machine'],
    'logon_type' : df_auth['logon_type'],
    'label'      : df_auth['label']
})
del df_auth
gc.collect()

print("2. Memuat sampel data Flows...")
flow_cols = [
    'time', 'duration', 'src_comp', 'src_port',
    'dst_comp', 'dst_port', 'protocol', 'packet_count', 'byte_count'
]
df_flow_raw = pd.read_csv(
    FLOWS_FILE, header=None, names=flow_cols, nrows=500_000
)
 
df_flow_graph = pd.DataFrame({
    'time'       : df_flow_raw['time'],
    'src_node'   : df_flow_raw['src_comp'],
    'dst_node'   : df_flow_raw['dst_comp'],
    'edge_type'  : 1,          # 1 = flow event
    'auth_type'  : 'N/A',      # Flow tidak memiliki auth_type
    'is_success' : 1,
    'is_machine' : 1,          # Flow = komunikasi mesin ke mesin
    'logon_type' : -1,         # Flow tidak memiliki logon type
    'label'      : 0
})
del df_flow_raw
gc.collect()

print("3. Menggabungkan Auth dan Flows...")
df_graph = pd.concat([df_auth_graph, df_flow_graph], ignore_index=True)
df_graph = df_graph.sort_values('time').reset_index(drop=True)\

print("4. Normalisasi Timestamp...")
t_min = df_graph['time'].min()
t_max = df_graph['time'].max()
t_range = t_max - t_min

df_graph['time_delta'] = (df_graph['time'] - t_min).astype(float)
df_graph['time_norm']  = df_graph['time_delta'] / t_range  # [0, 1]
 
print(f"   Rentang waktu dataset : {t_min} s → {t_max} s")
print(f"   Durasi total          : {t_range:,} detik ({t_range/86400:.1f} hari)")
print(f"   time_delta range      : 0.0 → {df_graph['time_delta'].max():.0f}")
print(f"   time_norm range       : 0.0 → {df_graph['time_norm'].max():.4f}")

print("5. Membuat Node Mapping...")
semua_node = sorted(
    pd.concat([df_graph['src_node'], df_graph['dst_node']]).dropna().unique()
)
node_map = {node_name: i for i, node_name in enumerate(semua_node)}
 
with open(OUT_NODEMAP, 'w') as f:
    json.dump(node_map, f)
print(f"   Total node unik: {len(node_map):,}")

print("6. Menerapkan Node Mapping ke Graf...")
df_graph['src_id'] = df_graph['src_node'].map(node_map)
df_graph['dst_id'] = df_graph['dst_node'].map(node_map)
df_graph = df_graph.drop(columns=['src_node', 'dst_node'])
 
# Susun kolom final — urutan sesuai standar input TGN:
# [timestamp, src, dst, edge_features..., label]
df_graph = df_graph[[
    'time', 'time_delta', 'time_norm',   # Tahap 2: raw + normalized timestamp
    'src_id', 'dst_id',                   # Tahap 1: node identifiers
    'edge_type', 'auth_type',             # Tahap 1: edge type + FIX auth_type
    'is_success', 'is_machine',           # Tahap 1: edge features
    'logon_type',                         # Tahap 1: edge features
    'label'                               # Ground truth
]]

print("7. Menyimpan struktur Graf Akhir...")
df_graph.to_parquet(OUT_EDGES, index=False)
 
print("\n=== RINGKASAN GRAPH ===")
print(f"Total edge           : {len(df_graph):,}")
print(f"Total node unik      : {len(node_map):,}")
print(f"Edge auth (type=0)   : {(df_graph['edge_type']==0).sum():,}")
print(f"Edge flow (type=1)   : {(df_graph['edge_type']==1).sum():,}")
print(f"Edge berlabel attack : {df_graph['label'].sum():,}")
print(f"\nFile disimpan di: {OUT_EDGES}")
print("Selesai! Lanjut ke Tahap 3.")