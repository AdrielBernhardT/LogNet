import pandas as pd
import json
import gc

AUTH_PARQUET = 'auth_sampled_balanced.parquet'
FLOWS_FILE = 'Resources/flows.txt.gz'
OUT_EDGES = 'graph_edges.parquet'
OUT_NODEMAP = 'node_map.json'

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
del df_auth
gc.collect()

print("2. Memuat sampel data Flows...")
flow_cols = ['time', 'duration', 'src_comp', 'src_port', 'dst_comp', 'dst_port', 'protocol', 'packet_count', 'byte_count']
df_flow_raw = pd.read_csv(FLOWS_FILE, header=None, names=flow_cols, nrows=500000)

df_flow_graph = pd.DataFrame({
    'time': df_flow_raw['time'],
    'src_node': df_flow_raw['src_comp'],
    'dst_node': df_flow_raw['dst_comp'],
    'edge_type': 1,                   
    'is_success': 1,                  
    'is_machine': 1,   # Flow jaringan mayoritas komunikasi mesin ke mesin               
    'logon_type': -1,  # Flow tidak memiliki tipe logon
    'label': 0                        
})
del df_flow_raw
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

# Susun kolom V4
df_graph = df_graph[['time', 'src_id', 'dst_id', 'edge_type', 'is_success', 'is_machine', 'logon_type', 'label']]

print("6. Menyimpan struktur Graf Akhir...")
df_graph.to_parquet(OUT_EDGES, index=False)
print("Selesai! Lanjut ke Tahap 3.")