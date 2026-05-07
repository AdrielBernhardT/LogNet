import gc
import json
import os
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd

# CONFIGURATION
AUTH_FILE    = '../../../Dataset RM/Pick/auth.txt.gz'
REDTEAM_FILE = '../../../Dataset RM/Pick/redteam.txt.gz'
OUTPUT_DIR   = './parsed/hetero_graph/'
CHUNK_SIZE   = 2_000_000   
NEG_RATIO    = 10    

# Categorical vocabularies
AUTH_TYPE_MAP = defaultdict(lambda: -1, {
    'NTLM': 0, 'Kerberos': 1, 'Negotiate': 2,
    'MICROSOFT_AUTHENTICATION_PACKAGE_V1_0': 3,
})

ORIENTATION_MAP = defaultdict(lambda: -1, {
    'LogOn': 0, 'LogOff': 1, 'AuthMap': 2, 'TGS': 3, 'TGT': 4,
})

os.makedirs(OUTPUT_DIR, exist_ok=True)

# HELPERS
def shannon_entropy(series: pd.Series) -> float:
    """Normalised Shannon entropy over value counts of a Series."""
    counts = series.value_counts(normalize=True).values
    counts = counts[counts > 0]
    return float(-np.sum(counts * np.log2(counts + 1e-12)))


def balance_eval_split(
        df: pd.DataFrame, 
        indices: list, 
        neg_ratio: int = 10 
    ) -> pd.DataFrame:

    split = df.loc[indices]
    attacks = split[split['label'] == 1]
    normals = split[split['label'] == 0]
    n_sample = min(len(normals), max(len(attacks) * neg_ratio, 1))
    sampled_normals = normals.sample(n=n_sample, random_state=42)
    return pd.concat([attacks, sampled_normals]).sort_values('time').reset_index(drop=True)


# Load redteam signatures (evaluation keys only)
print("=" * 60)
print("STEP 1 · Loading redteam signatures (eval labels only)")
print("=" * 60)

redteam_cols = ['time', 'user_domain', 'src_comp', 'dst_comp']
df_redteam   = pd.read_csv(REDTEAM_FILE, header=None, names=redteam_cols)

# Signature: (time, src_user_or_domain, src_comp, dst_comp)  — matches auth cols
redteam_signatures: set = set(
    zip(df_redteam['time'],
        df_redteam['user_domain'],
        df_redteam['src_comp'],
        df_redteam['dst_comp'])
)
del df_redteam
gc.collect()

print(f"  Unique attack signatures : {len(redteam_signatures):,}\n")


# STEP 2 — Stream auth logs → build rich edge list
print("=" * 60)
print("STEP 2 · Streaming auth log → rich edges")
print("=" * 60)

auth_cols = [
    'time', 'src_user', 'dst_user', 'src_comp', 'dst_comp',
    'auth_type', 'logon_type', 'orientation', 'success',
]

all_edge_chunks: list[pd.DataFrame] = []
chunks_done   = 0
total_rows    = 0

for chunk in pd.read_csv(
    AUTH_FILE, header=None, names=auth_cols, chunksize=CHUNK_SIZE
):
    # ---- Labels (kept for later eval assignment; NOT used in train) ----
    sigs = list(zip(chunk['time'], chunk['src_user'],
                    chunk['src_comp'], chunk['dst_comp']))
    chunk['label'] = np.array(
        [1 if s in redteam_signatures else 0 for s in sigs], dtype=np.int8
    )

    # ---- Node-type flags ------------------------------------------------
    src_is_machine = chunk['src_user'].astype(str).str.startswith('C')
    dst_is_machine = chunk['dst_user'].astype(str).str.startswith('C')

    chunk['src_is_human'] = (~src_is_machine).astype(np.int8)
    chunk['dst_is_human'] = (~dst_is_machine).astype(np.int8)

    # ---- Edge type ------------------------------------------------------
    # 0 = user→comp  |  1 = comp→comp  |  2 = user→user or other
    chunk['edge_type'] = np.select(
        condlist=[
            (~src_is_machine) & chunk['dst_comp'].notna(),  # human auth to a machine
            src_is_machine    & chunk['dst_comp'].notna(),  # machine-to-machine
        ],
        choicelist=[0, 1],
        default=2,
    ).astype(np.int8)

    # ---- Edge features --------------------------------------------------
    chunk['is_success'] = (chunk['success'] == 'Success').astype(np.int8)

    chunk['auth_type_enc'] = (
        chunk['auth_type'].map(AUTH_TYPE_MAP).fillna(-1).astype(np.int8)
    )
    chunk['orientation_enc'] = (
        chunk['orientation'].map(ORIENTATION_MAP).fillna(-1).astype(np.int8)
    )
    chunk['logon_type_enc'] = (
        pd.to_numeric(chunk['logon_type'], errors='coerce')
        .fillna(-1).astype(np.int8)
    )

    # ---- Temporal buckets -----------------------------------------------
    chunk['time_bucket_hour'] = (chunk['time'] // 3600).astype(np.int32)
    chunk['time_bucket_day']  = (chunk['time'] // 86400).astype(np.int16)

    # ---- Keep only the columns needed for the graph ---------------------
    keep = [
        'time', 'time_bucket_hour', 'time_bucket_day',
        'src_user', 'dst_user', 'src_comp', 'dst_comp',
        'edge_type', 'is_success', 'src_is_human', 'dst_is_human',
        'auth_type_enc', 'orientation_enc', 'logon_type_enc',
        'label',
    ]
    all_edge_chunks.append(chunk[keep].copy())

    total_rows   += len(chunk)
    chunks_done  += 1
    attacks_here  = int(chunk['label'].sum())
    print(
        f"  chunk {chunks_done:>3}  |  rows read: {total_rows:>12,}"
        f"  |  attack edges: {attacks_here:>6,}"
    )

    del chunk
    gc.collect()


# ============================================================
# STEP 3 — Build node indices (heterogeneous)
# ============================================================

print("\n" + "=" * 60)
print("STEP 3 · Building heterogeneous node indices")
print("=" * 60)

edges_df = pd.concat(all_edge_chunks, ignore_index=True)
del all_edge_chunks
gc.collect()

# --- USER nodes ---
all_users = pd.unique(
    pd.concat([edges_df['src_user'].dropna(), edges_df['dst_user'].dropna()])
)
user_to_idx: dict = {u: int(i) for i, u in enumerate(all_users)}

# --- COMPUTER nodes ---
all_comps = pd.unique(
    pd.concat([edges_df['src_comp'].dropna(), edges_df['dst_comp'].dropna()])
)
comp_to_idx: dict = {c: int(i) for i, c in enumerate(all_comps)}

print(f"  Unique USER nodes     : {len(user_to_idx):>10,}")
print(f"  Unique COMPUTER nodes : {len(comp_to_idx):>10,}")

# Map to integer indices (−1 = unknown / NaN)
edges_df['src_user_idx'] = (
    edges_df['src_user'].map(user_to_idx).fillna(-1).astype(np.int32)
)
edges_df['dst_user_idx'] = (
    edges_df['dst_user'].map(user_to_idx).fillna(-1).astype(np.int32)
)
edges_df['src_comp_idx'] = (
    edges_df['src_comp'].map(comp_to_idx).fillna(-1).astype(np.int32)
)
edges_df['dst_comp_idx'] = (
    edges_df['dst_comp'].map(comp_to_idx).fillna(-1).astype(np.int32)
)


# ============================================================
# STEP 4 — Node-level aggregate features
# ============================================================

print("\n" + "=" * 60)
print("STEP 4 · Computing node-level aggregate features")
print("=" * 60)

def build_node_features(group_col: str, idx_map: dict,
                        node_type: str) -> pd.DataFrame:
    """
    Aggregate per-entity behavioral statistics across the full corpus.
    These become node feature vectors for the GNN.
    """
    grp = edges_df.groupby(group_col)
    feat = pd.DataFrame({'entity': list(idx_map.keys())})
    feat['entity_idx']  = feat['entity'].map(idx_map)
    feat['entity_type'] = node_type

    stats = grp.agg(
        total_events       = ('label',          'count'),
        attack_events      = ('label',          'sum'),
        success_rate       = ('is_success',     'mean'),
        out_degree_comps   = ('dst_comp',       'nunique'),
        out_degree_users   = ('dst_user',       'nunique'),
        active_hours       = ('time_bucket_hour','nunique'),
        active_days        = ('time_bucket_day', 'nunique'),
        first_seen         = ('time',            'min'),
        last_seen          = ('time',            'max'),
    ).reset_index().rename(columns={group_col: 'entity'})

    # Shannon entropy of destination distribution (high → broad lateral movement)
    dst_entropy = (
        grp['dst_comp']
        .apply(shannon_entropy)
        .reset_index()
        .rename(columns={group_col: 'entity', 'dst_comp': 'dst_entropy'})
    )

    # Shannon entropy of auth_type distribution (high → varied method usage)
    auth_entropy = (
        grp['auth_type_enc']
        .apply(shannon_entropy)
        .reset_index()
        .rename(columns={group_col: 'entity', 'auth_type_enc': 'auth_type_entropy'})
    )

    feat = (feat
            .merge(stats,        on='entity', how='left')
            .merge(dst_entropy,  on='entity', how='left')
            .merge(auth_entropy, on='entity', how='left'))

    # Lifetime span in seconds
    feat['lifespan'] = feat['last_seen'] - feat['first_seen']

    # Attack ratio (used only in evaluation, not training)
    feat['attack_ratio'] = feat['attack_events'] / feat['total_events'].clip(lower=1)

    return feat.fillna(0)


user_feat = build_node_features('src_user', user_to_idx, 'user')
comp_feat = build_node_features('src_comp', comp_to_idx, 'computer')

print(f"  User feature rows     : {len(user_feat):,}")
print(f"  Computer feature rows : {len(comp_feat):,}")


# ============================================================
# STEP 5 — Temporal split
# ============================================================

print("\n" + "=" * 60)
print("STEP 5 · Temporal split (strictly time-ordered)")
print("=" * 60)

t_vals  = edges_df['time'].sort_values().values
t80     = float(np.percentile(t_vals, 80))
t90     = float(np.percentile(t_vals, 90))

# Train : normal edges only (no labels — self-supervised)
train_mask = (edges_df['time'] <= t80) & (edges_df['label'] == 0)
val_mask   = (edges_df['time'] > t80) & (edges_df['time'] <= t90)
test_mask  =  edges_df['time'] > t90

train_idx  = edges_df.index[train_mask].tolist()
val_idx    = edges_df.index[val_mask].tolist()
test_idx   = edges_df.index[test_mask].tolist()

print(f"  t80 (train cutoff) : {t80:,.0f} s")
print(f"  t90 (val cutoff)   : {t90:,.0f} s")
print(f"  Train edges        : {len(train_idx):>10,}  (normal only)")
print(f"  Val edges (raw)    : {len(val_idx):>10,}"
      f"  — attacks: {edges_df.loc[val_idx,  'label'].sum():,}")
print(f"  Test edges (raw)   : {len(test_idx):>10,}"
      f"  — attacks: {edges_df.loc[test_idx, 'label'].sum():,}")


# ============================================================
# STEP 6 — Edge balance for evaluation splits
# ============================================================

print("\n" + "=" * 60)
print(f"STEP 6 · Balancing val / test  (1 attack : {NEG_RATIO} normal)")
print("=" * 60)

val_balanced  = balance_eval_split(edges_df, val_idx,  NEG_RATIO)
test_balanced = balance_eval_split(edges_df, test_idx, NEG_RATIO)

print(f"  Val  balanced  : {len(val_balanced):>8,}"
      f"  — attacks: {val_balanced['label'].sum():,}"
      f"  normals: {(val_balanced['label']==0).sum():,}")
print(f"  Test balanced  : {len(test_balanced):>8,}"
      f"  — attacks: {test_balanced['label'].sum():,}"
      f"  normals: {(test_balanced['label']==0).sum():,}")


# ============================================================
# STEP 7 — Save all outputs
# ============================================================

print("\n" + "=" * 60)
print("STEP 7 · Saving outputs")
print("=" * 60)

# Columns in train edges — labels deliberately excluded
EDGE_FEATURE_COLS = [
    'time', 'time_bucket_hour', 'time_bucket_day',
    'src_user_idx', 'dst_user_idx',
    'src_comp_idx', 'dst_comp_idx',
    'edge_type',
    'is_success', 'src_is_human', 'dst_is_human',
    'auth_type_enc', 'orientation_enc', 'logon_type_enc',
]

# Eval edge columns include the label
EVAL_COLS = EDGE_FEATURE_COLS + ['label']

train_edges = edges_df.loc[train_idx, EDGE_FEATURE_COLS]

paths = {
    'train_edges'         : f"{OUTPUT_DIR}/train_edges.parquet",
    'val_balanced'        : f"{OUTPUT_DIR}/val_edges_balanced.parquet",
    'test_balanced'       : f"{OUTPUT_DIR}/test_edges_balanced.parquet",
    'user_node_features'  : f"{OUTPUT_DIR}/user_node_features.parquet",
    'comp_node_features'  : f"{OUTPUT_DIR}/comp_node_features.parquet",
    'node_mappings'       : f"{OUTPUT_DIR}/node_mappings.pkl",
    'metadata'            : f"{OUTPUT_DIR}/metadata.json",
}

train_edges.to_parquet(paths['train_edges'], index=False)
val_balanced[EVAL_COLS].to_parquet(paths['val_balanced'], index=False)
test_balanced[EVAL_COLS].to_parquet(paths['test_balanced'], index=False)
user_feat.to_parquet(paths['user_node_features'], index=False)
comp_feat.to_parquet(paths['comp_node_features'], index=False)

with open(paths['node_mappings'], 'wb') as f:
    pickle.dump({'user_to_idx': user_to_idx, 'comp_to_idx': comp_to_idx}, f)

metadata = {
    # Graph shape
    'n_user_nodes'         : len(user_to_idx),
    'n_computer_nodes'     : len(comp_to_idx),
    # Split sizes
    'n_train_edges'        : len(train_edges),
    'n_val_edges'          : len(val_balanced),
    'n_test_edges'         : len(test_balanced),
    'n_val_attacks'        : int(val_balanced['label'].sum()),
    'n_test_attacks'       : int(test_balanced['label'].sum()),
    # Split boundaries
    'time_split_val'       : t80,
    'time_split_test'      : t90,
    # Schema documentation
    'edge_feature_cols'    : EDGE_FEATURE_COLS,
    'node_feature_cols'    : [
        'total_events', 'attack_events', 'success_rate',
        'out_degree_comps', 'out_degree_users',
        'active_hours', 'active_days',
        'first_seen', 'last_seen', 'lifespan',
        'dst_entropy', 'auth_type_entropy', 'attack_ratio',
    ],
    'edge_types'           : {
        '0': 'user→computer (authentication)',
        '1': 'computer→computer (lateral movement)',
        '2': 'user→user or other',
    },
    'node_types'           : ['user', 'computer'],
    'train_has_labels'     : False,  # self-supervised: no labels in training
    'eval_neg_ratio'       : NEG_RATIO,
}

def _jsonify(v):
    if isinstance(v, (np.integer,)):  return int(v)
    if isinstance(v, (np.floating,)): return float(v)
    return v

with open(paths['metadata'], 'w') as f:
    json.dump({k: _jsonify(v) for k, v in metadata.items()}, f, indent=2)

# ============================================================
# SUMMARY
# ============================================================

print()
print("╔══════════════════════════════════════════════════════╗")
print("║              DATASET SUMMARY                        ║")
print("╠══════════════════════════════════════════════════════╣")
print(f"║  USER nodes          : {len(user_to_idx):>10,}                   ║")
print(f"║  COMPUTER nodes      : {len(comp_to_idx):>10,}                   ║")
print("║  ─────────────────────────────────────────────────  ║")
print(f"║  Train edges         : {len(train_edges):>10,}  (label-free)      ║")
print(f"║  Val   edges (bal.)  : {len(val_balanced):>10,}  "
      f"attack={val_balanced['label'].sum():,}            ║")
print(f"║  Test  edges (bal.)  : {len(test_balanced):>10,}  "
      f"attack={test_balanced['label'].sum():,}            ║")
print("║  ─────────────────────────────────────────────────  ║")
print("║  Files saved:                                       ║")
for label, path in paths.items():
    size_mb = os.path.getsize(path) / 1e6 if os.path.exists(path) else 0
    print(f"║    {os.path.basename(path):<38}  {size_mb:>5.1f} MB  ║")
print("╚══════════════════════════════════════════════════════╝")