import pandas as pd
df = pd.read_parquet('./graphs/graph_edges.parquet')
print(df.columns.tolist())
print(df.head(3))
print(df.dtypes)