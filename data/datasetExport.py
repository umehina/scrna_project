import scanpy as sc
import numpy as np

adata = sc.read_h5ad("HW1_subset.h5ad")
a_df = adata.to_df()

# select 500 random cells, keeping all genes 
np.random.seed(67)
random_indices = np.random.choice(adata.n_obs, 500, False)
downsampled_df = a_df.iloc[:,random_indices].copy()

# export downsampled data to a CSV
downsampled_df.to_csv("scRNA_dataset.csv")