import scanpy as sc
import numpy as np

adata = sc.read_10x_h5("./filtered_feature_bc_matrix.h5")
a_df = adata.to_df()

# select 500 random cells, keeping all genes 
np.random.seed(67)
random_indices = np.random.choice(adata.n_obs, 750, replace=False)
downsampled_df = a_df.iloc[random_indices, :].copy()

# export downsampled data to CSV
downsampled_df.to_csv("scRNA_dataset.csv")