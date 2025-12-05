
# UMAP Implementation - Yinan Zhu; Amy Ji 11/27/2025
###############################

library(tidyverse)
library(uwot)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 5) {
  stop("Usage: umap_script.R <input_pca_csv> <output_umap_csv>")
}

# We can customize these important parameters in go.
input_csv  <- args[1]  
output_csv <- args[2]
n_neighbors <- as.integer(args[3])
min_dist <- as.numeric(args[4])
metric <- args[5]


# Load PCA scores: rows = cells, cols = PCs, first column = cell/barcode
pca_scores <- read.csv(
  input_csv,
  header = TRUE,
  row.names = 1,
  check.names = FALSE
)

pca_mat <- as.matrix(pca_scores)

set.seed(123)
umap_emb <- umap(
  pca_mat,
  n_neighbors = n_neighbors,
  min_dist    = min_dist,
  metric      = metric
)

umap_df <- data.frame(
  UMAP1 = umap_emb[, 1],
  UMAP2 = umap_emb[, 2],
  cell  = rownames(pca_mat)
)

# Save for Go to read back
# cell as first column, then UMAP1, UMAP2
write.csv(umap_df, file = output_csv, row.names = FALSE)
