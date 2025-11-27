
# UMAP Implementation - Yinan Zhu 11/27/2025
###############################

library(tidyverse)
library(uwot)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: umap_script.R <input_pca_csv> <output_umap_csv>")
}
input_csv  <- args[0 + 1]  # R is 1-based
output_csv <- args[1 + 1]

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
  n_neighbors = 15,
  min_dist    = 0.3,
  metric      = "euclidean"
)

umap_df <- data.frame(
  UMAP1 = umap_emb[, 1],
  UMAP2 = umap_emb[, 2],
  cell  = rownames(pca_mat)
)

# Save for Go to read back
# cell as first column, then UMAP1, UMAP2
write.csv(umap_df, file = output_csv, row.names = FALSE)
