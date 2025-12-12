# verify_clustering.R
# Compares Seurat clustering with Go Leiden implementation
# Generates side-by-side UMAP plots

# Install/load required packages
if (!require("Seurat", quietly = TRUE)) {
  install.packages("Seurat", repos = "http://cran.r-project.org")
}
library(Seurat)

if (!require("ggplot2", quietly = TRUE)) {
  install.packages("ggplot2", repos = "http://cran.r-project.org")
}
library(ggplot2)

if (!require("patchwork", quietly = TRUE)) {
  install.packages("patchwork", repos = "http://cran.r-project.org")
}
library(patchwork)

if (!require("uwot", quietly = TRUE)) {
  install.packages("uwot", repos = "http://cran.r-project.org")
}
library(uwot)

if (!require("dplyr", quietly = TRUE)) {
  install.packages("dplyr", repos = "http://cran.r-project.org")
}
library(dplyr)

cat("=== Seurat Clustering ===\n")

# Load the original data
countsData <- read.csv("data/scRNA_dataset.csv", header = TRUE, row.names = 1)
transData <- t(countsData)

# Create Seurat object
pbmc <- CreateSeuratObject(counts = transData)

# Calculate mitochondrial percentage
pbmc[["percent.mt"]] <- PercentageFeatureSet(pbmc, pattern = "mt.")

# Filter cells (same filtering as in Go pipeline)
pbmc_filt <- subset(pbmc, nFeature_RNA > 200 & nFeature_RNA < 2500 & 
                    nCount_RNA > 500 & nCount_RNA < 5000 & percent.mt < 5)

cat("Filtered to", ncol(pbmc_filt), "cells\n")

# Normalize and scale
pbmc_filt <- NormalizeData(pbmc_filt, verbose = FALSE)
pbmc_filt <- FindVariableFeatures(pbmc_filt, verbose = FALSE)
pbmc_filt <- ScaleData(pbmc_filt, verbose = FALSE)

# Run PCA
pbmc_filt <- RunPCA(pbmc_filt, npcs = 30, verbose = FALSE)

# Run UMAP
pbmc_filt <- RunUMAP(pbmc_filt, dims = 1:30, verbose = FALSE)

# Find neighbors and clusters
pbmc_filt <- FindNeighbors(pbmc_filt, dims = 1:30, verbose = FALSE)
pbmc_filt <- FindClusters(pbmc_filt, resolution = 1.0, verbose = FALSE)

# Get Seurat clustering results
seurat_clusters <- pbmc_filt@meta.data$seurat_clusters
cat("Seurat found", length(unique(seurat_clusters)), "clusters\n")

# Create Seurat plot
p1 <- DimPlot(pbmc_filt, reduction = "umap", label = TRUE, label.size = 5) + 
  ggtitle("Seurat UMAP + Leiden Clustering") +
  theme_minimal() +
  theme(legend.position = "right")

cat("\n=== Go Leiden Clustering ===\n")

# Load Go clustering results
go_coords <- read.csv("R/leiden_export_coords.csv", header = TRUE)
go_labels <- read.csv("R/leiden_export_labels.csv", header = TRUE)

cat("Go implementation found", length(unique(go_labels$cluster)), "clusters\n")

# Extract PCA coordinates from Go output (now 30 PCs)
go_pca_data <- as.matrix(go_coords[, grep("^PC", names(go_coords))])

# Compute UMAP from Go PCA coordinates (same parameters as Seurat)
# Seurat uses: n_neighbors=30, metric=cosine, min_dist=0.3, spread=1
go_umap_result <- umap(go_pca_data, n_neighbors = 30, metric = "cosine", 
                       min_dist = 0.3, spread = 1, verbose = FALSE)
go_umap_coords <- as.data.frame(go_umap_result)
colnames(go_umap_coords) <- c("UMAP1", "UMAP2")
go_umap_coords$node <- go_coords$node

# Merge UMAP coordinates with Go cluster labels
go_data <- merge(go_umap_coords, go_labels, by = "node")
go_data$cluster <- as.factor(go_data$cluster)

# Create Go plot
p2 <- ggplot(go_data, aes(x = UMAP1, y = UMAP2, color = cluster)) +
  geom_point(size = 1.5, alpha = 0.7) +
  theme_minimal() +
  ggtitle("Go UMAP + Leiden Clustering") +
  labs(x = "UMAP1", y = "UMAP2", color = "Cluster") +
  theme(legend.position = "right") +
  geom_text(data = go_data %>% 
              group_by(cluster) %>% 
              summarize(x = mean(UMAP1), y = mean(UMAP2)),
            aes(x = x, y = y, label = cluster), 
            color = "black", size = 5, fontface = "bold")

# Combine plots side by side
comparison_plot <- p1 | p2

# Save comparison plot
ggsave("R/clustering_comparison.png", plot = comparison_plot, 
       width = 16, height = 6, dpi = 150)

cat("\n=== Cluster Size Comparison ===\n")
cat("\nSeurat cluster sizes:\n")
print(table(seurat_clusters))

cat("\nGo implementation cluster sizes:\n")
print(table(go_labels$cluster))

# Calculate agreement metrics if same number of cells
if (nrow(go_labels) == length(seurat_clusters)) {
  if (!require("mclust", quietly = TRUE)) {
    install.packages("mclust", repos = "http://cran.r-project.org")
  }
  library(mclust)
  
  ari <- adjustedRandIndex(as.numeric(seurat_clusters), go_labels$cluster)
  cat("\nAdjusted Rand Index (ARI):", round(ari, 3), "\n")
  cat("(ARI ranges from 0 to 1, where 1 = perfect agreement)\n")
}

cat("\nComparison plot saved to R/clustering_comparison.png\n")
cat("=== Verification Complete ===\n")
