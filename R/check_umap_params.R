# Quick script to check what UMAP parameters Seurat is using

library(Seurat)

# Load the original data
countsData <- read.csv("data/scRNA_dataset.csv", header = TRUE, row.names = 1)
transData <- t(countsData)

# Create Seurat object
pbmc <- CreateSeuratObject(counts = transData)
pbmc[["percent.mt"]] <- PercentageFeatureSet(pbmc, pattern = "mt.")
pbmc_filt <- subset(pbmc, nFeature_RNA > 200 & nFeature_RNA < 2500 & 
                    nCount_RNA > 500 & nCount_RNA < 5000 & percent.mt < 5)

# Normalize and run PCA
pbmc_filt <- NormalizeData(pbmc_filt, verbose = FALSE)
pbmc_filt <- FindVariableFeatures(pbmc_filt, verbose = FALSE)
pbmc_filt <- ScaleData(pbmc_filt, verbose = FALSE)
pbmc_filt <- RunPCA(pbmc_filt, verbose = FALSE)

# Run UMAP
pbmc_filt <- RunUMAP(pbmc_filt, dims = 1:30, verbose = TRUE)

# Check the UMAP parameters that were used
cat("\n=== Seurat UMAP Parameters ===\n")
print(pbmc_filt@reductions$umap)
