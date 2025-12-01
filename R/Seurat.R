library(tidyverse) # dplyr and ggplot2
library(Seurat) # Seurat toolkit
library(patchwork) # for plotting
library(sctransform)
library(ggplot2)

countsData <- read.csv("data/scRNA_dataset.csv", header = TRUE, row.names = 1)
transData <- t(countsData)

pbmc <- CreateSeuratObject(counts = transData)

pbmc[["percent.mt"]] <- PercentageFeatureSet(pbmc, pattern = "mt.")
metadata <- pbmc@meta.data

pbmc_filt<-subset(pbmc, nFeature_RNA > 200 & nFeature_RNA < 2500 & nCount_RNA > 500 &
                   nCount_RNA < 5000 & percent.mt < 5)

pbmc_filt <- SCTransform(pbmc_filt, vars.to.regress = "percent.mt", verbose = FALSE)

pbmc_filt
metadata1 <- pbmc_filt@meta.data

# These are now standard steps in the Seurat workflow for visualization and clustering
pbmc_filt <- RunPCA(pbmc_filt, verbose = FALSE)
pbmc_filt <- RunUMAP(pbmc_filt, dims = 1:30, verbose = FALSE)

pbmc_filt <- FindNeighbors(pbmc_filt, dims = 1:30, verbose = FALSE)
pbmc_filt <- FindClusters(pbmc_filt, verbose = FALSE)
DimPlot(pbmc_filt, label = TRUE)
