plot_umap_with_clusters <- function(umap_csv, labels_csv) {
  # Load packages
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Please install ggplot2: install.packages('ggplot2')")
  }
  if (!requireNamespace("dplyr", quietly = TRUE)) {
    stop("Please install dplyr: install.packages('dplyr')")
  }
  
  library(ggplot2)
  library(dplyr)
  
  # --- Debug: print paths and existence ---
  cat("Reading UMAP from:", umap_csv, "\n")
  cat("File exists?", file.exists(umap_csv), "\n")
  
  cat("Reading labels from:", labels_csv, "\n")
  cat("File exists?", file.exists(labels_csv), "\n")
  
  # 1. Read UMAP embedding
  umap_df <- read.csv(umap_csv, stringsAsFactors = FALSE)
  # Expect columns: node, UMAP1, UMAP2 (and maybe more UMAP dims)
  
  # 2. Read clustering labels
  label_df <- read.csv(labels_csv, stringsAsFactors = FALSE)
  # Expect columns: node, cluster
  
  # 3. Combine by "node"
  combined <- umap_df %>%
    inner_join(label_df, by = "node")
  # now we have: node, UMAP1, UMAP2, ..., cluster
  
  # Make sure cluster is treated as a factor (categorical)
  combined$cluster <- as.factor(combined$cluster)
  
  # 4. Plot UMAP colored by cluster
  p <- ggplot(combined, aes(x = UMAP1, y = UMAP2, color = cluster)) +
    geom_point(size = 1, alpha = 0.8) +
    coord_equal() +
    theme_minimal(base_size = 14) +
    labs(
      title = "UMAP Embedding with Leiden Clusters",
      x = "UMAP 1",
      y = "UMAP 2",
      color = "Cluster"
    )
  
  print(p)
  
  # Return the combined data in case you want to inspect it
  invisible(combined)
}

getwd()
# should be ".../scrna_project/R"

# Re-source the file that contains the function:
#source("plot_umap_with_clusters.R")

# Now call it:
plot_umap_with_clusters(
  umap_csv   = "../output/umap.csv",
  labels_csv = "../output/leiden_export_labels.csv"
)
