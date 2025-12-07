# plot_leiden.R
# Reads PCA coordinates and Leiden cluster labels exported from Go
# Plots PC1 vs PC2 colored by cluster

# Install/load ggplot2
if (!require("ggplot2", quietly = TRUE)) {
  install.packages("ggplot2", repos = "http://cran.r-project.org", quiet = TRUE)
}
library(ggplot2)

# Read PCA coordinates
coords_path <- "R/leiden_export_coords.csv"
if (!file.exists(coords_path)) {
  cat("Error: leiden_export_coords.csv not found\n")
  quit(status = 1)
}

coords_df <- read.csv(coords_path, header = TRUE, stringsAsFactors = FALSE)
cat("Loaded PCA coordinates for", nrow(coords_df), "cells\n")

# Read cluster labels
labels_path <- "R/leiden_export_labels.csv"
if (!file.exists(labels_path)) {
  cat("Error: leiden_export_labels.csv not found\n")
  quit(status = 1)
}

labels_df <- read.csv(labels_path, header = TRUE, stringsAsFactors = FALSE)
cat("Loaded", nrow(labels_df), "cluster labels\n")

# Merge coordinates and labels by node
merged <- merge(coords_df, labels_df, by = "node")
merged$cluster <- as.factor(merged$cluster)

cat("Number of clusters:", length(unique(merged$cluster)), "\n")

# Plot PC1 vs PC2 colored by Leiden clusters
p <- ggplot(merged, aes(x = PC1, y = PC2, color = cluster)) +
  geom_point(size = 1.5, alpha = 0.7) +
  theme_minimal() +
  theme(legend.position = "right") +
  labs(title = "PCA colored by Leiden clusters",
       x = "PC1",
       y = "PC2",
       color = "Cluster")

# Save plot
output_file <- "R/leiden_pca.png"
ggsave(filename = output_file, plot = p, width = 8, height = 6, dpi = 150)
cat("Saved plot to", output_file, "\n")
