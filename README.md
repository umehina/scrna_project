# scRNA-seq Analysis Pipeline

A single-cell RNA-seq analysis pipeline implemented in Go with interactive visualization through R Shiny. The pipeline performs normalization, dimensionality reduction (PCA), embedding (UMAP/t-SNE), and Leiden clustering.

## Prerequisites

### Go
- Go 1.25.0 or higher
- Install from: https://go.dev/doc/install
- Required Go packages (should be automatically handled by go.mod):
	github.com/e-gun/go-tsne/tsne v0.0.0-20230417234659-5e6e23b13c15
	gonum.org/v1/gonum v0.16.0
	gonum.org/v1/plot v0.16.0

### R and Required Packages
- R version 4.0 or higher
- Required R packages:
  ```r
  install.packages(c("shiny", "ggplot2", "dplyr", "readr", "gridExtra"))
  ```

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/umehina/scrna_project
   cd scrna_project
   ```

2. **Install Go dependencies:**
   ```bash
   go mod download
   ```

3. **Build the Go pipeline:**
   ```bash
   go build -o scrna_project
   ```

4. **Verify installation:**
   ```bash
   ./scrna_project -h
   ```

## Quick Start

### Launch the R Shiny App

```bash
# From the project root directory
Rscript -e "shiny::runApp('R')"
```

Or from within R:
```r
library(shiny)
runApp("app.R")
```

The Shiny app will open in your default browser at `http://127.0.0.1:XXXX`

Click 'Run Pipeline' button towards the bottom to use the default dataset.

**Features:**
- Upload custom count matrix CSV files (max 200mb)
- Adjust QC filters (min/max features, counts, mitochondrial content)
- Choose normalization method (Pearson residuals or log-normalization)
- Configure PCA parameters (number of PCs, axes to plot)
- Select embedding method (UMAP, t-SNE, or both)
- Tune Leiden clustering parameters (k, resolution)
- Adjust UMAP/t-SNE hyperparameters
- Real-time visualization of results
- Download plots and clustering results

## Running the Pipeline

### Command Line

**Available flags for Go backend:**

| Flag | Description | Default | Range/Options |
|------|-------------|---------|---------------|
| `-data` | Path to input count matrix CSV | `data/scRNA_dataset.csv` | Any CSV file path |
| `-norm` | Normalization method | `pearson` | `pearson`, `lognorm` |
| `-npcs` | Number of principal components | 30 | 2-50 |
| `-embed` | Embedding method | `umap` | `umap`, `tsne` |
| `-k` | Leiden k-nearest neighbors | 25 | 5-100 |
| `-resolution` | Leiden resolution parameter | 0.5 | 0.1-2.0 |
| `-umap_neighbors` | UMAP n_neighbors parameter | 30 | 5-100 |
| `-umap_lr` | UMAP learning rate | 0.1 | 0.01-1.0 |
| `-tsne_perp` | t-SNE perplexity | 30.0 | 5-50 |
| `-tsne_lr` | t-SNE learning rate | 200.0 | 10-1000 |

---

## Common Issues

1. **"command not found: scango"**
   - Run `go build -o scrna_project` first
   - Ensure you're in the project directory
   - Use `./scrna_project` (with `./`) to run

2. **"Package 'shiny' not found"**
   - Install R packages: `install.packages(c("shiny", "ggplot2", "dplyr"))`

## Project Structure

```
scrna_project/
├── main.go                   # Main pipeline entry point
├── clustering.go             # Leiden clustering implementation
├── pca.go                    # PCA computation
├── normalization.go          # Normalization methods
├── preprocessing.go          # QC and filtering
├── umap_script.go            # UMAP implementation.
├── tSNE.go                   # t-SNE package implementation
├── io.go                     # Data I/O functions
├── helpers.go                # Helper functions
├── datatypes.go              # Data structures
├── app.R                     # Main Shiny app
├── R/
│   ├── app.R              # Alternative Shiny interface
│   ├── verify_clustering.R   # Clustering verification
│   ├── plot_leiden.R         # Leiden plotting utilities
│   └── visualization.R       # General plotting functions
├── data/
│   └── scRNA_dataset.csv     # Input count matrix
├── output/                   # Pipeline output files
└── README.md                 # This file
```

---

## Citation

- **Leiden algorithm**: Traag, V.A., Waltman, L. & van Eck, N.J. From Louvain to Leiden: guaranteeing well-connected communities. Sci Rep 9, 5233 (2019).
- **UMAP**: McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. arXiv:1802.03426.
- **t-SNE**: van der Maaten, L., & Hinton, G. (2008). Visualizing Data using t-SNE. Journal of Machine Learning Research, 9, 2579-2605.
