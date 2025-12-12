# scRNA-seq Analysis Pipeline

A single-cell RNA-seq analysis pipeline implemented in Go with interactive visualization through R Shiny. The pipeline performs normalization, dimensionality reduction (PCA), embedding (UMAP/t-SNE), and Leiden clustering.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Running the Pipeline](#running-the-pipeline)
  - [Command Line](#command-line)
  - [R Shiny App](#r-shiny-app)
- [Pipeline Overview](#pipeline-overview)
- [Parameters](#parameters)
- [Output Files](#output-files)
- [Visualization](#visualization)

---

## Prerequisites

### Go
- Go 1.25.0 or higher
- Install from: https://go.dev/doc/install

### R and Required Packages
- R version 4.0 or higher
- Required R packages:
  ```r
  install.packages(c("shiny", "ggplot2", "dplyr", "readr", "gridExtra"))
  ```

### Python (Optional)
- Python 3.x with `umap-learn` for UMAP generation
- Install: `pip install umap-learn`

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd scrna_project
   ```

2. **Install Go dependencies:**
   ```bash
   go mod download
   ```

3. **Build the Go pipeline:**
   ```bash
   go build -o scango
   ```

4. **Verify installation:**
   ```bash
   ./scango -h
   ```

---

## Quick Start

### Run with Default Parameters

```bash
# Build the pipeline
go build -o scango

# Run with defaults (Pearson normalization, 30 PCs, UMAP embedding)
./scango
```

### Launch the R Shiny App

```bash
# From the project root directory
R -e "shiny::runApp('app.R')"
```

Or from within R:
```r
library(shiny)
runApp("app.R")
```

The Shiny app will open in your default browser at `http://127.0.0.1:XXXX`

---

## Running the Pipeline

### Command Line

The Go pipeline can be run directly with various parameters:

```bash
./scango \
  -data data/scRNA_dataset.csv \
  -norm pearson \
  -npcs 30 \
  -embed umap \
  -k 25 \
  -resolution 0.5 \
  -umap_neighbors 30 \
  -umap_lr 0.1
```

**Available flags:**

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

### R Shiny App

The R Shiny app provides an interactive interface to run the pipeline with custom parameters:

#### Method 1: Using `app.R` (Recommended)
```bash
R -e "shiny::runApp('app.R')"
```

#### Method 2: Using `R/Rshiny.R`
```bash
cd R
R -e "shiny::runApp('Rshiny.R')"
```

**Features:**
- Upload custom count matrix CSV files
- Adjust QC filters (min/max features, counts, mitochondrial content)
- Choose normalization method (Pearson residuals or log-normalization)
- Configure PCA parameters (number of PCs, axes to plot)
- Select embedding method (UMAP, t-SNE, or both)
- Tune Leiden clustering parameters (k, resolution)
- Adjust UMAP/t-SNE hyperparameters
- Real-time visualization of results
- Download plots and clustering results

---

## Pipeline Overview

### Workflow

```
Input CSV → QC Filtering → Normalization → PCA → Embedding (UMAP/t-SNE) → Leiden Clustering → Visualization
```

### Steps

1. **Quality Control (QC)**
   - Filter cells by feature count, total counts, and mitochondrial percentage
   - Remove low-quality cells

2. **Normalization**
   - **Pearson Residuals**: SCTransform-style normalization using Pearson residuals
   - **Log-normalization**: Traditional log(counts + 1) normalization

3. **Dimensionality Reduction (PCA)**
   - Compute principal components
   - Default: 30 PCs for downstream analysis

4. **Embedding**
   - **UMAP**: Uniform Manifold Approximation and Projection
   - **t-SNE**: t-distributed Stochastic Neighbor Embedding

5. **Leiden Clustering**
   - Community detection using the Leiden algorithm
   - Builds k-nearest neighbor graph
   - Optimizes modularity with refinement phase
   - Parameters: k (neighbors), resolution, gamma (γ-connectivity), theta (randomness)

6. **Visualization**
   - Generate PCA plots
   - Create embedding plots colored by cluster
   - Export clustering results

---

## Parameters

### Leiden Clustering Parameters

- **k (neighbors)**: Number of nearest neighbors in the KNN graph (default: 25)
  - Higher k → larger, more connected clusters
  - Lower k → smaller, more granular clusters

- **resolution**: Controls the number of clusters (default: 0.5-1.0)
  - Higher resolution → more clusters
  - Lower resolution → fewer clusters

- **gamma**: γ-connectivity parameter (default: 1.0)
  - Ensures communities are γ-connected
  - Standard value is 1.0

- **theta**: Randomness parameter in refinement (default: 0.01)
  - Lower values → more deterministic
  - Higher values → more exploration

### UMAP Parameters

- **n_neighbors**: Number of neighbors (default: 30)
  - Higher → more global structure
  - Lower → more local structure

- **learning_rate**: Optimization learning rate (default: 0.1)

- **min_dist**: Minimum distance between points (default: 0.3)

- **metric**: Distance metric (default: cosine)

### t-SNE Parameters

- **perplexity**: Balance between local and global structure (default: 30.0)
  - Higher → more global structure
  - Typically 5-50

- **learning_rate**: Optimization step size (default: 200.0)

---

## Output Files

### Generated in `output/` directory:

1. **leiden_export_coords.csv**: PCA coordinates for each cell
2. **leiden_export_edges.csv**: KNN graph edges with weights
3. **leiden_export_labels.csv**: Cluster assignments for each cell
4. **leiden_export_params.csv**: Parameters used for clustering
5. **umap.csv**: UMAP coordinates (if UMAP selected)
6. **tsne.csv**: t-SNE coordinates (if t-SNE selected)

### Generated in `R/` directory:

1. **leiden_pca.png**: PCA plot colored by cluster
2. **leiden_umap.png**: UMAP plot colored by cluster
3. **clustering_comparison.png**: Comparison of clustering results

### Generated in project root:

1. **pca.png**: Initial PCA visualization

---

## Visualization

### Verify Clustering Results

After running the pipeline, visualize results using:

```bash
cd R
Rscript verify_clustering.R
```

This generates comparison plots between the Go implementation and reference implementations.

### Interactive Exploration

Use the R Shiny app for interactive exploration:
- Adjust parameters in real-time
- View updated plots immediately
- Compare different parameter combinations
- Export publication-quality figures

---

## Troubleshooting

### Common Issues

1. **"command not found: scango"**
   - Run `go build -o scango` first
   - Ensure you're in the project directory
   - Use `./scango` (with `./`) to run

2. **"Package 'shiny' not found"**
   - Install R packages: `install.packages(c("shiny", "ggplot2", "dplyr"))`

3. **"UMAP not available"**
   - Install Python UMAP: `pip install umap-learn`
   - Check that Python is in your PATH

4. **"Cannot find data file"**
   - Verify data file path with `-data` flag
   - Default location: `data/scRNA_dataset.csv`

5. **Memory errors with large datasets**
   - Reduce number of PCs (`-npcs`)
   - Use more aggressive QC filtering
   - Consider running on a machine with more RAM

### Getting Help

- Check flag options: `./scango -h`
- Review error messages in the console
- Verify input data format (CSV with genes as rows, cells as columns)

---

## Project Structure

```
scrna_project/
├── main.go                 # Main pipeline entry point
├── clustering.go           # Leiden clustering implementation
├── pca.go                  # PCA computation
├── normalization.go        # Normalization methods
├── preprocessing.go        # QC and filtering
├── umap_script.go         # UMAP interface
├── tSNE.go                # t-SNE implementation
├── io.go                  # Data I/O functions
├── helpers.go             # Helper functions
├── datatypes.go           # Data structures
├── app.R                  # Main Shiny app
├── R/
│   ├── Rshiny.R          # Alternative Shiny interface
│   ├── verify_clustering.R   # Clustering verification
│   ├── plot_leiden.R     # Leiden plotting utilities
│   └── visualization.R   # General plotting functions
├── data/
│   └── scRNA_dataset.csv # Input count matrix
├── output/               # Pipeline output files
└── README.md            # This file
```

---

## Citation

If you use this pipeline in your research, please cite:

- **Leiden algorithm**: Traag, V.A., Waltman, L. & van Eck, N.J. From Louvain to Leiden: guaranteeing well-connected communities. Sci Rep 9, 5233 (2019).
- **UMAP**: McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. arXiv:1802.03426.
- **t-SNE**: van der Maaten, L., & Hinton, G. (2008). Visualizing Data using t-SNE. Journal of Machine Learning Research, 9, 2579-2605.

---

## License

[Add your license information here]

## Contributors

[Add contributor information here]

---

## Contact

For questions or issues, please open an issue on the GitHub repository or contact [your contact information].
