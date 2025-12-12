// Amy Ji - 11/1/2025
package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"

	"gonum.org/v1/gonum/mat"
)


// main now takes CLI flags so R Shiny can control the pipeline:
//
//   -norm   : "pearson" or "lognorm"
//   -npcs   : number of PCs (2–50)
//   -embed  : "umap" or "tsne"
//
// seperate the original main function into subfunctions.

func main() {
	// -----------------------------------------------------------------
	// CLI flags so R Shiny can control the pipeline
	// -----------------------------------------------------------------
	normMethod := flag.String("norm", "pearson",
		"Normalization method: 'pearson' or 'lognorm'")
	nPCs := flag.Int("npcs", 30,
		"Number of principal components (2–50)")
	embedMethod := flag.String("embed", "umap",
		"Embedding method: 'umap' or 'tsne'")
	dataPath := flag.String("data", "data/scRNA_dataset.csv",
		"Path to input count matrix CSV (relative to project root)")

	flag.Parse()

	// clamp PCs
	if *nPCs < 2 {
		*nPCs = 2
	}
	if *nPCs > 50 {
		*nPCs = 50
	}

	fmt.Printf("Running pipeline with norm=%s, nPCs=%d, embed=%s\n",
		*normMethod, *nPCs, *embedMethod)
	fmt.Println("Loading dataset from:", *dataPath)

	// -----------------------------------------------------------------
	// 1) Load + QC filter dataset
	// -----------------------------------------------------------------
	_, filtered := loadAndFilterDataset(*dataPath)

	// -----------------------------------------------------------------
	// 2) Build, normalize, scale expression matrix
	// -----------------------------------------------------------------
	// ASSUMPTION: you already have a function like
	//   buildAndPreprocessMatrix(filtered *CountMatrix, normMethod string) *ExpressionMatrix
	// If yours currently has a different signature (e.g., no normMethod),
	// you can either:
	//   - add the normMethod parameter to that function, OR
	//   - ignore this argument and keep internal logic as-is for now.
	em := buildAndPreprocessMatrix(filtered, *normMethod)

	// -----------------------------------------------------------------
	// 3) Run PCA and save PCA plot
	// -----------------------------------------------------------------
	// ASSUMPTION: you have something like
	//   runPCAAndPlot(em *ExpressionMatrix, outPath string, pcX, pcY, nPCs int) *PCAResult
	pcaPlotPath := filepath.Join("output", "pca.png")
	pcResult := runPCAAndPlot(em, pcaPlotPath, 0, 1, *nPCs)

	// -----------------------------------------------------------------
	// 4) Run Leiden clustering and export CSVs into ./output
	// -----------------------------------------------------------------
	// You can keep your original k / resolution / gamma / theta here
	k := 20       // example default
	maxIter := 10 // example default
	resolution := 1.0
	gamma := 1.0
	theta := 0.01

	runLeidenandExport(pcResult.scores, k, maxIter, resolution, gamma, theta)

	// -----------------------------------------------------------------
	// 5) Embedding: UMAP or t-SNE
	// -----------------------------------------------------------------
	switch *embedMethod {
	case "tsne":
		// t-SNE pipeline writes ./output/tsne.csv
		if err := runTSNEPipeline(pcResult, filepath.Join("output", "tsne.csv")); err != nil {
			log.Fatalf("t-SNE pipeline failed: %v", err)
		}
		fmt.Println("t-SNE pipeline finished successfully.")
	default: // "umap"
		// UMAP pipeline (already defined in your main.go)
		// ASSUMPTION: you have:
		//   func runUMAPPipeline(coordPath, outPath string) error
		coordPath := filepath.Join("output", "leiden_export_coords.csv")
		umapPath := filepath.Join("output", "umap.csv")
		if err := runUMAPPipeline(coordPath, umapPath); err != nil {
			log.Fatalf("UMAP pipeline failed: %v", err)
		}
		fmt.Println("UMAP pipeline finished successfully.")
	}
}


// ---------------- Data loading & QC ----------------

func loadAndFilterDataset(path string) (dataset, filtered *CountMatrix) {
	fmt.Println("Loading dataset:", path)

	d, err := ParseCountMatrixFromFile(path)
	if err != nil {
		log.Fatalf("Error loading dataset: %v", err)
	}
	d.ImportSummary()

	// QC thresholds (parameterize later if desired)
	minFeatures, maxFeatures := 200, 2500
	minCounts, maxCounts := 500, 5000
	maxPercentMT := 0.05

	fmt.Println("Applying QC filters...")
	f := d.FilterBy(minFeatures, maxFeatures, minCounts, maxCounts, maxPercentMT)
	f.ImportSummary()

	return d, f
}

// ---------------- Normalization & scaling ----------------

// buildAndPreprocessMatrix now supports both Pearson and LogNormalize.
//
// normMethod: "pearson" or "lognorm"
func buildAndPreprocessMatrix(filtered *CountMatrix, normMethod string) *ExpressionMatrix {
	fmt.Print("Building expression matrix...")
	em, _, _ := BuildMatrix(filtered)
	fmt.Println(" done.")

	// Normalization
	switch normMethod {
	case "lognorm":
		fmt.Print("Normalizing (LogNormalize)...")
		// standard Seurat/Scanpy-like choice, can be tuned later
		em.LogNormalize(1e4)
	default:
		fmt.Print("Normalizing (Pearson residuals)...")
		// Pearson returns a *ExpressionMatrix; copy it back into em
		normalized := em.Pearson(100)
		em = *normalized
	}
	fmt.Println(" done.")

	// Scaling
	fmt.Print("Scaling data...")
	em.ScaleData(10)
	fmt.Println(" done.")

	return &em
}

// ---------------- PCA ----------------

func runPCAAndPlot(em *ExpressionMatrix, outPath string, pcX, pcY, nPCs int) *PCAResult {
	fmt.Printf("Running PCA with %d components...", nPCs)
	pcs := em.PCA(nPCs)
	fmt.Println(" done.")

	fmt.Println("PC variances:", pcs.variances)

	fmt.Println("Plotting PCA scatter...")
	if err := pcs.PlotPCAScatter(outPath, pcX, pcY); err != nil {
		log.Fatalf("failed to plot PCA scatter: %v", err)
	}
	fmt.Printf("PCA plot saved as %s\n", outPath)

	return pcs
}

// ---------------- Leiden clustering ----------------

func runLeidenandExport(pcaCoords *mat.Dense, k, maxIter int, resolution, gamma, theta float64) {
	g, clusteredNodes := RunLeiden(pcaCoords, k, maxIter, resolution, gamma, theta)
	ExportLeiden(g, clusteredNodes, pcaCoords, k, maxIter, resolution, gamma, theta)
}

// ---------------- UMAP pipeline (unchanged API) ----------------
//
// runUMAPPipeline is as in your existing code; kept here for completeness.
// It reads "output/leiden_export_coords.csv", runs UMAP, and writes "output/umap.csv".
//
func runUMAPPipeline(coordPath, outPath string) error {
	// 1. Load PCs from Leiden coords CSV
	fmt.Println("Loading Leiden coordinates from:", coordPath)
	data, nodeIDs, err := loadLeidenCoords(coordPath)
	if err != nil {
		return fmt.Errorf("failed to load coords: %w", err)
	}
	fmt.Printf("Loaded %d cells with %d PCs\n", len(data), len(data[0]))
	fmt.Printf("NodeID range: %d..%d\n", nodeIDs[0], nodeIDs[len(nodeIDs)-1])

	// 2. UMAP hyperparameters (tune here for now)
	nNeighbors := 30
	nComponents := 2
	nEpochs := 1000
	learningRate := 0.1
	negativeSamples := 10
	minDist := 0.1

	fmt.Println("Running UMAP...")
	embedding := UMAP(
		data,
		nNeighbors,
		nComponents,
		nEpochs,
		learningRate,
		negativeSamples,
		minDist,
	)

	// Quick sanity check
	if len(embedding) != len(data) {
		return fmt.Errorf("embedding length mismatch: got %d, want %d", len(embedding), len(data))
	}
	fmt.Printf("UMAP embedding shape: %d x %d\n", len(embedding), len(embedding[0]))

	// Write embedding to CSV
	if err := writeEmbeddingCSV(outPath, nodeIDs, embedding); err != nil {
		return fmt.Errorf("failed to write UMAP csv: %w", err)
	}
	fmt.Printf("UMAP embedding written to %s\n", outPath)

	return nil
}

// ---------------- t-SNE pipeline (NEW) ----------------
//
// This runs t-SNE on the PCA scores and writes "output/tsne.csv".
// It assumes writeEmbeddingCSV(outPath, nodeIDs, embedding) already exists
// and will name columns appropriately (e.g., TSNE1/TSNE2).
//
func runTSNEPipeline(pcs *PCAResult, outPath string) error {
	if pcs == nil || pcs.scores == nil {
		return fmt.Errorf("t-SNE pipeline: PCA result is nil")
	}

	fmt.Println("Running t-SNE...")
	// dimsOut, perplexity, learning rate, maxIter
	tsneRes := pcs.TSNE(2, 30.0, 200.0, 1000)
	if tsneRes == nil || tsneRes.scores == nil {
		return fmt.Errorf("t-SNE pipeline: TSNE result is nil")
	}

	rows, cols := tsneRes.scores.Dims()
	fmt.Printf("t-SNE embedding shape: %d x %d\n", rows, cols)

	// convert mat.Dense -> [][]float64
	embedding := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		row := tsneRes.scores.RawRowView(i)
		embedding[i] = make([]float64, cols)
		copy(embedding[i], row)
	}

	// node IDs must match Leiden export (1-based)
	nodeIDs := make([]int, rows)
	for i := range nodeIDs {
		nodeIDs[i] = i + 1
	}

	if err := writeEmbeddingCSV(outPath, nodeIDs, embedding); err != nil {
		return fmt.Errorf("failed to write t-SNE csv: %w", err)
	}
	fmt.Printf("t-SNE embedding written to %s\n", outPath)
	return nil
}
