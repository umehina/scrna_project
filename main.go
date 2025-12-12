//Amy Ji - 11/1/2025
package main

import (
	"fmt"
	"log"

	"gonum.org/v1/gonum/mat"
)

// seperate the original main function into subfunctions.
func main() {
	// Load + QC filter dataset
	_, filtered := loadAndFilterDataset("data/scRNA_dataset.csv")

	// Build, normalize, scale expression matrix
	em := buildAndPreprocessMatrix(filtered)

	// Run PCA and save PCA plot
	pcResult := runPCAAndPlot(em, "pca.png", 0, 1)
	

	// Run Leiden clustering and export csv.
	k := 20 // Match Seurat's k.param
	maxIter := 10
	resolution := 1.0 // standard resolution
	gamma := 1.0
	theta := 0.01

	runLeidenandExport(pcResult.scores, k, maxIter, resolution, gamma, theta)



	// 4. Run UMAP on Leiden PCs and save embedding
	coordpath := "output/leiden_export_coords.csv"
	outpath:= "output/umap.csv"
	if err := runUMAPPipeline(coordpath, outpath); err != nil {
		log.Fatalf("UMAP pipeline failed: %v", err)
	}

	fmt.Println("UMAP Pipeline finished successfully.")
}


func loadAndFilterDataset(path string) (dataset, filtered *CountMatrix) {
	fmt.Println("Loading dataset:", path)

	d, err := ParseCountMatrixFromFile(path)
	if err != nil {
		log.Fatalf("Error loading dataset: %v", err)
	}
	d.ImportSummary()

	// QC thresholds (parameterize these or move them to a config later)
	minFeatures, maxFeatures := 200, 2500
	minCounts, maxCounts := 500, 5000
	maxPercentMT := 0.05

	fmt.Println("Applying QC filters...")
	f := d.FilterBy(minFeatures, maxFeatures, minCounts, maxCounts, maxPercentMT)
	f.ImportSummary()

	return d, f
}


func buildAndPreprocessMatrix(filtered *CountMatrix) *ExpressionMatrix {
	fmt.Print("Building expression matrix...")
	em, _, _ := BuildMatrix(filtered)
	fmt.Println(" done.")

	// Normalization
	fmt.Print("Normalizing (Pearson)...")
	em.Pearson(filtered, 100)
	fmt.Println(" done.")

	// Scaling
	fmt.Print("Scaling data...")
	em.ScaleData(10)
	fmt.Println(" done.")

	return &em
}

func runPCAAndPlot(em *ExpressionMatrix, outPath string, pcX, pcY int) *PCAResult {
	fmt.Print("Running PCA...")
	pcs := em.PCA(30)
	fmt.Println(" done.")

	fmt.Println("PC variances:", pcs.variances)

	fmt.Println("Plotting PCA scatter...")
	if err := pcs.PlotPCAScatter(outPath, pcX, pcY); err != nil {
		log.Fatalf("failed to plot PCA scatter: %v", err)
	}
	fmt.Printf("PCA plot saved as %s\n", outPath)

	return pcs
}

func runLeidenandExport(pcaCoords *mat.Dense, k, maxIter int, resolution, gamma, theta float64){
	g,clusteredNodes := RunLeiden(pcaCoords, k,maxIter,resolution,gamma,theta)
	ExportLeiden(g, clusteredNodes,pcaCoords,k,maxIter,resolution,gamma,theta)
}

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
	nNeighbors := 15
	nComponents := 2
	nEpochs := 1000
	learningRate := 0.1
	negativeSamples := 10
	minDist := 0.02
	// Note: if fuzzy edges mean is not within 0.2–0.6, parameters may not be ideal.

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