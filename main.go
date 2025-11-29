package main

import (
	"fmt"
	"log"
	
)
//Amy Ji - 11/1/2025
func main() {
	// load dataset, returns CountMatrix struct
	dataset, err := ParseCountMatrixFromFile("data/scRNA_dataset.csv")
	if err != nil {
		log.Fatalf("Error: %v", err)
	}
	dataset.ImportSummary()

	// perform QC filtering

	filtered := dataset.FilterBy(200, 2500, 500, 5000, 0.05)
	filtered.ImportSummary()

	fmt.Print("Building em...")
	em, _, _ := BuildMatrix(filtered)
	fmt.Println(" done.")

	fmt.Print("Normalizing...")
	em.LogNormalize(1e6)
	fmt.Println(" done.")

	fmt.Print("Scaling...")
	em.ScaleData(10)
	fmt.Println(" done.")

	// ================ Run PCA ====================
	fmt.Print("Running PCA...")
	pcs := em.PCA(30)
	fmt.Println(" done.")
	fmt.Println("PC variances:", pcs.variances)

	fmt.Println("Plotting PCA...")
	err = pcs.PlotPCAScatter("pca.png", 0, 1)
	if err != nil {
		panic(err)
	}
	fmt.Println("PCA plot saved as pca.png")

	// ================= Run TSNE ====================
	fmt.Println("Running t-SNE")
	tsneResult := pcs.TSNE(2,30,200,1000) // to cluster after tSNE, call variable name tsneResult
	if err != nil {
		log.Fatal(err)
	}
	PlotTSNE(tsneResult.scores,"tsne.png")
	fmt.Println("TSNE plot saved as tsne.png")
	
	// ================== Run UMAP ====================
	fmt.Println("converting scores to csv file")
	SavePCAScoresForR(pcs,"pca_scores.csv")

	fmt.Println("Running UMAP in R")
	// set UMAP parameters here!
	if err := RunRUMAP("umap_script.R", "pca_scores.csv", "umap_out.csv", 30, 0.3, "euclidean"); err != nil {
    	log.Fatal(err)
	}

	// Read UMAP embedding back into Go
	// umapCells are the last column, checkout umap_out.csv file.
	// It is omitted here, but we may need the cell barcode info in the future.
	umapResult,_, err := LoadUMAPFromCSV("umap_out.csv") // to cluster after umap, call variable name umapResult
	if err != nil {
    	log.Fatal(err)
	}

	fmt.Println("Plotting UMAP")
	PlotTSNE(umapResult,"umap.png")

}
