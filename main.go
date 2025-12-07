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
	em.Pearson(filtered,100)
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

	// ================ Run UMAP =====================
	// 1. Load PCs from Leiden coords CSV
	data, nodeIDs, err := loadLeidenCoords("leiden_export_coords.csv")
	if err != nil {
		log.Fatalf("failed to load coords: %v", err)
	}
	fmt.Printf("Loaded %d cells with %d PCs\n", len(data), len(data[0]))
	fmt.Printf("NodeID range: %d..%d\n", nodeIDs[0], nodeIDs[len(nodeIDs)-1])

	// 2. UMAP hyperparameters (tune here)
	nNeighbors := 30
	nComponents := 2
	nEpochs := 1000
	learningRate := 0.2
	negativeSamples := 5
	minDist := 0.1
	// if fuzzy edges mean is not within range 0.2-0.6, the parameter is not good. 

	// 3. Run UMAP
	embedding := UMAP(
		data,
		nNeighbors,
		nComponents,
		nEpochs,
		learningRate,
		negativeSamples,
		minDist,
	)

	// 4. Quick sanity check
	if len(embedding) != len(data) {
		log.Fatalf("embedding length mismatch: got %d, want %d", len(embedding), len(data))
	}
	fmt.Printf("UMAP embedding shape: %d x %d\n", len(embedding), len(embedding[0]))

	if err := writeEmbeddingCSV("umap_out.csv", nodeIDs, embedding); err != nil {
		log.Fatalf("failed to write UMAP csv: %v", err)
	}

	fmt.Println("UMAP embedding written to umap_out.csv")


	/*
	// ================= Run TSNE ====================
	fmt.Println("Running t-SNE")
	tsneResult := pcs.TSNE(2,30,200,1000) // to cluster after tSNE, call variable name tsneResult
	if err != nil {
		log.Fatal(err)
	}
	t_title:="tsne"
	t_xlabel:="tsne_1"
	t_ylabel:="tsne_2"
	PlotEmb(tsneResult.scores,"tsne.png",t_title,t_xlabel,t_ylabel )
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
	u_title:="umap"
	u_xlabel:="umap_1"
	u_ylabel:="umap_2"
	PlotEmb(umapResult,"umap.png", u_title,u_xlabel,u_ylabel)
	*/
}
