package main

import (
	"log"
	"fmt"
)

func main() {
	// load dataset, returns CountMatrix struct
	dataset, err := ParseCountMatrixFromFile("data/scRNA_dataset.csv")
	if err != nil {
		log.Fatalf("Error: %v", err)
	}
	dataset.Summary()

	// perform QC filtering
	filtered := dataset.FilterBy(200, 2500, 500, 5000, 0.05)
	filtered.Summary()

	//perform normalization
	sf := 10000.0
	normalized := filtered.LogNormalize(sf)
	normalized.Summary()
	
	// transform countMatrix to a *mat.Dense object
	MMatrix := cmToDense(normalized)

	// =================== TSNE ============================
	// Define perplexity and iterations for RunTSNE
	perplexity := 30.0
	iterations := 5000
	tsne := RunTSNE(MMatrix, perplexity, iterations)
	// Print cell numbers and dimension
	fmt.Println(tsne.RawMatrix().Rows, tsne.RawMatrix().Cols)
	// plot mat.Dense after tSNE
	err1 := PlotTSNE(tsne, "tsne.png")
	if err1 != nil {
    	panic(err1)
	}



}
