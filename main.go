package main

import (
	"fmt"
	"log"
)

func main() {
	//Amy Ji - 11/1/2025
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

	// normalized := em.Pearson(100)

	// fmt.Print("Normalizing...")
	em.LogNormalize(1e6)
	// fmt.Println(" done.")

	// fmt.Print("Scaling...")
	em.ScaleData(10)
	// fmt.Println(" done.")

	fmt.Print("Running PCA...")
	pcs := em.PCA(2)
	fmt.Println(" done.")
	fmt.Println("PC variances:", pcs.variances)

	fmt.Println("Plotting PCA...")
	err = pcs.PlotPCAScatter("pca.png", 0, 1)
	if err != nil {
		panic(err)
	}
	fmt.Println("PCA plot saved as pca.png")

	// //perform normalization
	// sff := 10000.00
	// // LogNormalize modifies 'filtered' in place and does not return a value, so call it directly.
	// filtered.LogNormalize(sff)
	// //filtered.Summary()

	// // transform countMatrix to a *mat.Dense object
	// MMatrix := CmToDense(filtered)
	// // =================== TSNE ============================
	// // Amy Ji - 11/13/2025
	// // Define perplexity and iterations for RunTSNE
	// perplexity := 30.0
	// iterations := 1000
	// tsne := RunTSNE(MMatrix, perplexity, iterations)
	// // Print cell numbers and dimension
	// fmt.Println(tsne.RawMatrix().Rows, tsne.RawMatrix().Cols)
	// // plot mat.Dense after tSNE
	// err1 := PlotTSNE(tsne, "tsne.png")
	// if err1 != nil {
	// 	panic(err1)
	// }

	// // =================== PCA ========================
	// proj := RunPCA(MMatrix, 2)
	// _ = proj // just so we can graph/plot it later on.
	// PlotPCA2D(proj, "pca.png")

	// // ================ PEARSON NORMALIZATION ==================
	// em, numCells, numGenes := BuildMatrix(filtered)

	// normalized := em.Pearson(filtered, numCells, numGenes, 100)

	// // print first 5
	// normalized.PearsonSummarize(10)

}
