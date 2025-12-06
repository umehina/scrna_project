package main

import (
	"fmt"
	"log"
	"os/exec"
)

/*
list of input arguments:
filename: "data/scRNA_dataset.csv"

minFeatures, maxFeatures, minCounts, max Counts, percentMT: [200, 2500, 500, 5000, 0.05]

NormalizationType: [Pearson, Log1p]

(tSNE) perplexity, iterations: 30.0, 1000

(pearson) theta: 0.5

PCs : 2

*/

func main() {
	// placeholder values
	pearsonTheta := 100.0
	normalizeMethod := "pearson"
	pcPlot := 30

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
	emv, _, _ := BuildMatrix(filtered)
	em := &emv
	fmt.Println(" done.")

	// normalize based on chosen method

	fmt.Print("Normalizing...")
	if normalizeMethod == "pearson" {
		em = em.Pearson(pearsonTheta)
	} else if normalizeMethod == "log1p" {
		em.Log1p()
	}

	fmt.Println("Done Normalizing. Scaling Normalized Data...")
	em.ScaleData(10)

	// run PCA and plot results
	fmt.Println("Done Scaling.. Running PCA")
	pcs := em.PCA(pcPlot)
	fmt.Println("Done PCA. PC Variances:", pcs.variances)

	fmt.Println("Plotting PCA...")
	err = pcs.PlotPCAScatter("pca.png", 0, 1)
	if err != nil {
		panic(err)
	}
	fmt.Println("PCA plot saved as pca.png")

	// --- Leiden clustering test and visualization ---
	// Use the PCA scores as input to build KNN and run Leiden (works on a copy internally)
	// Defaults: k=10, maxIter=10, resolution=0.8, gamma=0.5, theta=0.01
	fmt.Println("Running Leiden clustering test and exporting to R...")
	// get the PCA scores (2D embeddings) to use as coordinates for visualization
	pcaScores := pcs.GetScores()

	// also get the full scaled data for distance calculations
	dataDense := em.ToDense()
	if dataDense == nil {
		log.Printf("data dense is nil")
	} else {
		k := 5
		maxIter := 10
		resolution := 1.0 // standard resolution
		gamma := 1.0
		theta := 0.01

		fmt.Println("Running Leiden on full data...")
		g, clusters := RunLeiden(dataDense, k, maxIter, resolution, gamma, theta)

		// Count unique clusters
		uniqueClusters := make(map[int]bool)
		for _, c := range clusters {
			uniqueClusters[c] = true
		}
		fmt.Printf("Leiden produced %d clusters from %d nodes\n", len(uniqueClusters), len(clusters))

		fmt.Println("Done with Leiden. Exporting graphs and PCA coordinates...")
		// export graph edges, cluster labels, and PCA coordinates into R/ for plotting
		ExportLeiden(g, clusters, pcaScores)

		fmt.Println("Done exporting. Running R Script...")
		// run the verification script to compare with Seurat
		cmd := exec.Command("Rscript", "R/verify_clustering.R")
		out, err := cmd.CombinedOutput()
		if err != nil {
			log.Printf("Rscript failed: %v\nOutput:\n%s", err, string(out))
		} else {
			fmt.Println("Clustering comparison generated successfully")
		}
	}

	/*
		//perform normalization
		sff := 10000.00
		// LogNormalize modifies 'filtered' in place and does not return a value, so call it directly.
		filtered.LogNormalize(sff)
		//filtered.Summary()
	*/

	/*
		// transform countMatrix to a *mat.Dense object
		MMatrix := CmToDense(filtered)

		// =================== TSNE ============================
		// Amy Ji - 11/13/2025
		// Define perplexity and iterations for RunTSNE
		perplexity := 30.0
		iterations := 1000
		tsne := RunTSNE(MMatrix, perplexity, iterations)

		// Print cell numbers and dimension
		fmt.Println(tsne.RawMatrix().Rows, tsne.RawMatrix().Cols)
		// plot mat.Dense after tSNE
		err1 := PlotTSNE(tsne, "tsne.png")

		if err1 != nil {
			panic(err1)
		}

		// =================== PCA ========================
		proj := RunPCA(MMatrix, 2)
		_ = proj // just so we can graph/plot it later on.
		PlotPCA2D(proj, "pca.png")

		// ================ PEARSON NORMALIZATION ==================
		em, numCells, numGenes := BuildMatrix(filtered)

		theta := 0.5
		normalized := em.Pearson(theta)

		// print first 5
		normalized.PearsonSummarize(10)

	*/

}
