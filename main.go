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
	pcPlot := 30    // For visualization only
	pcCluster := 30 // For clustering (match Seurat)

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
		em = em.Pearson(filtered, pearsonTheta)
	} else if normalizeMethod == "log1p" {
		em.Log1p()
	}

	fmt.Println("Done Normalizing. Scaling Normalized Data...")
	em.ScaleData(10)

	// run PCA for plotting (2 components)
	fmt.Println("Done Scaling.. Running PCA")
	pcsPlot := em.PCA(pcPlot)
	fmt.Println("Done PCA. PC Variances (first 2):", pcsPlot.variances)

	fmt.Println("Plotting PCA...")
	err = pcsPlot.PlotPCAScatter("pca.png", 0, 1)
	if err != nil {
		panic(err)
	}
	fmt.Println("PCA plot saved as pca.png")

	// --- Leiden clustering test and visualization ---
	// Compute PCA with more components for clustering (match Seurat)
	fmt.Println("Running Leiden clustering test and exporting to R...")
	fmt.Printf("Computing %d PCs for clustering...\n", pcCluster)
	pcsCluster := em.PCA(pcCluster)
	pcaScores := pcsCluster.GetScores()

	k := 20 // Match Seurat's k.param
	maxIter := 10
	resolution := 1.0 // standard resolution
	gamma := 1.0
	theta := 0.01

	fmt.Println("Running Leiden on PCA scores...")
	g, clusters := RunLeiden(pcaScores, k, maxIter, resolution, gamma, theta)

	// Count unique clusters
	uniqueClusters := make(map[int]bool)
	for _, c := range clusters {
		uniqueClusters[c] = true
	}
	fmt.Printf("Leiden produced %d clusters from %d nodes\n", len(uniqueClusters), len(clusters))

	fmt.Println("Done with Leiden. Exporting graphs and PCA coordinates...")
	// export graph edges, cluster labels, PCA coordinates, and parameters into R/ for plotting
	ExportLeiden(g, clusters, pcaScores, k, maxIter, resolution, gamma, theta)

	fmt.Println("Done exporting. Running verification script...")
	// run the verification script to compare with Seurat
	cmd := exec.Command("Rscript", "R/verify_clustering.R")
	out, err := cmd.CombinedOutput()
	if err != nil {
		log.Printf("Verification script failed: %v\nOutput:\n%s", err, string(out))
	} else {
		fmt.Println("Clustering comparison plot created successfully")
	}
}

/*
	//perform normalization
	sff := 10000.00
	// LogNormalize modifies 'filtered' in place and does not return a value, so call it directly.
	filtered.LogNormalize(sff)
	//filtered.Summary()
*/
