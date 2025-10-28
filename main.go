package main

import (
	"log"
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

	// perform normalization
	// normalized := filtered.Normalize()
	// normalized.Summary()
}
