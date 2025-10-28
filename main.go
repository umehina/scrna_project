package main

import (
	"fmt"
	"log"
)

func main() {
	/* example usage (overall structure)
	countmatrix := LoadCountMatrix("data/scRNA_dataset.csv")

	cells := LoadCells(countmatrix) // however we ended up building the map

	AssignIndices(cells) // sets each cell.idx = its original position

	// apply QC filters
	indices := FilterCellIndices(cells, 200, 2500, 500, 15000, 0.05)
	filteredCells := FilterCells(cells, indices)

	// proceed with analysis on filteredCells
	normalizedCells := NormalizeCells(filteredCells)

	*/

	dataset, err := ParseCountMatrixFromFile("data/scRNA_dataset.csv")
	if err != nil {
		log.Fatalf("Error: %v", err)
	}

	indices := FilterCellIndices(dataset.cells, 200, 2500, 500, 15000, 0.05)
	filteredCells := FilterCells(dataset.cells, indices)

	c := filteredCells[234]
	fmt.Println("Number of Parsed cells:", len(dataset.cells))
	fmt.Println("Number of filtered cells:", len(filteredCells))
	fmt.Printf("Example cell:\n  Barcode:   %s\n  nFeatures: %d\n  nCounts:   %d\n  percentMT: %.2f%%\n",
		c.barcode,
		c.qcMetrics.nFeatureRNA,
		c.qcMetrics.nCountRNA,
		c.qcMetrics.percentMT)
}
