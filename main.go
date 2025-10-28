package main


import ("fmt"
		"log")

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

	dataset, err := ParseCountMatrix("data/scRNA_dataset.csv")
	if err != nil {
		log.Fatalf("Error: %v", err)
	}

	fmt.Printf("Parsed %d cells.\n", len(dataset.cells))
	fmt.Printf("Example cell: barcode=%s, nFeatures=%d, nCounts=%d\n",
		dataset.cells[0].barcode,
		dataset.cells[0].qcMetrics.nFeatureRNA,
		dataset.cells[0].qcMetrics.nCountRNA)

	indices := FilterCellIndices(dataset.cells, 200, 2500, 500, 15000, 0.05)
	//filteredCells := FilterCells(dataset.cells, indices)

	fmt.Println(len(indices))

}
