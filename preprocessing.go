// preprocessing.go

package main

import (
	"fmt"
)

/* ----------
QC Filtering Functions

TODO: implement concurrency here
*/

// (cm *CountMatrix) FilterByThresholds: the high-level QC filtering function that combines getting indices and filtering.
// Qinglin Kong - 10/28/2025
// Input: minFeatures, maxFeatures, minCounts, maxCounts, maxPercentMT
// Output: *CountMatrix with cells that pass the filters
func (cm *CountMatrix) FilterBy(minFeatures, maxFeatures, minCounts, maxCounts int, maxPercentMT float64) *CountMatrix {
	indices := cm.getIndicesBy(minFeatures, maxFeatures, minCounts, maxCounts, maxPercentMT)
	return cm.filterByIndices(indices)
}

// (cm *CountMatrix) filterByIndices: filters the CountMatrix based on provided indices.
// Qinglin Kong - 10/21/2025
// Input: slice of cell indices
// Output: *CountMatrix with cells at those indices
func (cm *CountMatrix) filterByIndices(indices []int) *CountMatrix {
	if cm == nil || len(cm.cells) == 0 {
		return &CountMatrix{}
	}

	// create a new CountMatrix to hold filtered cells
	filtered := &CountMatrix{
		cells: make([]*Cell, 0, len(indices)),
	}

	for _, idx := range indices {
		if idx < 0 || idx >= len(cm.cells) {
			continue // skip out-of-range indices
		}
		c := cm.cells[idx]
		if c == nil {
			continue // skip nil cells
		}
		// append cell to filtered CountMatrix
		filtered.cells = append(filtered.cells, c)
	}

	return filtered
}

// (cm *CountMatrix) getIndicesWithin: gets indices of cells that pass the QC filters.
// Qinglin Kong - 10/21/2025
// Input: minFeatures, maxFeatures, minCounts, maxCounts, maxPercentMT
// Output: slice of cell indices that pass the filters
func (cm *CountMatrix) getIndicesBy(minFeatures, maxFeatures, minCounts, maxCounts int, maxPercentMT float64) []int {
	if cm == nil || len(cm.cells) == 0 {
		return nil
	}

	// collect indices of cells that pass the filters
	indices := make([]int, 0, len(cm.cells))
	for _, c := range cm.cells {
		if c == nil || c.qcMetrics == nil {
			continue
		}

		q := c.qcMetrics
		if q.nFeatureRNA >= float64(minFeatures) && q.nFeatureRNA <= float64(maxFeatures) && q.nCountRNA >= float64(minCounts) && q.nCountRNA <= float64(maxCounts) && q.percentMT <= maxPercentMT {
			// cell passes all filters, add its index to the slice
			indices = append(indices, c.idx)
		}
	}
	return indices
}

/* ----------
Normalization Functions
TODO: also implement concurrency here
TODO: find out the math behind normalization logic
TODO: actually implement normalization logic
*/

// buildMatrix constructs an ExpressionMatrix from a CountMatrix. it puts cells as rows and genes as columns.
// Qinglin Kong - 11/15/2025
func (cm *CountMatrix) buildMatrix() *ExpressionMatrix {
	numCell := len(cm.cells)
	data := make([][]float64, numCell)

	genes := cm.FindAllGenes()
	numGene := len(genes)

	for i := 0; i < numCell; i++ {
		values := make([]float64, numGene)
		cell := cm.cells[i]

		for j, gene := range genes {
			values[j] = cell.features[gene]
		}

		data[i] = values
	}

	return &ExpressionMatrix{data: data, genes: genes}
}

// ===============================================================================
// Summary prints basic statistics of the CountMatrix.
// Qinglin Kong - 10/28/2025
// Input: none
// Output: none
func (cm *CountMatrix) ImportSummary() {
	if cm == nil || len(cm.cells) == 0 {
		fmt.Println("CountMatrix is empty.")
		return
	}

	totalCells := len(cm.cells)
	var totalFeatures float64
	var totalCounts float64

	for _, c := range cm.cells {
		if c == nil || c.qcMetrics == nil {
			continue
		}
		totalFeatures += c.qcMetrics.nFeatureRNA
		totalCounts += c.qcMetrics.nCountRNA
	}

	avgFeatures := float64(totalFeatures) / float64(totalCells)
	avgCounts := float64(totalCounts) / float64(totalCells)

	fmt.Println("CountMatrix Summary:")
	fmt.Println("-> Total Cells:", totalCells)
	fmt.Println("   -> Average Features per Cell:", int(avgFeatures))
	fmt.Println("   -> Average Counts per Cell:", int(avgCounts))
}
