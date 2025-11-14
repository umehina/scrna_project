// preprocessing.go

package main

import (
	"fmt"
	"math"
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

// Vania Halim - 11/1/2025
// LogNormalize is a CountMatrix method that takes a scaleFactor float64 as input and modifies the CountMatrix in place, log normalizing each feature count
func (cm *CountMatrix) LogNormalize(scaleFactor float64) {

	// range through each row (cell) in the counts matrix
	for _, cell := range cm.cells {
		// calculate total # features for that cell
		totalCount := cell.CountTotalGenes()

		// range through every feature/gene in the cell
		for feature, count := range cell.features {
			// scale the feature count and log normalize
			norm := count / totalCount * scaleFactor
			logNorm := math.Log1p(norm)

			cell.features[feature] = logNorm
		}
	}

}

// Vania Halim - 11/1/2025
// CountTotalGenes is a *Cell method that returns the total feature count (genes) for the input cell
func (cell *Cell) CountTotalGenes() float64 {

	var sum float64

	for _, count := range cell.features {
		sum += count
	}

	return sum

}

// TODO!
// Create a method that returns the total per gene count

// FindAllGenes returns a list of all genes as strings for a given CountMatrix
// Vania Halim - 11/1/2025
func (cm *CountMatrix) FindAllGenes() []string {

	genes := make([]string, 0)

	// assumes that the first cell contains a count for all genes
	for gene := range cm.cells[0].features {
		genes = append(genes, gene)
	}

	return genes

}

// ===================== PEARSON RESIDUALS NORMALIZATION =========================

// TotalGeneCounts returns the total number of observed counts for a given gene in a CountMatrix
// Vania Halim - 11/13/2025
func (em *ExpressionMatrix) GeneTotals() []int {
	// create output table of total count for that gene
	numGenes := len(em.data[0]) // number of columns = # genes
	numCells := len(em.data)    // number of rows = # cells

	geneTotals := make([]int, numGenes)

	// for each column, range through every row and save it in geneTotals
	for gene := 0; gene < numGenes; gene++ {
		sum := 0.0
		for cell := 0; cell < numCells; cell++ {
			sum += em.data[cell][gene]
		}
		geneTotals[gene] = int(sum)
	}

	return geneTotals
}

// buildMatrix constructs an ExpressionMatrix from a CountMatrix. it puts cells as rows and genes as columns.
// Qinglin Kong - 11/15/2025
func buildMatrix(cm *CountMatrix) ExpressionMatrix {
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

	return ExpressionMatrix{data: data, genes: genes}
}

// Tot

// ===============================================================================

// Summary prints basic statistics of the CountMatrix.
// Qinglin Kong - 10/28/2025
// Input: none
// Output: none
func (cm *CountMatrix) Summary() {
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
