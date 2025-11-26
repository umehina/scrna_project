package main

import (
	"fmt"
	"math"
)

// PearsonSummarize takes as input a number of rows as integer
// It prints the first n genes for the first cell of the normalized *ExpressionMatrix
// Vania Halim - 11/20/2025
func (em *ExpressionMatrix) PearsonSummarize(n int) {
	// Defensive checks to avoid panics
	if len(em.data) == 0 || len(em.genes) == 0 {
		return
	}
	if len(em.data[0]) == 0 {
		return
	}

	fmt.Println("Pearson Residuals Summary:")

	maxGenes := len(em.genes)
	maxData := len(em.data[0])
	limit := n
	if limit > maxGenes {
		limit = maxGenes
	}
	if limit > maxData {
		limit = maxData
	}

	for g := 0; g < limit; g++ {
		fmt.Println("-> ", em.genes[g], "count: ", em.data[0][g])
	}
}

// InitializeExpressionMatrix takes as input the number of cells and genes as an integer
// Vania Halim - 11/20/2025
func InitializeExpressionMatrix(numCells, numGenes int) *ExpressionMatrix {

	// initialize output ExpressionMatrix

	data := make([][]float64, numCells)

	for i := range data {
		data[i] = make([]float64, numGenes)
	}

	return &ExpressionMatrix{data: data}

}

// TotalCounts is a *CountMatrix Method that returns the total number of observed counts for all cells in the matrix
// Vania Halim - 11/20/2025
func (cm *CountMatrix) TotalCounts() float64 {

	var totalCount float64

	for _, currCell := range cm.cells {
		totalCount += currCell.qcMetrics.nCountRNA
	}

	return totalCount

}

// countTotalGenes is a *Cell method that returns the total feature count (genes) for the input cell
// Vania Halim - 11/1/2025
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

// GeneTotals returns the total number of observed counts for a given gene in a CountMatrix
// Vania Halim - 11/13/2025
func (em *ExpressionMatrix) GeneTotals() []float64 {
	numCells := len(em.data) // number of rows = # cells
	if numCells == 0 {
		return nil // empty matrix, nothing to do
	}
	// create output table of total count for that gene
	numGenes := len(em.data[0]) // number of columns = # genes
	geneTotals := make([]float64, numGenes)

	// for each column, range through every row and save it in geneTotals
	for gene := 0; gene < numGenes; gene++ {
		sum := 0.0
		for cell := 0; cell < numCells; cell++ {
			sum += em.data[cell][gene]
		}
		geneTotals[gene] = sum
	}

	return geneTotals
}

// CellTotals returns the total number of observed counts for a given cell in an ExpressionMatrix
// Qinglin Kong - 11/14/2025
func (em *ExpressionMatrix) CellTotals() []float64 {
	numCells := len(em.data)
	if numCells == 0 {
		return nil // empty matrix, nothing to do
	}
	numGenes := len(em.data[0])
	cellTotals := make([]float64, numCells)

	for cell := 0; cell < numCells; cell++ {
		sum := 0.0
		row := em.data[cell]
		for gene := 0; gene < numGenes; gene++ {
			sum += row[gene]
		}
		cellTotals[cell] = sum
	}

	return cellTotals
}

// buildMatrix constructs an ExpressionMatrix from a CountMatrix. it puts cells as rows and genes as columns.
// Qinglin Kong - 11/15/2025
func BuildMatrix(cm *CountMatrix) (ExpressionMatrix, int, int) {
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

	return ExpressionMatrix{data: data, genes: genes}, numCell, numGene
}

// Mean computes the mean of a slice.
func Mean(vals []float64) float64 {
	n := len(vals)
	if n == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range vals {
		sum += v
	}
	return sum / float64(n)
}

// Variance computes the population variance given a slice and its mean.
func Variance(vals []float64, mean float64) float64 {
	n := len(vals)
	if n == 0 {
		return 0
	}
	sumsq := 0.0
	for _, v := range vals {
		sumsq += v * v
	}
	variance := sumsq/float64(n) - mean*mean
	if variance < 0 && variance > -1e-12 {
		variance = 0
	}
	return variance
}

// Std computes the standard deviation from variance.
func Std(variance float64) float64 {
	return math.Sqrt(variance)
}
