package main

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
func BuildMatrix(cm *CountMatrix) ExpressionMatrix {
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
