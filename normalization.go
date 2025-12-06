package main

import (
	"math"
)

// ============= Pearson Residuals Normalization ================

// TODO: make sure the outputted results make sense, do hand calculation
// TODO: change pearson summary to output several rows

// Pearson takes as input a non-normalized ExpressionMatrix and the input countsMatrix
// returns a pointer to a new ExpressionMatrix where the values are the Pearson residuals of the observed counts
// Vania Halim 11/20/2025; Qinglin Kong - 11/30/2025
func (em *ExpressionMatrix) Pearson(theta float64) *ExpressionMatrix {
	numCells := len(em.data)
	if numCells == 0 {
		return em.InitializeEmptyCopy()
	}
	numGenes := len(em.genes)

	normalized := em.InitializeEmptyCopy()
	expected := em.Expected()

	// recommended clipping bound: ±sqrt(n_cells)
	clip := math.Sqrt(float64(numCells))

	// PR = (X_cg - u_cg)/sqrt(u_cg + (u_cg^2)/theta)
	for c := 0; c < numCells; c++ {
		for g := 0; g < numGenes; g++ {
			if c >= len(em.data) || g >= len(em.data[0]) {
				continue
			}
			if c >= len(expected.data) || g >= len(expected.data[0]) {
				continue
			}

			observedCount := em.data[c][g] // X_cg
			mu := expected.data[c][g]      // mu_cg

			denominator := math.Sqrt(mu + (mu*mu)/theta)
			if denominator == 0 {
				normalized.data[c][g] = 0
				continue
			}

			// clip residuals to [-clip, clip]
			z := (observedCount - mu) / denominator
			normalized.data[c][g] = clipValue(z, clip)
		}
	}

	return normalized
}

// Expected returns the expected value matrix for a given ExpressionMatrix
// It takes as input the total counts, computed from the countx matrix
// mu_cg = (n_c x T_g)T
// Vania Halim 11/20/2025; Qinglin Kong - 11/30/2025
func (em *ExpressionMatrix) Expected() *ExpressionMatrix {
	numCells := len(em.data)
	if numCells == 0 {
		return em.InitializeEmptyCopy()
	}
	numGenes := len(em.genes)

	// create output expression matrix with the same genes as em
	//expectedMatrix := &ExpressionMatrix{genes: em.genes}
	expectedMatrix := em.InitializeEmptyCopy()

	cellTotals := make([]float64, numCells)
	geneTotals := make([]float64, numGenes)

	var totalCounts float64

	// range through all cells: first pass to compute totals
	for c := range numCells {
		// range through all genes
		for g := range numGenes {
			x := em.data[c][g]
			cellTotals[c] += x
			geneTotals[g] += x

			// compute total counts overall
			totalCounts += x
		}
	}

	// return if all counts are zero to avoid division by zero
	if totalCounts == 0 {
		return expectedMatrix
	}

	// second pass: actually compute expected values
	for c := range numCells {
		rowTotal := cellTotals[c]
		for g := range numGenes {
			if c >= len(expectedMatrix.data) || g >= len(expectedMatrix.data[0]) {
				continue
			}
			// compute expected count of the cell and assign to current expectedMatrix input
			expectedMatrix.data[c][g] = (rowTotal * geneTotals[g]) / totalCounts
		}
	}

	return expectedMatrix
}

// ==================== Log Normalization =========================

// LogNormalize normalizes the ExpressionMatrix using log normalization with a given scale factor.
// This method first normalizes the data using counts per million (CPM) normalization and then applies the natural logarithm of one plus the value to each element.
// Vania Halim - 11/1/2025, Qinglin Kong - 11/14/2025
func (em *ExpressionMatrix) LogNormalize(scaleFactor float64) {
	em.NormalizeCPM(scaleFactor)
	em.Log1p()
}

// NormalizeCPM normalizes the ExpressionMatrix using counts per million normalization.
// this is basically Vania's initial LogNormalize method for CountMatrix without the log step.
// it is a reimplementation of Seurat's NormalizeData with normalization.method = "CPM" and scanpy's pp.normalize_total with target_sum=scaleFactor
// Vania Halim - 11/1/2025, Qinglin Kong - 11/14/2025
func (em *ExpressionMatrix) NormalizeCPM(scaleFactor float64) {
	numCells := len(em.data)
	if numCells == 0 {
		return // empty matrix, nothing to do
	}

	numGenes := len(em.data[0])
	cellTotals := em.CellTotals()

	// range through each cell
	for cell := 0; cell < numCells; cell++ {
		// get total count for that cell
		total := cellTotals[cell]
		row := em.data[cell]

		// avoid division by zero
		if total == 0 {
			for gene := 0; gene < numGenes; gene++ {
				row[gene] = 0
			}
			continue
		}

		// range through every feature in the cell
		factor := scaleFactor / total
		for gene := 0; gene < numGenes; gene++ {
			row[gene] *= factor
		}
	}
}

// Log1p applies the natural logarithm of one plus the value to each element in the ExpressionMatrix.
// Vania Halim - 11/1/2025, Qinglin Kong - 11/14/2025
func (em *ExpressionMatrix) Log1p() {
	for cell := range em.data {
		row := em.data[cell]
		for gene := range row {
			row[gene] = math.Log1p(row[gene])
		}
	}
}

// ScaleData standardizes the ExpressionMatrix by centering and scaling each gene (column). Each gene's values are transformed to have a mean of 0 and a standard deviation of 1. The method has a clip which the default is set to 10 based on Seurat's implementation.
// Qinglin Kong - 11/26/2025
func (em *ExpressionMatrix) ScaleData(clip float64) {
	// get num of cells/genes and return if empty
	numCells := len(em.data)
	if numCells == 0 {
		return
	}
	numGenes := len(em.data[0])
	if numGenes == 0 {
		return
	}

	// range through each gene (column)
	for g := 0; g < numGenes; g++ {
		// get all values for that gene
		vals := make([]float64, numCells)
		for c := 0; c < numCells; c++ {
			vals[c] = em.data[c][g]
		}

		// Compute mean and standard deviation of this gene
		mean := Mean(vals)
		sd := Std(Variance(vals, mean))

		// avoid division by zero
		if sd == 0 {
			// if sd is 0, set all values to 0, because they are all the same
			for c := 0; c < numCells; c++ {
				em.data[c][g] = 0
			}
			continue // move to next gene
		}

		for c := 0; c < numCells; c++ {
			// center and scale the value
			// first subtract the mean, then divide by sd
			z := (em.data[c][g] - mean) / sd
			// assign back to matrix
			em.data[c][g] = clipValue(z, clip)
		}
	}
}

// clipValue clips the input value to be within the range [-clip, clip]
// Qinglin Kong - 11/30/2025
func clipValue(z, clip float64) float64 {
	if z > clip {
		return clip
	} else if z < -clip {
		return -clip
	}
	return z
}
