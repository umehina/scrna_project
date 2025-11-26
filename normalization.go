package main

import (
	"math"
)

// ===================== PEARSON RESIDUALS NORMALIZATION =========================

// TODO: make sure the outputted results make sense, do hand calculation
// TODO: change pearson summary to output several rows

// Pearson takes as input a non-normalized ExpressionMatrix and the input countsMatrix
// returns a pointer to a new ExpressionMatrix where the values are the Pearson residuals of the observed counts
// Vania Halim 11/20/2025
func (em *ExpressionMatrix) Pearson(cm *CountMatrix, numCells, numGenes int, theta float64) *ExpressionMatrix {
	// initialize output ExpressionMatrix
	//normalized := &ExpressionMatrix{genes: em.genes}
	normalized := InitializeExpressionMatrix(numCells, numGenes)
	normalized.genes = em.genes

	// create the expected value matrix
	expected := em.Expected(cm, numCells, numGenes)

	// range through the input and expected matrix, updating the output matrix
	// PR = (X_cg - u_cg)/sqrt(u_cg + (u_cg^2)/theta)
	for c := range numCells {
		for g := range numGenes {

			// bounds safety checks
			if c >= len(em.data) || g >= len(em.data[0]) {
				continue
			}
			if c >= len(expected.data) || g >= len(expected.data[0]) {
				continue
			}

			observedCount := em.data[c][g]
			expectedCount := expected.data[c][g]

			numerator := observedCount - expectedCount
			denominator := math.Sqrt(expectedCount + ((expectedCount * expectedCount) / theta))

			// avoid division by zero
			if denominator == 0 {
				normalized.data[c][g] = 0
			} else {
				normalized.data[c][g] = numerator / denominator
			}

		}
	}

	return normalized
}

// Expected returns the expected value matrix for a given ExpressionMatrix
// It takes as input the total counts, computed from the countx matrix
// mu_cg = (n_c x T_g)T

func (em *ExpressionMatrix) Expected(cm *CountMatrix, numCells, numGenes int) *ExpressionMatrix {

	geneTotals := em.GeneTotals() // list of gene totals for all cells
	totalCounts := cm.TotalCounts()

	// create output expression matrix with the same genes as em
	//expectedMatrix := &ExpressionMatrix{genes: em.genes}
	expectedMatrix := InitializeExpressionMatrix(numCells, numGenes)
	expectedMatrix.genes = em.genes

	// range through all cells
	for c := range numCells {
		// range through all genes
		for g := range numGenes {

			// compute expected count of the cell and assign to current expectedMatrix input
			if c >= len(cm.cells) || g >= len(geneTotals) {
				continue
			}

			currCell := cm.cells[c]
			cellTotal := currCell.qcMetrics.nCountRNA
			geneTotal := geneTotals[g]

			expectedMatrix.data[c][g] = (cellTotal * geneTotal) / totalCounts
		}
	}

	return expectedMatrix

}

// ==================== Log Normalization =========================

// LogNormalize is a CountMatrix method that takes a scaleFactor float64 as input and modifies the CountMatrix in place, log normalizing each feature count
// Vania Halim - 11/1/2025
func (cm *CountMatrix) LogNormalize(scaleFactor float64) {
	// range through each row (cell) in the counts matrix
	for _, cell := range cm.cells {
		// calculate total # features for that cell
		totalCount := cell.CountTotalGenes()

		// if totalCount is 0, skip to avoid division by zero
		if totalCount == 0 {
			for feature := range cell.features {
				cell.features[feature] = 0.0
			}
			continue
		}

		// range through every feature/gene in the cell
		for feature, count := range cell.features {
			// scale the feature count and log normalize
			norm := count / totalCount * scaleFactor
			cell.features[feature] = math.Log1p(norm)
		}
	}
}

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

// Scale standardizes the ExpressionMatrix by scaling each gene to have mean 0 and standard deviation 1, with optional clipping of extreme values.
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

		// compute mean and sd
		mean, sd := meanSD(vals)

		// avoid division by zero
		if sd == 0 {
			// if sd is 0, set all values to 0, because they are all the same
			for c := 0; c < numCells; c++ {
				em.data[c][g] = 0
			}
			continue // move to next gene
		}

		// go through each cell again and scale the value
		for c := 0; c < numCells; c++ {
			// center and scale the value
			// first subtract the mean, then divide by sd
			v := (em.data[c][g] - mean) / sd

			// optional clipping to handel extreme outliers.
			if clip > 0 {
				if v > clip {
					v = clip
				} else if v < -clip {
					v = -clip
				}
			}

			// assign back to matrix
			em.data[c][g] = v
		}
	}
}

// ============== Pearson Residuals Normalization =================

// TODO:

// ==== Deprecated: Negative Binomial Regression Normalization ====

// // Fit a simple log-linear regression: log(y+1) = β0 + β1 * log(s)
// func fitGeneCoefficients(values []float64, depth []float64) (float64, float64) {
// 	x := make([]float64, len(values))
// 	logValues := make([]float64, len(values))

// 	// populate x and logValues slices
// 	for i := range values {

// 		// safety for small values to avoid log(0)
// 		if depth[i] <= 0 {
// 			depth[i] = 1e-6
// 		}

// 		// what ?
// 		x[i] = math.Log(depth[i])
// 		logValues[i] = math.Log(values[i] + 1)
// 	}

// 	meanX, meanY := mean(x), mean(logValues)

// 	var num, den float64

// 	//
// 	for j := range values {
// 		num += (x[j] - meanX) * (logValues[j] - meanY)
// 		den += (x[j] - meanX) * (x[j] - meanX)
// 	}

// 	beta1 := num / den
// 	beta0 := meanY - beta1*meanX
// 	return beta0, beta1
// }

// func mean(arr []float64) float64 {
// 	if len(arr) == 0 {
// 		return 0
// 	}
// 	sum := 0.0
// 	for _, v := range arr {
// 		sum += v
// 	}
// 	return sum / float64(len(arr))
// }

// func estimateTheta(counts []float64) float64 {
// 	n := float64(len(counts))
// 	if n < 2 {
// 		return 1.0
// 	}
// 	sum, sumsq := 0.0, 0.0
// 	for _, x := range counts {
// 		sum += x
// 		sumsq += x * x
// 	}
// 	mean := sum / n
// 	variance := sumsq/n - mean*mean
// 	if variance <= mean {
// 		return 1e6 // ~Poisson
// 	}
// 	theta := mean * mean / (variance - mean)
// 	if theta < 1e-3 {
// 		theta = 1e-3
// 	}
// 	return theta
// }

// // computeSizeFactors calculates per-cell library size normalization factors.
// // s_j = n_j / median(total_counts)
// func (cm *CountMatrix) computeSizeFactors() []float64 {
// 	if cm == nil || len(cm.cells) == 0 {
// 		return nil
// 	}

// 	sizes := make([]float64, len(cm.cells))
// 	for i, c := range cm.cells {
// 		if c != nil && c.qcMetrics != nil {
// 			sizes[i] = float64(c.qcMetrics.nCountRNA)
// 		}
// 	}

// 	sorted := append([]float64(nil), sizes...)
// 	sort.Float64s(sorted)
// 	median := sorted[len(sorted)/2]
// 	if median == 0 {
// 		median = 1.0
// 	}

// 	for i := range sizes {
// 		sizes[i] /= median
// 	}

// 	return sizes
// }
