// preprocessing.go

package main

/* ----------
QC Filtering Functions

TODO: we should implement concurrency here if we want to speed this up for large datasets.
Maybe after we learn about this in class.
*/

// FilterCells filter cells based on provided indices that identify which cells to keep.
// Qinglin Kong - 10/21/2025
// Input: slice of Cell pointers and slice of indices to keep
// Output: slice of Cell pointers corresponding to the provided indices
func FilterCells(cells []*Cell, indices []int) []*Cell {
	filtered := make([]*Cell, 0, len(indices))
	for _, idx := range indices {
		// skip out-of-bounds indices
		if idx < 0 || idx >= len(cells) {
			continue
		}

		// append the cell at this index to the filtered slice
		c := cells[idx]

		// skip nil cells
		// (?) i guess at this point we can even create a new func that checks for nil cells
		if c == nil {
			continue
		}

		filtered = append(filtered, c)
	}
	return filtered
}

// FilterCellIndices gets a list of cell indices that pass the QC filters.
// Qinglin Kong - 10/21/2025
// Input: slice of Cell pointers and QC thresholds
// Output: slice of indices of cells that pass the filters
func FilterCellIndices(cells []*Cell, minFeatures, maxFeatures, minCounts, maxCounts int, maxPercentMT float64) []int {
	indices := make([]int, 0, len(cells))

	// iterate through every cell and check if it passes the filters
	for _, c := range cells {
		if c == nil {
			continue // skip nil cells
		}

		// get the metrics for this cell
		m := c.qcMetrics
		if m.nFeatureRNA >= minFeatures &&
			m.nFeatureRNA <= maxFeatures &&
			m.nCountRNA >= minCounts &&
			m.nCountRNA <= maxCounts &&
			m.percentMT <= maxPercentMT {
			indices = append(indices, c.idx)
		}
	}
	return indices
}

// CalcQCMetrics is a Cell method that calculates QC metrics for the cell.
// Qinglin Kong - 10/21/2025
// Input: a pointer to a Cell struct
// Output: none (it updates the QCMetrics field of the Cell struct)
func (c *Cell) CalcQCMetrics() {

	// if cell is nil, do nothing
	if c == nil {
		return
	}

	// if features map is nil, set qcMetrics to zero values
	if c.features == nil {
		c.qcMetrics = QCMetrics{}
		return
	}

	// then we iterate through the features map for this cell to calculate the metrics
	var nFeature, nCount, mtCount int
	for gene, count := range c.features {
		if count == 0 {
			continue
		}

		nFeature++
		nCount += count

		if isMTGene(gene) {
			mtCount += count
		}
	}

	// apply the calculated metrics to the Cell struct
	c.qcMetrics.nFeatureRNA = nFeature
	c.qcMetrics.nCountRNA = nCount
	if nCount > 0 {
		c.qcMetrics.percentMT = float64(mtCount) / float64(nCount) // fraction 0..1
	} else {
		c.qcMetrics.percentMT = 0
	}
}

// isMTGene checks if a gene is a mitochondrial gene based on its name prefix.
// Qinglin Kong - 10/21/2025
// Input: name string
// Output: bool that is true if the gene is mitochondrial
func isMTGene(name string) bool {
	// if the name is shorter than 3 characters, it cannot be MT- or mt-
	if len(name) < 3 {
		return false
	}

	// extract the first three characters
	b0 := name[0]
	b1 := name[1]
	b2 := name[2]

	// check if the name starts with "MT-" or "mt-" and return the result
	// this should be faster than strings.HasPrefix
	return (b0 == 'M' || b0 == 'm') && (b1 == 'T' || b1 == 't') && b2 == '-'
}

/* ----------
Normalization Functions
TODO: also implement concurrency here in the future
TODO: find out the math behind normalization logic
TODO: actually implement normalization logic
*/

// NormalizeCells normalizes the features of each cell in the input slice.
// Input: slice of Cell pointers
// Output: slice of normalized Cell pointers
func NormalizeCells(cells []*Cell) []*Cell {
	normalized := make([]*Cell, len(cells))

	for i, c := range cells {
		if c == nil {
			continue
		}

		normalized[i] = c.Normalize()
	}

	return normalized
}

// TODO: implement normalization logic
func (c *Cell) Normalize() *Cell {
	// placeholder function
	return c
}
