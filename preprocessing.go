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

