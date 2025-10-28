// parser.go
package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strconv"
)

// ParseCountMatrix reads a CSV where each row = cell barcode,
// and columns = genes (float64 counts converted to int).
func ParseCountMatrix(filename string) (*CountMatrix, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %v", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	reader.TrimLeadingSpace = true

	// --- Header: Gene names ---
	header, err := reader.Read()
	if err != nil {
		return nil, fmt.Errorf("failed to read header: %v", err)
	}
	if len(header) < 2 {
		return nil, fmt.Errorf("expected at least one gene column")
	}
	genes := header[1:] // first column = barcode

	var dataset CountMatrix
	cellIndex := 0

	// --- Rows: Each cell ---
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("error reading row: %v", err)
		}

		if len(record) != len(header) {
			return nil, fmt.Errorf("row length mismatch: expected %d, got %d", len(header), len(record))
		}

		barcode := record[0]
		features := make(map[string]int)
		totalCount := 0

		// --- Fixed version of parsing loop ---
		for i, val := range record[1:] {
			if val == "" {
				continue
			}
			f, convErr := strconv.ParseFloat(val, 64)
			if convErr != nil {
				return nil, fmt.Errorf("invalid number at cell %s, gene %s: %v", barcode, genes[i], convErr)
			}
			count := int(f)
			if count > 0 {
				features[genes[i]] = count
			}
			totalCount += count
		}

		cell := &Cell{
			idx:      cellIndex,
			barcode:  barcode,
			features: features,
			qcMetrics: &QCMetrics{
				nFeatureRNA: 0,
				nCountRNA:   0,
				percentMT:   0.0,
			},
		}

		// Compute QC metrics for this cell
		cell.CalcQCMetrics()

		dataset.cells = append(dataset.cells, cell)
		cellIndex++
	}

	return &dataset, nil
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
		c.qcMetrics = &QCMetrics{}
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
