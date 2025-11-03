// parser.go
package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strconv"
)

// ParseCountMatrixFromFile reads a CSV file and returns a CountMatrix.
func ParseCountMatrixFromFile(filename string) (*CountMatrix, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %v", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	reader.TrimLeadingSpace = true

	return parseCountMatrix(reader)
}

// parseCountMatrix reads a CSV where each row = cell barcode,
// and columns = genes (float64 counts converted to int).
func parseCountMatrix(reader *csv.Reader) (*CountMatrix, error) {
	// read header row
	header, err := parseHeader(reader)
	if err != nil {
		return nil, err
	}
	genes := header[1:]

	var dataset CountMatrix
	cellIdx := 0

	// each for loop iteration reads one row (one cell)
	for {
		// read the next record
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("error reading row: %v", err)
		}

		// call helper to parse this record into a Cell struct
		cell, err := parseCellRecord(record, genes, cellIdx)
		if err != nil {
			return nil, err
		}

		// append the cell to the dataset and increment index
		dataset.cells = append(dataset.cells, cell)
		cellIdx++
	}

	return &dataset, nil
}

// parseHeader reads the header row from a CSV reader and returns the gene names.
// Input: a pointer to a csv.Reader
// Output: slice of gene names and error if any
func parseHeader(reader *csv.Reader) ([]string, error) {
	header, err := reader.Read()
	if err != nil {
		return nil, fmt.Errorf("failed to read header: %v", err)
	}
	if len(header) < 2 {
		return nil, fmt.Errorf("expected at least one gene column")
	}
	return header, nil
}

// parseCellRecord parses a single cell record from CSV and returns a Cell struct.
// Input: slice of strings (CSV row), slice of gene names, cell index
// Output: pointer to Cell struct and error if any
func parseCellRecord(record []string, genes []string, idx int) (*Cell, error) {
	if len(record) != len(genes)+1 {
		return nil, fmt.Errorf("row length mismatch: expected %d, got %d", len(genes)+1, len(record))
	}

	barcode := record[0]
	features := make(map[string]float64)
	totalCount := 0.0

	for i, val := range record[1:] {
		if val == "" {
			continue
		}
		count, err := strconv.ParseFloat(val, 64)
		if err != nil {
			return nil, fmt.Errorf("invalid number at cell %s, gene %s: %v", barcode, genes[i], err)
		}
		if count > 0 {
			features[genes[i]] = count
		}
		totalCount += count
	}

	cell := &Cell{
		idx:      idx,
		barcode:  barcode,
		features: features,
		qcMetrics: &QCMetrics{
			nFeatureRNA: float64(len(features)),
			nCountRNA:   totalCount,
		},
	}

	cell.calcPercentMT()
	return cell, nil
}

// calcPercentMT is a Cell method that calculates the percentage of mitochondrial gene counts.
// Qinglin Kong - 10/21/2025
// Input: a pointer to a Cell struct
// Output: none (it updates the QCMetrics field of the Cell struct)
func (c *Cell) calcPercentMT() {
	if c == nil || c.features == nil || c.qcMetrics == nil {
		return
	}
	// calculate and set the percentMT field
	c.qcMetrics.percentMT = calcMTFraction(c.features, c.qcMetrics.nCountRNA)
}

// calcMTFraction calculates the fraction of mitochondrial gene counts.
// Qinglin Kong - 10/21/2025
// Input: features map and total counts nCount
// Output: float64 fraction of mitochondrial gene counts
func calcMTFraction(features map[string]float64, nCount float64) float64 {
	if nCount == 0 {
		return 0
	}
	mtCount := 0.0
	for gene, count := range features {
		if count > 0 && isMTGene(gene) {
			mtCount += count
		}
	}
	return mtCount / nCount
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
