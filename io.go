// parser.go
package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strconv"
)

// ParseCountMatrix reads a count matrix CSV where each row is a cell (barcode)
// and each column (after the first) is a gene name.
func ParseCountMatrix(filename string) (*CountMatrix, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %v", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	reader.TrimLeadingSpace = true

	// Read header (gene names)
	header, err := reader.Read()
	if err != nil {
		return nil, fmt.Errorf("failed to read header: %v", err)
	}
	if len(header) < 2 {
		return nil, fmt.Errorf("expected at least one gene column")
	}

	genes := header[1:] // skip the first column (barcode)

	var countMatrix CountMatrix
	cellIndex := 0

	// Read each row = one cell
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

		for i, val := range record[1:] {
			if val == "" {
				continue
			}
			f, convErr := strconv.ParseFloat(val, 64)
			if convErr != nil {
				return nil, fmt.Errorf("invalid number at cell %s, gene %s: %v", barcode, genes[i], convErr)
			}
			count := int(f)
			features[genes[i]] = count
			totalCount += count
		}

		qc := &QCMetrics{
			nFeatureRNA: len(features),
			nCountRNA:   totalCount,
			percentMT:   0.0, // placeholder — compute later if MT genes known
		}

		cell := &Cell{
			idx:       cellIndex,
			barcode:   barcode,
			features:  features,
			qcMetrics: qc,
		}

		countMatrix.cells = append(countMatrix.cells, cell)
		cellIndex++
	}

	return &countMatrix, nil
}

