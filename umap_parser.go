package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strconv"
)

// loadLeidenCoords reads a Leiden coords CSV of the form:
// node,PC1,PC2,...,PC30
// and returns:
//   data[i]   = []float64{PC1..PC30} for row i
//   nodeIDs[i] = original node ID (1-based in CSV)
func loadLeidenCoords(path string) (data [][]float64, nodeIDs []int, err error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, fmt.Errorf("open coords csv: %w", err)
	}
	defer f.Close()

	r := csv.NewReader(f)
	r.ReuseRecord = true

	// --- read header ---
	header, err := r.Read()
	if err != nil {
		return nil, nil, fmt.Errorf("read header: %w", err)
	}
	if len(header) < 2 {
		return nil, nil, fmt.Errorf("coords csv header has too few columns")
	}
	// header[0] should be "node", rest should be PC columns
	numPCs := len(header) - 1

	data = make([][]float64, 0)
	nodeIDs = make([]int, 0)

	// --- read rows ---
	for {
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, nil, fmt.Errorf("read record: %w", err)
		}
		if len(record) < numPCs+1 {
			// skip malformed line
			continue
		}

		// parse node ID (first column)
		nodeID, err := strconv.Atoi(record[0])
		if err != nil {
			return nil, nil, fmt.Errorf("parse node ID %q: %w", record[0], err)
		}

		// parse PC1..PCn
		pcs := make([]float64, numPCs)
		for j := 0; j < numPCs; j++ {
			v, err := strconv.ParseFloat(record[j+1], 64)
			if err != nil {
				return nil, nil, fmt.Errorf("parse PC col %d (%q): %w", j+1, record[j+1], err)
			}
			pcs[j] = v
		}

		nodeIDs = append(nodeIDs, nodeID)
		data = append(data, pcs)
	}

	return data, nodeIDs, nil
}

// writeEmbeddingCSV writes a CSV with columns:
// node,UMAP1,UMAP2,...,UMAPd
// where d = len(embedding[0]).
func writeEmbeddingCSV(path string, nodeIDs []int, embedding [][]float64) error {
	if len(nodeIDs) != len(embedding) {
		return fmt.Errorf("nodeIDs and embedding length mismatch: %d vs %d",
			len(nodeIDs), len(embedding))
	}
	if len(embedding) == 0 {
		return fmt.Errorf("empty embedding")
	}

	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create output csv: %w", err)
	}
	defer f.Close()

	w := csv.NewWriter(f)
	defer w.Flush()

	dim := len(embedding[0])

	// --- header ---
	header := make([]string, 0, dim+1)
	header = append(header, "node")
	for i := 0; i < dim; i++ {
		header = append(header, fmt.Sprintf("UMAP%d", i+1))
	}
	if err := w.Write(header); err != nil {
		return fmt.Errorf("write header: %w", err)
	}

	// --- rows ---
	for idx, coords := range embedding {
		if len(coords) != dim {
			return fmt.Errorf("embedding row %d has wrong dim: %d vs %d", idx, len(coords), dim)
		}
		row := make([]string, 0, dim+1)
		row = append(row, strconv.Itoa(nodeIDs[idx]))
		for d := 0; d < dim; d++ {
			row = append(row, fmt.Sprintf("%.6f", coords[d]))
		}
		if err := w.Write(row); err != nil {
			return fmt.Errorf("write row %d: %w", idx, err)
		}
	}

	return nil
}
