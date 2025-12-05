// umap.go
// Amy Ji, Nov 28th, 2025
package main

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strconv"

	"gonum.org/v1/gonum/mat"
)

// SavePCAScoresForR writes any *mat.Dense to CSV for R UMAP.
// Even though the name says "PCA", this works for any embedding matrix.
// Output format:
//   cell,PC1,PC2,...,PCk
//   cell_0,x11,x12,...,x1k
//   cell_1,x21,x22,...,x2k
//   ...
func SavePCAScoresForR(X *mat.Dense, filename string) error {
	if X == nil {
		return fmt.Errorf("SavePCAScoresForR: input matrix is nil")
	}

	n, k := X.Dims()

	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer f.Close()

	w := bufio.NewWriter(f)
	defer w.Flush()

	// header: cell,PC1,PC2,...
	fmt.Fprint(w, "cell")
	for j := 0; j < k; j++ {
		fmt.Fprintf(w, ",PC%d", j+1)
	}
	fmt.Fprint(w, "\n")

	// rows
	for i := 0; i < n; i++ {
		// simple row label; later you can swap in real barcodes if you want
		fmt.Fprintf(w, "cell_%d", i)
		for j := 0; j < k; j++ {
			fmt.Fprintf(w, ",%g", X.At(i, j))
		}
		fmt.Fprint(w, "\n")
	}

	return nil
}

// RunRUMAP calls the Rscript and runs UMAP.
// scriptPath  : path to umap_script.R
// inputCSV    : CSV in the format produced by SavePCAScoresForR
// outputCSV   : where the UMAP embedding will be written
// nNeighbors  : UMAP n_neighbors
// minDist     : UMAP min_dist
// metric      : distance metric (e.g. "euclidean")
func RunRUMAP(scriptPath, inputCSV, outputCSV string, nNeighbors int, minDist float64, metric string) error {
	nnStr := strconv.Itoa(nNeighbors)
	mdStr := fmt.Sprintf("%g", minDist) // %g is fine for R's as.numeric

	cmd := exec.Command(
		"Rscript",
		scriptPath,
		inputCSV,
		outputCSV,
		nnStr,
		mdStr,
		metric,
	)

	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("running R UMAP failed: %v\nOutput:\n%s", err, string(out))
	}
	return nil
}

// LoadUMAPFromCSV converts the UMAP result CSV back into a mat.Dense.
//
// Expected CSV format (columns):
//   UMAP1,UMAP2,cell
// or more generally at least 3 columns where the first two are numeric UMAP dims.
func LoadUMAPFromCSV(filename string) (*mat.Dense, []string, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()

	r := csv.NewReader(f)
	header, err := r.Read()
	if err != nil {
		return nil, nil, err
	}

	// expect something like: UMAP1,UMAP2,cell
	if len(header) < 3 {
		return nil, nil, fmt.Errorf("expected at least 3 columns in UMAP CSV")
	}

	var data []float64
	var cells []string
	numCols := 2 // UMAP1, UMAP2

	for {
		rec, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, nil, err
		}
		if len(rec) < 3 {
			return nil, nil, fmt.Errorf("row with too few columns")
		}

		// parse UMAP1, UMAP2
		for j := 0; j < numCols; j++ {
			v, err := strconv.ParseFloat(rec[j], 64)
			if err != nil {
				return nil, nil, fmt.Errorf("parse float: %v", err)
			}
			data = append(data, v)
		}
		// cell name (3rd column)
		cells = append(cells, rec[2])
	}

	n := len(cells)
	emb := mat.NewDense(n, numCols, data) // convert csv back to mat.Dense object
	return emb, cells, nil
}

// runUMAPOnDense is a helper that:
//  1. saves X to CSV,
//  2. runs R UMAP,
//  3. loads the embedding,
//  4. plots it with PlotEmb.
// It assumes PipelineConfig has:
//   PCAScoresCSV, UMAPOutCSV, UMAPNeighbors, UMAPMinDist, UMAPMetric, UMAPPlot
func runUMAPOnDense(X *mat.Dense, cfg PipelineConfig) error {
    if X == nil {
        return fmt.Errorf("runUMAPOnDense: input matrix is nil")
    }

    fmt.Println("Saving matrix for UMAP...")
    if err := SavePCAScoresForR(X, cfg.PCAScoresCSV); err != nil {
        return fmt.Errorf("runUMAPOnDense: SavePCAScoresForR failed: %w", err)
    }

    fmt.Println("Running UMAP in R...")
    if err := RunRUMAP("umap_script.R", cfg.PCAScoresCSV, cfg.UMAPOutCSV,
        cfg.UMAPNeighbors, cfg.UMAPMinDist, cfg.UMAPMetric); err != nil {
        return fmt.Errorf("runUMAPOnDense: RunRUMAP failed: %w", err)
    }

    emb, _, err := LoadUMAPFromCSV(cfg.UMAPOutCSV)
    if err != nil {
        return fmt.Errorf("runUMAPOnDense: LoadUMAPFromCSV failed: %w", err)
    }

    if err := PlotEmb(emb, cfg.UMAPPlot, "umap", "umap_1", "umap_2"); err != nil {
        return fmt.Errorf("runUMAPOnDense: PlotEmb failed: %w", err)
    }
    fmt.Println("UMAP plot saved as", cfg.UMAPPlot)
    return nil
}
