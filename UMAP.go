package main

import(
    "fmt"
	"os"
    "os/exec"
	"bufio"
	"encoding/csv"
	"io"
	"strconv"
	"gonum.org/v1/gonum/mat"
)

// convert PCAResult.score to a csv file for R
func SavePCAScoresForR(p *PCAResult, filename string) error {
    if p == nil || p.scores == nil {
        return fmt.Errorf("SavePCAScoresForR: PCAResult or scores is nil")
    }
    n, k := p.scores.Dims()
    if len(p.barcodes) != n {
        return fmt.Errorf("barcode length (%d) != number of rows (%d)", len(p.barcodes), n)
    }

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
        fmt.Fprintf(w, "%s", p.barcodes[i]) // row name
        for j := 0; j < k; j++ {
            fmt.Fprintf(w, ",%g", p.scores.At(i, j))
        }
        fmt.Fprint(w, "\n")
    }

    return nil
}

// RunRUMAP calls the Rscript and run UMAP.
func RunRUMAP(scriptPath, inputCSV, outputCSV string)error{
	// execute the R file "umap_script.R"
	// specify inputCSV and outputCSV
    cmd := exec.Command("Rscript", scriptPath, inputCSV, outputCSV)
    
    out, err := cmd.CombinedOutput()
    if err != nil {
        return fmt.Errorf("running R UMAP failed: %v\nOutput:\n%s", err, string(out))
    }
	return nil
}

// Convert UMAP result from csv back to mat.Dense
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
        // cell name
        cells = append(cells, rec[2])
    }

    n := len(cells)
    emb := mat.NewDense(n, numCols, data) // convert csv back to mat.Dense object
    return emb, cells, nil
}