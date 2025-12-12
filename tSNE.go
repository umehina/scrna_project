// tSNE.go
// Amy Ji - 12/04/2025

package main
/*
import (
	"fmt"
	//"gonum.org/v1/gonum/floats"
	"github.com/e-gun/go-tsne/tsne"
	"gonum.org/v1/gonum/mat"
	//"gonum.org/v1/gonum/stat"
)
type TSNEResult struct {
	scores *mat.Dense
}

// Run t-SNE on any dense matrix X and (optionally) plot it.
func runTSNEOnDense(X *mat.Dense, cfg PipelineConfig) error {
    if X == nil {
        return fmt.Errorf("runTSNEOnDense: input matrix is nil")
    }

    fmt.Println("Running t-SNE...")
    t := tsne.NewTSNE(cfg.TSNEDims, cfg.TSNEPerplexity, cfg.TSNETheta, cfg.TSNEIter, false)
    Y := t.EmbedData(X, nil).(*mat.Dense)

    if err := PlotEmb(Y, cfg.TSNEPlot, "tsne", "tsne_1", "tsne_2"); err != nil {
        return fmt.Errorf("runTSNEOnDense: PlotEmb failed: %w", err)
    }
    fmt.Println("t-SNE plot saved as", cfg.TSNEPlot)
    return nil
}
*/