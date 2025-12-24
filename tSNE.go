package main

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

// TSNE performs t-SNE on the PCAResult p, returning a TSNEResult.
func (p *PCAResult) TSNE(dimsOut int, perplexity, learningRate float64, maxIter int) (*TSNEResult, error) {
	if p == nil || p.scores == nil {
		return nil, fmt.Errorf("TSNE: PCAResult or score is nil")
	}

	// Create a new t-SNE object
	t := tsne.NewTSNE(dimsOut, perplexity, learningRate, maxIter, false)
	Y := t.EmbedData(p.scores, nil).(*mat.Dense)

	return &TSNEResult{
		scores: Y, //Y is a mat.Dense object
	}, nil

}
