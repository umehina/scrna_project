// functions.go

package main

import (
	//"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"github.com/e-gun/go-tsne/tsne"

)

// Amy Ji - 11/05/2025
// cmToDense takes as input a countMatrix and convert it into mat.Matrix (*mat.Dense) object usable by many stat packages (i.e. pca,tSNE,UMAP)
// NOTE: it builds a *mat.Dense internally, but returns a mat.Matrix interface type. 
func cmToDense(cm *CountMatrix) *mat.Dense {
	// findAllGenes is a countMatrix attribute (we defined this in preprocessing.go)
	genes := cm.FindAllGenes() // this returns a []string, in which each string is a gene name.

	nCells := len(cm.cells)
	nGenes := len(genes)

	//mat.NewDense is a function in Gonum/mat.
	data := mat.NewDense(nCells, nGenes, nil)

	// Fill the matrix
	for i, cell := range cm.cells {
		for j, g := range genes {
			val := cell.features[g]
			// data.Set is a function in Gonum/mat
			data.Set(i, j, val)
		}
	}
	return data
}

//Yinan Elise Zhu - 11/06/2025 
//implementing tSNE using Gonum

func RunTSNE(data *mat.Dense, perplexity float64, iterations int) *mat.Dense {

    outDims := 2 //change as an input 
    learningRate := 200.0

    verbose := true

    tsneModel := tsne.NewTSNE(outDims, perplexity, learningRate, iterations, verbose)

    embedding := tsneModel.EmbedData(data, nil)

    return embedding.(*mat.Dense)
}

