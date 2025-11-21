// functions.go

package main

import (
	"fmt"
	//"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
	"github.com/e-gun/go-tsne/tsne"


)

// CmToDense takes as input a CountMatrix and converts it into a mat.Matrix (*mat.Dense) object usable by many stat packages (i.e. PCA, tSNE, UMAP)
// NOTE: it builds a *mat.Dense internally, but returns a mat.Matrix interface type.
// Amy Ji - 11/05/2025
func CmToDense(cm *CountMatrix) *mat.Dense {
	// FindAllGenes is a CountMatrix method
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
// https://pkg.go.dev/github.com/e-gun/go-tsne/tsne
func RunTSNE(data *mat.Dense, perplexity float64, iterations int) *mat.Dense {
	if data == nil {
		panic("RunUMAP: data matrix is nil")
	}
    outDims := 2 //change as an input 
    learningRate := 200.0

// 	verbose := true

// 	tsneModel := tsne.NewTSNE(outDims, perplexity, learningRate, iterations, verbose)

// 	embedding := tsneModel.EmbedData(data, nil)

// 	return embedding.(*mat.Dense)
// }

// Amy Ji - 11/12/2025
// Implementing PCA using func PC in gonum.stats
func RunPCA(data *mat.Dense, k int) *mat.Dense {
	if data == nil {
		panic("RunPCA: data matrix is nil")
	}
	_,d := data.Dims()
	var pc stat.PC
	ok := pc.PrincipalComponents(data, nil)
	if !ok {
		panic("PCA computation failed")
	}
	fmt.Printf("variance=%.4f\n\n",pc.VarsTo(nil))

	// Get eigenvectors (loadings); 
	// vec is d*d eigenvectors in columns
	var vec mat.Dense 
	pc.VectorsTo(&vec)
	// Project data (n*d) onto the first k principle components.
	// d*k --> n*k
	var proj mat.Dense
	proj.Mul(data, vec.Slice(0,d,0,k))
	fmt.Printf("proj=\n%v\n", mat.Formatted(&proj, mat.Prefix(" ")))
	return   &proj
}




