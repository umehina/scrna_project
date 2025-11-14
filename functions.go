// functions.go

package main

import (
	"gonum.org/v1/gonum/mat"
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

// //Yinan Elise Zhu - 11/06/2025
// //implementing tSNE using Gonum

// func RunTSNE(data *mat.Dense, perplexity float64, iterations int) *mat.Dense {

// 	outDims := 2 //change as an input
// 	learningRate := 200.0

// 	verbose := true

// 	tsneModel := tsne.NewTSNE(outDims, perplexity, learningRate, iterations, verbose)

// 	embedding := tsneModel.EmbedData(data, nil)

// 	return embedding.(*mat.Dense)
// }

// //Yinan Elise Zhu - 11/06/2025, 11/13/2025
// //implementing UMAP using Gonum
// // RunUMAP runs UMAP dimensionality reduction on a *mat.Dense matrix
// // data: input matrix of shape (n_samples × n_features)
// // nNeighbors: number of neighboring points used in local approximations (5–50)
// // minDist: controls how tightly points are packed together ( 0.0–0.5)

// func RunUMAP(data *mat.Dense, nNeighbors int, minDist float64) *mat.Dense {

// 	// Basic sanity checks
// 	if data == nil {
// 		panic("RunUMAP: data matrix is nil")
// 	}

// 	// Define the number of output dimensions for visualization (typically 2D)
// 	outDims := 2 //change as an input

// 	// Create a new UMAP model using the chosen hyperparameters:
// 	//   outDims = target embedding dimension (2D)
// 	//   nNeighbors = number of neighboring points
// 	//   minDist = minimum distance between points in the embedding
// 	umapModel := umap.NewUMAP(outDims, nNeighbors, minDist)

// 	// Fit the model on the input data and obtain the 2D embedding
// 	//   data = input matrix (*mat.Dense)
// 	// The result is a *mat.Dense (n_samples × outDims) containing the 2D coordinates
// 	embedding := umapModel.EmbedData(data)

// 	// Return the computed low-dimensional embedding
// 	return embedding.(*mat.Dense)
// }
