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
func cmToDense(cm *CountMatrix) mat.Matrix {
	// findAllGenes is a countMatrix attribute (we defined this in preprocessing.go)
	genes := cm.findAllGenes() // this returns a []string, in which each string is a gene name. 

	nCells := len(cm.cells)
	nGenes := len(genes)

	//mat.NewDense is a function in Gonum/mat.
	data := mat.NewDense(nCells,nGenes,nil)

	// Fill the matrix
	for i,cell := range cm.cells {
		for j,g := range genes {
			val := cell.features[g]
			// data.Set is a function in Gonum/mat
			data.Set(i,j,val)
		}
	}
	return data
}

//Yinan Elise Zhu - 11/06/2025 
//implementing tSNE using Gonum
// RunTSNE runs t-SNE on a *mat.Dense input

// RunTSNE runs t-SNE dimensionality reduction on a *mat.Dense matrix
// data: input matrix of shape (n_samples × n_features)
// perplexity: controls the balance between local vs global structure (usually 5–50)
// iterations: number of optimization steps (usually 1000–2000)
func RunTSNE(data *mat.Dense, perplexity float64, iterations int) *mat.Dense {

    // Define the number of output dimensions for visualization (typically 2D)
    outDims := 2 //change as an input 

    // Set the learning rate (step size for gradient descent)
    // A value around 200 works well for most datasets
    learningRate := 200.0

    // Set verbose = true to print progress information to the console
    verbose := true

    // Create a new t-SNE model using the chosen hyperparameters:
    //   outDims = target embedding dimension (2D)
    //   perplexity = neighborhood size (~number of nearest neighbors)
    //   learningRate = gradient descent step size
    //   iterations = how many times optimization runs
    //   verbose = print intermediate results
    tsneModel := tsne.NewTSNE(outDims, perplexity, learningRate, iterations, verbose)

    // Fit the model on the input data and obtain the 2D embedding
    //   data = input matrix (*mat.Dense)
    //   nil = placeholder for an optional callback (not needed here)
    // The result is a *mat.Dense (n_samples × outDims) containing the 2D coordinates
    embedding := tsneModel.EmbedData(data, nil)

    // Return the computed low-dimensional embedding
    return embedding.(*mat.Dense)
}


// Amy Ji - 11/05/2025
func 