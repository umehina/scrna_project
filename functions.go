// functions.go

package main

import (
	//"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
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

// Amy Ji - 11/05/2025
func 