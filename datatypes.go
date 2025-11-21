// datatypes.go

package main

// CountMatrix is a 2D slice of cells. Columns are genes and the row are cells
type CountMatrix struct {
	// ...
	cells []*Cell
	// ...
}

type Cell struct {
	// ...
	idx       int
	barcode   string
	features  map[string]float64 // genes
	qcMetrics *QCMetrics
	// ...
}

type QCMetrics struct {
	nFeatureRNA float64
	nCountRNA   float64
	percentMT   float64 // fraction 0..1
}

type ExpressionMatrix struct {
	data  [][]float64
	genes []string
}

type Neighbor struct {
	ID     int
	Weight float64 // similarity, not distance
}

type Node struct {
	ID        int
	Community int
	Neighbors []Neighbor
}

type Graph struct {
	Nodes []Node
}
