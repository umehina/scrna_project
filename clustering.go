package main

import (
	"math"
	"sort"

	"gonum.org/v1/gonum/mat"
)

type Graph struct {
	Nodes int
	Edges map[int][]Edge
}

type Edge struct {
	To     int
	Weight float64 // smaller distance = stronger connection
}

type Neighbor struct {
	Index    int
	Distance float64
}

func (em *ExpressionMatrix) Cluster(k int, pcs *mat.Dense) {
	// First, we create a distance matrix based on the cells
	// Next, we connect each cell on the distance matrix to its K most similar cells using Leiden.

	// The KNN graph reflects the underlying topology of the expression data by representing dense regions with respect to expression space also as densely connected regions in the graph

	// The starting point is a singleton partition in which each node functions as its own community (a).
	// As a next step, the algorithm creates partitions by moving individual nodes from one community to another (b), which is refined afterwards to enhance the partitioning (c).
	// The refined partition is then aggregated to a network (d).
	// Subsequently, the algorithm moves again individual nodes in the aggregate network (e), until refinement does no longer change the partition (f).
	// All steps are repeated until the final clustering is created and partitions no longer change.

	// build a knn graph from the pcs and k
	// g := BuildKNNGraph(pcs, k)
	// un leiden clustering on the graph
	// communities := g.Leiden()

	// return communities
}

// workflow of this clustering

// DistanceMatrix takes as input the normalized and reduced data as a pointer to a mat.Dense object. It returns a Euclidean Distance Matrix between all cells as a *mat.Dense object
// Vania Halim 11/27/2025
func DistanceMatrix(data *mat.Dense) *mat.Dense {

	// initialize output distance matrix
	rows, cols := data.Dims()
	distMtx := mat.NewDense(rows, cols, nil)

	// calculate and store euclidean distance of each cell in distMtx
	for r := range rows {

		// initialize slice of euclidean distances for cell i
		cellDistance := make([]float64, cols)

		cellOne := data.RawRowView(r)

		for c := range cols {
			cellTwo := data.RawRowView(c)

			// calculate EuclideanDistance between two cells and add it to the cell's total cellDistance
			cellDistance[c] = Euclidean(cellOne, cellTwo)

		}

		// set the cell's euclidean distance
		data.SetRow(r, cellDistance)

	}

	return distMtx

}

// Euclidean takes as input two slices of floats, corresponding to the principal components of two different cells. It returns the Euclidean distance of the principal components between two cells. It is called within DistanceMatrix() to calculate the Euclidean distance between all cells.
// // Vania Halim 11/27/2025
func Euclidean(firstCell, secondCell []float64) float64 {

	// initialize output euclidean distance
	euclidean := 0.0

	// range through all the principal components
	for i := range firstCell {
		diff := firstCell[i] - secondCell[i]
		euclidean += (diff * diff)
	}

	return math.Sqrt(euclidean)
}

func BuildKNNGraph(distanceMtx *mat.Dense, k int) *Graph {

	rows, _ := distanceMtx.Dims()
	graph := &Graph{Nodes: rows} // output graph

	// range through each cell's distances
	for r := range rows {
		// create temporary neighbors slice
		n := make([]Neighbor, 0)
		row := distanceMtx.RawRowView(r)

		// range through columns
		for c := range row {

			if r == c { // skip self
				continue
			}

			// make neighbor
			neighbor := Neighbor{Index: c, Distance: row[c]}
			n = append(n, neighbor)

		}

		// after all cell's neighbors have been added, sort to find k smallest ones
		sort.Slice(n, func(a, b int) bool {
			return n[a].Distance < n[b].Distance
		})

		// assign k closest neighbors as edges in the graph
		for i := range k {
			edge := Edge{To: n[i].Index, Weight: n[i].Distance}
			graph.Edges[i] = append(graph.Edges[i], edge)
		}

	}

	return graph

}

// func (g *Graph) Leiden(resolution float64, maxIter int) []int

// func (g *Graph) InitSingletonPartition() []int
// func (g *Graph) ModularityGain(i int, community int, partition []int, resolution float64) float64
// func (g *Graph) MoveNodes(partition []int, resolution float64) []int
// func (g *Graph) RefineCommunities(partition []int) []int
// func (g *Graph) Aggregate(partition []int) *Graph
