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
	rows, _ := data.Dims()
	distMtx := mat.NewDense(rows, rows, nil)

	// calculate and store euclidean distance of each cell in distMtx
	for r := range rows {

		// initialize slice of euclidean distances for cell i
		cellDistance := make([]float64, rows)

		cellOne := data.RawRowView(r)

		for c := range rows {
			cellTwo := data.RawRowView(c)

			// calculate EuclideanDistance between two cells and add it to the cell's total cellDistance
			cellDistance[c] = Euclidean(cellOne, cellTwo)

		}

		// set the cell's euclidean distance
		distMtx.SetRow(r, cellDistance)

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

// BuildKNNGraph takes as input a distance matrix as a pointer to a mat.Dense object and an integer k representing the number of nearest neighbors to connect each cell to. It returns a pointer to a Graph object representing the KNN graph constructed from the distance matrix.
// // Vania Halim 11/27/2025
func BuildKNNGraph(distanceMtx *mat.Dense, k int) *Graph {

	rows, _ := distanceMtx.Dims()
	graph := &Graph{Nodes: rows, Edges: make(map[int][]Edge)} // output graph

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
			graph.Edges[r] = append(graph.Edges[r], edge)
		}

	}

	return graph

}

// Leiden is a method of the *Graph object. It applies the Leiden algorithm to the KNN graph.
// Input: a resolution parameter as a decimal and a maximum number of iterations as an integer.
// Output: a slice of integers representing the communities that each original node is assigned to after Leiden is applied.
// Vania Halim 11/28/2025
func (g *Graph) Leiden(resolution float64, maxIter int) []int {

	// initialize clusters with each node in its community
	clusters := g.InitSingletonPartition()

	for i := range maxIter {
		// create copy of clusters to compare to the new one
		old := Copy(clusters)

		// move all nodes until clusters reaches local convergence
		clusters = g.MoveNodes(clusters, resolution)
		clusters = g.Refine(clusters)

		// if clusters have converged globally, return clusters
		if Compare(old, clusters) {
			return clusters
		}

		// else aggregate and repeat
		g = g.Aggregate(clusters)
		clusters = g.InitSingletonPartition()
	}

	return clusters

}

func (g *Graph) InitSingletonPartition() []int {

	partition := make([]int, g.Nodes)

	// put each node in its own partition
	for i := range g.Nodes {
		partition[i] = i
	}

	return partition

}

// ModularityGain computes the change in modularity by moving node i into the given cluster for a given resolution.
// Vania Halim 11/28/2025
func (g *Graph) ModularityGain(i, cluster int, partition []int, resolution float64) float64 {

	var observed float64
	var ki float64
	var kj float64
	var total float64

	var edges []Edge // temp slice of edges

	// compute total weight
	for _, edges := range g.Edges {
		for _, e := range edges {
			total += e.Weight
		}

	}

	// compute observed: sum of outgoing edge weights from node i to the cluster
	edges = g.Edges[i]
	for _, e := range edges {
		if partition[e.To] == cluster {
			observed += e.Weight
		}
	}

	// compute kj: sum of outgoing edge weights from nodes in cluster
	for node, community := range partition {

		// if the node is not in the community cluster then skip it
		if community != cluster {
			continue
		}

		// else explore the edge weights of that node
		edges = g.Edges[node]
		for _, e := range edges {
			kj += e.Weight
		}

	}

	// compute ki : sum of outgoing edge weights from node i
	for _, e := range g.Edges[i] {
		ki += e.Weight
	}

	// compute deltaQ
	expected := (ki * kj) / total
	deltaQ := observed - (resolution * expected)

	return deltaQ
}

// func (g *Graph) MoveNodes(partition []int, resolution float64) []int
// func (g *Graph) Refine(partition []int) []int
// func (g *Graph) Aggregate(partition []int) *Graph
// func Compare(oldPartition, newPartition []int) bool
