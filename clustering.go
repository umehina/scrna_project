package main

import "gonum.org/v1/gonum/mat"

type Graph struct {
	Nodes int
	Edges map[int][]Edge
}

type Edge struct {
	To     int
	Weight float64 // smaller distance = stronger connection
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

// workflow of this clhstingering.d

// func DistanceMatrix(data *mat.Dense) *mat.Dense
// func EuclideanDistance(a, b []float64) float64

// func BuildKNNGraph(pcs *mat.Dense, k int) *Graph

// func (g *Graph) Leiden(resolution float64, maxIter int) []int

// func (g *Graph) InitSingletonPartition() []int
// func (g *Graph) ModularityGain(i int, community int, partition []int, resolution float64) float64
// func (g *Graph) MoveNodes(partition []int, resolution float64) []int
// func (g *Graph) RefineCommunities(partition []int) []int
// func (g *Graph) Aggregate(partition []int) *Graph
