package main

import (
	"math"
	"math/rand"
	"sort"

	"gonum.org/v1/gonum/mat"
)

type Graph struct {
	Nodes       int
	Edges       map[int][]Edge
	TotalWeight float64
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

// ==================== Building a Distance Matrix ====================

// DistanceMatrix takes as input the normalized and reduced data as a pointer to a mat.Dense object. It returns a Euclidean Distance Matrix between all cells as a *mat.Dense object
// Vania Halim 11/27/2025
func DistanceMatrix(data *mat.Dense) *mat.Dense {
	rows, _ := data.Dims()
	distMtx := mat.NewDense(rows, rows, nil)

	// calculate and store euclidean distance of each cell in distMtx
	for r := range rows {
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
	euclidean := 0.0
	// range through all the principal components
	for i := range firstCell {
		diff := firstCell[i] - secondCell[i]
		euclidean += (diff * diff)
	}
	return math.Sqrt(euclidean)
}

// ==================== Building a KNN Graph ====================

// BuildKNNGraph takes as input a distance matrix as a pointer to a mat.Dense object and an integer k representing the number of nearest neighbors to connect each cell to. It returns a pointer to a Graph object representing the KNN graph constructed from the distance matrix.
// Vania Halim 11/27/2025; Qinglin Kong 11/29/2025
func BuildKNNGraph(distanceMtx *mat.Dense, k int) *Graph {
	rows, _ := distanceMtx.Dims()

	if rows == 0 {
		return &Graph{Nodes: 0, Edges: make(map[int][]Edge)}
	}
	if k >= rows {
		k = rows - 1
	}
	if k <= 0 {
		return &Graph{Nodes: 0, Edges: make(map[int][]Edge)}
	}

	// compute directed KNN weight matrix)
	directed := fillDirectedKNNWeights(distanceMtx, k)

	// make an undirected graph by symmetrizing the weights
	edges, totalWeight := symmetrizeWeightsUsing(directed)

	// find total weight of all edges in the top half of edges only

	// build final Graph object to return
	graph := &Graph{Nodes: rows, Edges: edges, TotalWeight: totalWeight}
	return graph
}

// fillDirectedKNNWeights takes as input a distance matrix as a pointer to a mat.Dense object and an integer k representing the number of nearest neighbors to connect each cell to. It returns a *mat.Dense representing the directed KNN weights between nodes.
// Vania Halim 11/27/2025; Qinglin Kong 11/29/2025
func fillDirectedKNNWeights(distanceMtx *mat.Dense, k int) *mat.Dense {
	rows, _ := distanceMtx.Dims() // only need rows since distance matrix is square
	directed := mat.NewDense(rows, rows, nil)

	// for each node, find its k nearest neighbors and set the corresponding weights
	for r := range rows {
		row := distanceMtx.RawRowView(r)        // get the row corresponding to the current node
		allNeighbors := getNeighborsRow(row, r) // get all neighbors
		knn := topKNeighbors(allNeighbors, k)   // pick k nearest neighbors

		// assign k closest neighbors as edges in the graph
		for i := range len(knn) {
			neighbor := knn[i]
			weight := distanceToWeight(neighbor.Distance) // convert distance to weight

			// assign weight directly (no need to check for maximum)
			directed.Set(r, neighbor.Index, weight)
		}
	}

	return directed
}

// getNeighborsRow takes as input a slice of floats representing a row of the distance matrix and an integer r representing the index of the current row. It returns a slice of Neighbor structs representing all neighbors of the current node, excluding itself.
// Vania Halim 11/27/2025; Qinglin Kong 11/29/2025
func getNeighborsRow(row []float64, r int) []Neighbor {
	// make neighbors
	neighbors := make([]Neighbor, 0)

	// range through columns
	for c := range row {
		if c == r {
			continue // skip self
		}
		neighbors = append(neighbors, Neighbor{Index: c, Distance: row[c]})
	}

	return neighbors
}

// topKNeighbors takes as input a slice of Neighbor structs and an integer k representing the number of nearest neighbors to select. It returns a slice of Neighbor structs representing the k nearest neighbors, sorted by distance in ascending order.
// Vania Halim 11/27/2025; Qinglin Kong 11/29/2025
func topKNeighbors(n []Neighbor, k int) []Neighbor {
	// sort ascending by distance
	sort.Slice(n, func(a, b int) bool {
		return n[a].Distance < n[b].Distance
	})
	// prevent out of bounds
	if k > len(n) {
		k = len(n)
	}
	return n[:k] // the k nearest neighbors
}

// distanceToWeight takes as input a float representing the distance between two nodes. It returns a float representing the weight of the edge connecting the two nodes, calculated as 1 / (1 + distance).
// This function is used to covert distances into weights for the edges in the KNN graph, where smaller distances correspond to stornger connections (higher weights).
// Qinlgin K0ng 11/29/2025
func distanceToWeight(distance float64) float64 {
	return 1.0 / (1.0 + distance)
}

// symmetrizeWeightsUsing takes as input a pointer to a mat.Dense representing the directed KNN weights between nodes. It returns a map[int][]Edge representing the undirected edges of the graph, where each key is a node index and the value is a slice of Edge structs representing the edges connected to that node.
// Qinglin Kong 11/29/2025, Vania Halim 11/29/2025
func symmetrizeWeightsUsing(directed *mat.Dense) (map[int][]Edge, float64) {
	row, _ := directed.Dims()
	edges := make(map[int][]Edge) // final undirected adjacency map

	var totalWeight float64 // sum of all undirected weights

	// ensure every node appears in the adjacency map, even if isolated
	for i := range row {
		edges[i] = nil
	}

	// range through all pairs of nodes to symmetrize weights
	for i := 0; i < row; i++ {
		for j := i + 1; j < row; j++ {
			weightIJ := directed.At(i, j) // weight from i to j
			weightJI := directed.At(j, i) // weight from j to i

			// if no edge exists in either direction, skip
			if weightIJ == 0 && weightJI == 0 {
				continue
			}

			// undirected weight is the maximum of the two directions
			undirectedWeight := math.Max(weightIJ, weightJI)

			// skip if undirected weight is non-positive
			if undirectedWeight <= 0 {
				continue
			}

			// add edge i <-> j
			edges[i] = append(edges[i], Edge{To: j, Weight: undirectedWeight})
			edges[j] = append(edges[j], Edge{To: i, Weight: undirectedWeight})

			// add weight to totalWeight
			totalWeight += undirectedWeight
		}
	}

	return edges, totalWeight
}

// ==================== Leiden Clustering ====================

// Leiden is a method of the *Graph object. It applies the Leiden algorithm to the KNN graph.
// Input: a resolution parameter as a decimal and a maximum number of iterations as an integer.
// Output: a slice of integers representing the communities that each original node is assigned to after Leiden is applied.
// Vania Halim 11/28/2025
func (g *Graph) Leiden(resolution float64, maxIter int) []int {

	// initialize clusters with each node in its community
	clusters := g.InitSingletonPartition()

	for i := 0; i < maxIter; i++ {
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

// MoveNodes is a *Graph method that iteratively moves nodes in the graph to different communities until no single node moves result in a modularity gain
// Input: a partition as a slice of integers denoting the cluster for each node and a resolution parameter as a decimal
// Output: a partition assigning each node to a cluster
func (g *Graph) MoveNodes(partition []int, resolution float64) []int {

	improved := true

	for improved {

		improved = false

		random := RandomNodeOrder(len(partition))

		// move each node order[i]
		for _, i := range random {

			var nodeImproved bool // true if single node improvement resulted in a modularity gain

			// find candidate clusters - clusters that node i's neighbors belong to, excluding node i's own cluster
			candidateClusters := FindCandidateClusters(i, g.Edges[i], partition)
			partition[i], nodeImproved = FindBestCluster(i, g, candidateClusters, partition, resolution)

			if nodeImproved {
				improved = true
			}

		}

	}

	return partition

}

func FindBestCluster(node int, g *Graph, candidateClusters, partition []int, resolution float64) (int, bool) {

	// initialize maxGain and bestCluster for each node
	var maxGain float64
	currCluster := partition[node]
	bestCluster := currCluster

	// range through each cluster in candidateClusters
	for _, cluster := range candidateClusters {

		// find deltaQ of moving node i to each cluster in candidateClusters
		gain := g.ModularityGain(node, cluster, partition, resolution)

		// update maxGain and bestCluster if needed
		if gain > maxGain {
			maxGain = gain
			bestCluster = cluster
		}

	}

	improved := (bestCluster != currCluster)

	return bestCluster, improved

}

func RandomNodeOrder(n int) []int {

	// initialize slice of nodeIDs in order
	nodes := make([]int, n)
	for i := range nodes {
		nodes[i] = i
	}

	// randomize nodes order
	randomized := make([]int, n)
	for i := n - 1; i > 0; i-- {
		j := rand.Intn(i + 1)
		nodes[i], nodes[j] = nodes[j], nodes[i]
	}

	return randomized

}

// FindCandidateClusters returns the candidate clusters that node i can move into based on its neighbors
// Input: a node id i as an integer, node i's edges as a []Edge, and a partition as a []int mapping node id to cluster id
// Output: a []int where each value is the id of a cluster that node i could move to, excluding itself
// Vania Halim 11/29/2025

func FindCandidateClusters(node int, edges []Edge, partition []int) []int {

	// identify candidate clusters from neighbors of the current Node
	candidateClusters := make([]int, 0)
	currCluster := partition[node]

	for _, e := range edges {

		neighborCluster := partition[e.To]

		// skip if the neighbor's cluster is the same as the node's cluster
		if neighborCluster == currCluster {
			continue
		}

		// else add that cluster to the list of candidate clusters
		candidateClusters = append(candidateClusters, neighborCluster)
	}

	return candidateClusters

}

// InitSingletonPartition initializes a partition where each node is its own cluster. It returns a slice of integers representing the initial partition.
// Vania Halim 11/28/2025
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

	// compute observed: sum of edge weights between node i and nodes in the cluster
	for _, e := range g.Edges[i] { // ranges through all edges in node i
		if partition[e.To] == cluster { // if node i has an edge to a node in cluster
			observed += e.Weight // add weight of that edge to sum of observed edge weights
		}
	}

	// compute kj (cluster degree): sum of edge weights for all nodes in the cluster
	for node, community := range partition {

		// if the node is not in the community cluster then skip it
		if community != cluster {
			continue
		}

		// else explore the edge weights of that node
		for _, e := range g.Edges[node] {
			kj += e.Weight
		}

	}

	// compute ki (node i degree) : sum of edge weights incident to node i
	for _, e := range g.Edges[i] {
		ki += e.Weight
	}

	// compute deltaQ
	expected := (ki * kj) / g.TotalWeight
	deltaQ := observed - (resolution * expected)

	return deltaQ
}

// func (g *Graph) Refine(partition []int) []int
// func (g *Graph) Aggregate(partition []int) *Graph

// Copy creates a copy of the given partition slice and returns it.
// Qinglin Kong 11/29/2025
func Copy(partition []int) []int {
	newPartition := make([]int, len(partition))
	copy(newPartition, partition)
	return newPartition
}

// Compare checks if two partitions (basically two slices of integers) are equal.
// Qinglin Kong 11/29/2025
func Compare(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
