package main

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"testing"

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
// Vania Halim 11/27/2025, Yinan Zhu 12/06/2025
func DistanceMatrix(data *mat.Dense) (*mat.Dense, error) {
	rows, cols := data.Dims()

	// check for negative values in input data
	for r:=0; r<rows; r++{
		for c:=0; c<cols; c++{
			val :=data.At(r,c)
			if val < 0{
				return nil, fmt.Errorf("DistanceMatrix: negative value %f found at row %d, column %d", val, r, c)
			}
		}
	}

	//initialize distance matrix
	distMtx := mat.NewDense(rows, rows, nil)

	// calculate and store euclidean distance of each cell in distMtx
	for r:=0; r<rows; r++{
		
		// the diagonal is always zero
		cellOne := data.RawRowView(r)
		distMtx.Set(r, r, 0)

		for c := r + 1; c < rows; c++ {
			cellTwo := data.RawRowView(c)
			// calculate Euclidean distance between two cells and set it in the matrix
			distance := Euclidean(cellOne, cellTwo)
			distMtx.Set(r, c, distance)
			distMtx.Set(c, r, distance)
		}
	}

	return distMtx, nil
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
// Qinglin Kong 11/29/2025
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
func (g *Graph) Leiden(resolution, gamma, theta float64, maxIter int) []int {

	// initialize clusters with each node in its community
	clusters := g.InitSingletonPartition()

	for i := 0; i < maxIter; i++ {
		// create copy of clusters to compare to the new one
		old := Copy(clusters)

		// move all nodes until clusters reaches local convergence
		clusters = g.MoveNodes(clusters, resolution)
		clusters = g.RefinePartition(clusters, resolution, gamma, theta)

		// if clusters have converged globally, return clusters
		if Compare(old, clusters) {
			return clusters
		}

		// else aggregate and repeat
		clusters = g.Refine(clusters)
		g = g.Aggregate(clusters)
		clusters = g.InitSingletonPartition()
	}

	return clusters

}

// ==================================== AGGREGATION STAGE OF LEIDEN ====================================

// Refine looks at each cluster and sees if it can be further subdivided into clusters
// Input: a []int partition mapping node IDs to cluster IDs, e.g. [3,3,0,0,0,3,2]
// Output: a []int with normalized partition values: [0,0,1,1,1,0,2]
// Vania Halim - 11/29/2025
func (g *Graph) Refine(partition []int) []int {

	normalized := make(map[int]int, 0)
	var numUnique int

	// collect unique partition values and map them to "normalized" partition values
	for i, cluster := range partition {

		_, exists := normalized[cluster]

		if !exists { // encountering new unique cluster ID
			normalized[cluster] = numUnique
			numUnique++
		}

		// set value based on normalized
		partition[i] = normalized[cluster]
	}

	return partition
}

//func (g *Graph) Aggregate(partition []int) *Graph

// ==================================== REFINEMENT STAGE OF LEIDEN ====================================

// RefinePartition
// Input: partition []int mapping nodes to cluster IDs after the local move stage, and parameters resolution, gamma, theta as float64.
// Output: a refined partition as a []int, which may be further subdivided into different clusters
// Vania Halim - 11/29/2025
func (g *Graph) RefinePartition(partition []int, resolution, gamma, theta float64) []int {
	n := len(partition)

	refinedPartition := InitSingletonPartition(n) // initialize
	clusters := g.NodesByCluster(partition)

	for _, nodes := range clusters {
		refinedPartition = g.MergeNodesSubset(nodes, partition, refinedPartition, resolution, gamma, theta)
	}

	return refinedPartition
}

// InitSingletonPartition takes as input an integer n representing the number of nodes. It returns a slice of integers of length n where each node is assigned to its own unique cluster.
// Vania Halim - 11/29/2025
func InitSingletonPartition(n int) []int {
	partition := make([]int, n)
	// put each node in its own partition
	for i := range partition {
		partition[i] = i
	}
	return partition
}

// (g *Graph) InitSingletonPartition() takes no input and returns a slice []int of length g.Nodes where each node is mapped to its own cluster
// Qinglin Kong - 11/30/2025
func (g *Graph) InitSingletonPartition() []int {
	return InitSingletonPartition(g.Nodes)
}

// NodesByCluster takes a partition as a []int and returns a mapping of cluster IDs []int containing nodes belonging to that clusterID
// Vania Halim - 11/29/2025
func (g *Graph) NodesByCluster(partition []int) map[int][]int {

	grouped := make(map[int][]int)

	for node, cluster := range partition {

		grouped[cluster] = append(grouped[cluster], node)

	}

	return grouped
}

// MergeNodesSubset takes as input a refinedPartition []int where each node is originally a singleton. It modifies refinedPartition in place so that new sub clusters might be formed.
// Vania Halim - 11/29/2025
func (g *Graph) MergeNodesSubset(nodes, partition, refinedPartition []int, resolution, gamma, theta float64) []int {
	if len(nodes) <= 1 {
		return refinedPartition // nothing to refine
	}

	wellConnectedNodes := g.FindWellConnectedNodes(nodes, partition, gamma) // well connected nodes within a cluster
	randomNodes := RandomNodeOrder(len(wellConnectedNodes))

	improved := false

	for improved {

		for _, i := range randomNodes {

			currNode := wellConnectedNodes[i]

			if !Singleton(currNode, partition) {
				continue
			}

			// only process singleton nodes
			wellConnectedClusters := g.FindWellConnectedClusters(nodes, refinedPartition, gamma)

			if len(wellConnectedClusters) == 0 { // nonzero number of candidate clusters
				continue
			}

			// compute likelihood of moving to a candidate cluster
			probs := g.ComputeMoveProbability(currNode, wellConnectedClusters, refinedPartition, theta, resolution)

			// move to one of the candidate clusters based on probabilities computed
			newCluster := g.SampleCommunity(wellConnectedClusters, probs)
			refinedPartition[currNode] = newCluster

			improved = true

		}
	}

	return refinedPartition
}

// Find Well-connected cluster
// Amy Ji - 11/30/2025, Vania Halim - 11/30/2025
func (g *Graph) FindWellConnectedClusters(nodes, refinedPartition []int, gamma float64) []int {

	// a map of cluster id: []node
	clusterMembers := g.NodesByCluster(nodes)

	//candidates will be a slice of community ID that passes the test
	candidates := make([]int, 0)

	// equation in the paper was: E(C,S\C) >= y*kc(ks-kc)
	ks := g.ClusterDegree(nodes) // sum of edge weights in cluster S

	// sum of edge weights between community C and the other nodes in S: E(C, S\C)

	for cid, members := range clusterMembers {

		kc := g.ClusterDegree(members)
		if kc == 0 {
			continue
		}

		// find nodes that belongs to C
		inC := make(map[int]struct{}, len(members))
		for _, v := range members {
			inC[v] = struct{}{}
		}

		// find members in S minus C (S\C), as we want nodes in S that are not in C
		sMinusC := make([]int, 0, len(nodes)-len(members))

		// loop over all nodes in S, and if we find one that is not in C, we append ti to sMinusC
		for _, v := range nodes {

			_, exists := inC[v] // check if node v exists in C

			if !exists {
				sMinusC = append(sMinusC, v)
			}

		}

		// if sMinusC is zero, that means C==S
		if len(sMinusC) == 0 {
			continue
		}

		var eCS float64 // eCS stands for total edge weigt between C and S\C
		for _, v := range members {
			// for each node in C
			eCS += g.EdgesToCluster(v, sMinusC)
		}
		if eCS >= gamma*kc*(ks-kc) {
			candidates = append(candidates, cid)
		}

	}
	return candidates
}

// SampleCommunity samples a cluster from the given clusters based on the provided probabilities.
// It takes as input a slice of cluster IDs and a corresponding slice of probabilities. It returns the selected cluster ID.
// Choose random community C'
// Qinglin Kong 11/30/2025
func (g *Graph) SampleCommunity(clusters []int, probs map[int]float64) int {
	// normalize probabilities
	total, cumulative := 0.0, 0.0
	for _, p := range probs {
		total += p
	}

	// sample a random number between 0 and total
	r := rand.Float64() * total

	// v |-> C'
	for _, c := range clusters {
		cumulative += probs[c]
		if r <= cumulative {
			return c // v |-> C'
		}
	}

	return clusters[len(clusters)-1] // fallback
}

// ComputeMoveProbability computes and maps the clusterID of the new cluster C' to the probability of moving node v into cluster C' for all clusters in the subset S of candidate clusters according to the randomness parameter theta
// Vania Halim - 11/30/2025
func (g *Graph) ComputeMoveProbability(currNode int, candidateClusters, refinedPartition []int, theta, resolution float64) map[int]float64 {

	probs := make(map[int]float64) // maps clusterID to the probability of moving into that cluster
	var sum float64                // for normalization

	// compute probability of moving into each cluster in candidateClusters (unnormalized)
	for _, c := range candidateClusters {

		dQ := g.ModularityGain(currNode, c, refinedPartition, resolution)

		if dQ < 0 {
			probs[c] = 0.0
			continue
		}

		// otherwise compute the probability
		p := math.Exp((1 / theta) * dQ) // individual probability of v -> C'
		probs[c] = p
		sum += p

	}

	// normalize probabilities
	if sum > 0 {
		for key := range probs {
			probs[key] = probs[key] / sum
		}
	}

	return probs

}

// FindWellConnectedNodes
// Vania Halim - 11/29/2025
func (g *Graph) FindWellConnectedNodes(nodes, partition []int, gamma float64) []int {
	connected := make([]int, 0)
	kCluster := g.ClusterDegree(nodes) // total weights

	// range over all nodes in the original cluster
	for _, node := range nodes {

		kNode := g.NodeDegree(node)

		// compute E(v, S-v)
		ev := g.EdgesToCluster(node, nodes)

		// compute connectivity
		threshold := gamma*kNode - (kCluster * kNode)
		if ev > threshold {
			connected = append(connected, node)
		}

	}

	return connected

}

// EdgesToCluster computes the total edge weights of the node to all nodes in the subset except itself E(C,S\C)
// Vania Halim - 11/29/2025
func (g *Graph) EdgesToCluster(node int, cluster []int) float64 {
	var sum float64

	// create set of nodes in the cluster
	clusterSet := make(map[int]struct{}, len(cluster))

	// range through each node in the cluster and add it to the map
	for _, currNode := range cluster {

		if currNode != node {
			clusterSet[currNode] = struct{}{}
		}

	}

	// range through each edge, if it is to a node in the cluster, add weight to sum
	for _, e := range g.Edges[node] {

		_, exists := clusterSet[e.To] // check if e.To is in the cluster

		if exists {
			sum += e.Weight
		}
	}

	return sum
}

// ClusterDegree computes the total edge weights of all nodes in a cluster
// Vania Halim - 11/29/2025
func (g *Graph) ClusterDegree(nodes []int) float64 {
	var sum float64
	// range through all nodes in the cluster
	for _, node := range nodes {

		// range through all edges in each node
		for _, e := range g.Edges[node] {
			sum += e.Weight
		}

	}

	return sum
}

// Singleton checks whether a given node ID in refinedPartition is the only node in its cluster returning true if so and false otherwise
// Vania Halim - 11/29/2025
func Singleton(node int, refinedPartition []int) bool {
	count := 0
	nodeCluster := refinedPartition[node]

	for _, cluster := range refinedPartition {
		if cluster == nodeCluster {
			count++
		}

		if count > 1 {
			return false
		}
	}

	return true
}

// ============================ LOCAL MOVE NODES PHASE OF LEIDEN ===============================

// MoveNodes is a *Graph method that iteratively moves nodes in the graph to different communities until no single node moves result in a modularity gain
// Input: a partition as a slice of integers denoting the cluster for each node and a resolution parameter as a decimal
// Output: a partition as a slice of ints where indices are node IDs and the value is the cluster ID
// Vania Halim - 11/29/2025
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

// FindBestCluster identifies the cluster among a node's candidate clusters that would result in the highest modularity gain
// Input: node as an integer, the undirected graph as a *Graph, slices of integers representing the ids of candidate clusters that node will be moved into, partition as a []int mapping node IDs to clusterIDs, and the resolution parameter as a float64.
// Output: the cluster ID that results in the highest modularity gain, and a boolean that is true when there is a modularity gain from moving the node into any of the candidate clusters
// Vania Halim - 11/29/2025
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

// RandomNodeOrder implements Fisher-Yates swapping on the slice of node IDs, returning a randomized order of node IDs to walk through
// Input: n, the number of nodes
// Output: a randomized slice of node IDs
// Vania Halim - 11/29/2025; Qinglin Kong - 12/4/2025
func RandomNodeOrder(n int) []int {
	// initialize slice of nodeIDs in order
	nodes := make([]int, n)
	for i := range nodes {
		nodes[i] = i
	}

	for i := n - 1; i > 0; i-- {
		j := rand.Intn(i + 1)
		nodes[i], nodes[j] = nodes[j], nodes[i]
	}

	return nodes
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

func (g *Graph) NodeDegree(nodeID int) float64 {

	var ki float64

	for _, e := range g.Edges[nodeID] {
		ki += e.Weight
	}

	return ki
}

// Aggregate takes as input a partition as a []int mapping node IDs to cluster IDs.
// It returns a new *Graph where each cluster in the partition is represented as a single node, and the edge weights between clusters are the sum of the edge weights between nodes in the original graph.
// Qinglin Kong - 11/30/2025
func (g *Graph) Aggregate(partition []int) *Graph {
	// v <- p
	clusters := g.NodesByCluster(partition)
	numClusters := len(clusters)

	// create new adjacency map for edges between clusters
	// newAdjacencyMap = clusterU -> (clusterV -> weight)
	newAdjacencyMap := make(map[int]map[int]float64, numClusters)
	for c := range clusters {
		newAdjacencyMap[c] = make(map[int]float64)
	}

	// range through all edges in the original graph
	// e <- {(C,D) | (u,v) ∈ E(G), u ∈ C ∈ P, v ∈ D ∈ P}
	for u, edges := range g.Edges {
		clusterU := partition[u] // cluster of node u, so u ∈ C
		for _, e := range edges {
			v := e.To
			clusterV := partition[v] // cluster of node v, so v ∈ D

			// add edge weight to the edge between clusterU and clusterV
			newAdjacencyMap[clusterU][clusterV] += e.Weight
		}
	}

	// convert newAdjacencyMap to map[int][]Edge
	e, totalWeight := make(map[int][]Edge, numClusters), 0.0

	for c, neighbors := range newAdjacencyMap {
		for d, weight := range neighbors {
			// skip non-positive weights
			if weight <= 0 {
				continue
			}

			// add edge C -> D
			e[c] = append(e[c], Edge{To: d, Weight: weight})

			// sum total weight (only count top half to avoid double counting)
			if c < d {
				totalWeight += weight
			}
		}
	}

	// retunr G'
	return &Graph{numClusters, e, totalWeight}
}

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

// ==================== END OF clustering.go ====================


// ==================== testing functions for the functions in clustering ====================


//==================== 
//==================== 
// ****** testing cases for DistanceMatrix ******

//name: Yinan Zhu 
//date: 12/4/2025

// helper function to compare two float 64 with tolerance 
func almostEqual(a, b, tol float64) bool {
	return math.Abs(a-b) <= tol
}

//name: Yinan Zhu 
//date: 12/4/2025

func TestDistanceMatrix_SmallExample(t *testing.T) {
	// ===== Case 1: input has 1 row and 1 column--> output distance matrix should be 1×1 with value 0 ===
	
	data:=mat.NewDense (1, 1, []float64 {0.0})

	//call the DistanceMatrix function 
	dist, err:=DistanceMatrix(data)
	if err != nil {
		t.Errorf("TestDistanceMatrix_SmallExample Case 1 failed with error: %v", err)
	}

	// expected output: 1 x1 matrix with value 0
	expected:=mat.NewDense(1, 1, []float64 {0.0})

	//comparing the result (dist) with the expected output 
	//mat.EqualApprox compares two matrices with a tolerance level
	//if the two matrices are not approximately equal, report an error
	if !mat.EqualApprox(dist, expected, 1e-9) {
		t.Errorf("TestDistanceMatrix_SmallExample Case 1 failed: expected %v, got %v", expected, dist)
	}

	// ===== Case 2: input has 2 row and 2 column --> the distance should be 5  ===
	//input matrix: (0,0) and (3,4) 
	data2:= mat.NewDense(2, 2, []float64{0.0, 0.0, 3.0, 4.0})

	//call the DistanceMatrix function 
	dist2, err:=DistanceMatrix(data2)
	if err != nil {
		t.Errorf("TestDistanceMatrix_SmallExample Case 2 failed with error: %v", err)
	}

	// expected output: 2 x2 matrix with value 5 at (0,1) and (1,0)
	expected2:=mat.NewDense(2, 2, []float64 {0.0, 5.0, 5.0, 0.0})

	//comparing the result (dist2) with the expected output 
	//if the two matrices are not approximately equal, report an error
	if !mat.EqualApprox(dist2, expected2, 1e-9) {
		t.Errorf("TestDistanceMatrix_SmallExample Case 2 failed: expected %v, got %v", expected2, dist2)
	}


	// ===== Case 3: input has 3 row and 1 column (non-symmetric input) ===
	//input matrix: (1), (4), (7)
	data3 := mat.NewDense(3, 1, []float64{1.0, 4.0, 7.0})

	//call the DistanceMatrix function
	dist3, err:=DistanceMatrix(data3)
	if err != nil {
		t.Errorf("TestDistanceMatrix_SmallExample Case 3 failed with error: %v", err)
	}

	// expected output: 3 x3 matrix with value 3 at (0,1), (1,0), (1,2), (2,1) and value 6 at (0,2), (2,0)
	expected3:=mat.NewDense(3, 3, []float64 {0.0, 3.0, 6.0, 3.0, 0.0, 3.0, 6.0, 3.0, 0.0})

	//comparing the result (dist3) with the expected output 
	//if the two matrices are not approximately equal, report an error
	if !mat.EqualApprox(dist3, expected3, 1e-9) {
		t.Errorf("TestDistanceMatrix_SmallExample Case 3 failed: expected %v, got %v", expected3, dist3)
	}

	// ===== Case 4: negative numbers should trigger an error ===
	
	data4 := mat.NewDense(2, 2, []float64{-1.0, -1.0, 3.0, 4.0})

	// call DistanceMatrix (we only care about the error)
	_, err := DistanceMatrix(data4)

	expectedErrMsg := "DistanceMatrix: negative value -1.000000 found at row 0, column 0"

	// compare result (err) with expected error message
	if err == nil || err.Error() != expectedErrMsg {
    t.Errorf("Test Case 4 failed: expected error %v, got %v", expectedErrMsg, err)


// ****** testing cases for Euclidean distance ******
//name: Yinan Zhu 
//date: 12/4/2025

func TestEuclidean(t *testing.T){

	tests :=[]struct{
		description string
		first_vector []float64
		second_vector []float64
		expected_result float64
		PanicExpected bool
	}{
		{
			// case 1: identical vectors should return distance 0. 
			description: "Test Case 1: 2D vectors",
			first_vector: []float64{1.0, 2.0, 5.0},
			second_vector: []float64{1.0, 2.0, 5.0},
			expected_result: 0.0,
			PanicExpected: false,
		},
		{
			// case 2: "normal 1D vectors should return correct distance"
			description: "Test Case 2: normal 1D vectors",
			first_vector: []float64{0.0},
			second_vector: []float64{2.0},
			expected_result: 2.0,
			PanicExpected: false,

		},
		{
			// case 3: "normal 2D vectors should return correct distance"
			description: "Test Case 3: normal 2D vectors",
			first_vector:[]float64{0.0,0.0},
			second_vector:[]float64{3.0,4.0},
			expected_result:5.0,
			PanicExpected: false,

		},
		{

			// case 4: "normal 3D "
			description: "Test Case 3: normal 3D vectors should return correct distance",
			first_vector: []float64{1.0, 2.0, 2.0},
			second_vector: []float64{4.0, 6.0, 6.0},
			expected: 7.0,
			PanicExpected: false,

		},
		{
			// case 5: vectors with mismatched lengths should trigger a panic
			description: "Test Case 5: mismatched lengths should trigger a panic",
			first_vector: []float64{1.0, 1.0},
			second_vector: []float64{1.0, 7.0, 8.0},
			expected_result: math.NaN(),//meaningless since we expect a panic (consulted chatGPT for the use of math.NaN())
			PanicExpected: true,
			
		},
		{
			// case 6: negative input should trigger a panic
			description: "Test Case 5: negative input should trigger a panic",
			first_vector: []float64{1.0, -2.0},
			second_vector: []float64{3.0, 4.0},
			expected_result: math.NaN(),//meaningless since we expect a panic
			PanicExpected: true,

		},	

	}

	// we start to check all the test cases defined above 
	for _, test := range tests {

		// consulted chatGPT for t.Run syntax
		// use t.Run to create a subtest for each test case
		t.Run(test.description, func (t *testing.T) {

			// consulted ChatGPT for the defer panic handling syntax
			defer func() {
				//consulted chatGPT for recover syntax
				if r := recover(); r != nil {
					if !test.PanicExpected {
						t.Errorf("Test failed: unexpected panic: %v", r)
					}
				} else {
					if test.PanicExpected {
						t.Errorf("Test failed: expected panic but none occurred")
					}
				}
			}

		}
	}



// ****** testing cases for BuildKNNGraph ******

// helper: check if u has an edge to v

type GraphTestCase struct {
	case_name           string
	distance       *mat.Dense
	k              int
	expectEdgesEmpty bool
	expectFullConnect bool   // used only for k > rows
}

//helper to check if edge u to v exists in the graph 
func edgeExists(g *Graph, u, v int) bool {
	for _, e := range g.Edges[u] {
		if e.To == v {
			return true
		}
	}
	return false
}

func TestBuildKNNGraph(t *testing.T) {
	testCases := []GraphTestCase{
	//test case 1: empoty distance matrix 
	{
		case_name: "Empty distance matrix",
		distance: mat.NewDense(0, 0, nil),
		k: 3,
		expectEdgesEmpty: true,
		expectFullConnect: false,
	},

	// test case 2: when k is larger than the number of rows
	{
		case_name: "k larger than number of rows",
		distance: mat.NewDense(2, 2, []float64{0.0, 1.9, 1.0, 100.0}),
		k: 5,
		expectEdgesEmpty: false,
		expectFullConnect: true,
	},

	//test case 3: when k is equal to 0 
	{
		case_name: "k equal to 0",
		distance: mat.NewDense(3, 3, []float64{0.9, 1.0, 2.0, 1.0, 0.89, 3.0, 2.0, 3.7, 20.0}),
		k: 0,
		expectEdgesEmpty: true,
		expectFullConnect: false,
	},
	// test case 4: when k is less than 0
	{
		case_name: "k less than 0",
		distance: mat.NewDense(3, 3, []float64{0.0, 1.0, 2.0, 1.0, 0.0, 3.0, 2.0, 3.0, 0.0}),
		k: -2,
		expectEdgesEmpty: true,
		expectFullConnect: false,
	
	},
	//test case 5: normal case with k =10 
	{
		case_name: "Normal case with k=10",
		distance: mat.NewDense(4, 4, []float64{0.0, 0.0, 2.0, 3.0, 1.5, 20.0, 4.0, 5.0, 2.0, 4.0, 0.0, 6.0, 3.0, 5.0, 6.0, 0.0}),
		k: 10,
		expectEdgesEmpty: false,
		expectFullConnect: true,
	},
}
	// iterate through each test case
	for _, tc := range testCases {
		// consulted chatGPT for t.Run syntax
		// use t.Run to create a subtest for each test case
		t.Run(tc.case_name, func(t *testing.T) {
			//call BuildKNNGrah function 
			graph, err := BuildKNNGraph(tc.distance, tc.k)
			if err != nil {
				t.Errorf("BuildKNNGraph failed with error: %v", err)
			}

			// check if edges are empty when expected
			if tc.expectEdgesEmpty {
				for _, edges := range graph.Edges {
					if len(edges) != 0 {
						t.Errorf("Expected edges to be empty, but found edges: %v", edges)
					}
				}
			}

			// check for full connectivity when expected
			if tc.expectFullConnect {
				numNodes, _ := tc.distance.Dims()
				for u := 0; u < numNodes; u++ {
					for v := 0; v < numNodes; v++ {
						if u != v && !edgeExists(graph, u, v) {
							t.Errorf("Expected full connectivity, but edge from %d to %d is missing", u, v)
						}
					}
				}
			}
		}
	}
}


//****** testing cases for fillDirectedKNNWeights ******
// case 1: Test fillDirectedKNNWeights on a 4x4 distance matrix with k = 1.
func TestFillDirectedKNNWeights_K1(t *testing.T) {
	
	// distances between 4 nodes:
	// row 0: [0, 1, 4, 3] -> nearest neighbor = 1
	// row 1: [1, 0, 2, 5]  -> nearest neighbor = 0
	// row 2: [4, 2, 0, 1]  -> nearest neighbor = 3
	// row 3: [3, 5, 1, 0]  -> nearest neighbor = 2

	distance := mat.NewDense(4, 4, []float64{
		0, 1, 4, 3,
		1, 0, 2, 5,
		4, 2, 0, 1,
		3, 5, 1, 0,
	})

	k := 1
	directed := fillDirectedKNNWeights(distance, k)

	//set a tolerance number used to safely compare floating-point numbers.
	eps := 1e-9

	expected := [][]int{
		{1}, // row 0 -> neighbor 1
		{0}, // row 1 -> neighbor 0
		{3}, // row 2 -> neighbor 3
		{2}, // row 3 ->  neighbor 2
	}

	for r := 0; r < 4; r++ {
		// get non-zero indices in row r
		idxs := nonZeroIndicesInRow(directed, r, eps)

		// check length
		// if the number of non-zero entries does not match expected, report an error
		if len(idxs) != len(expected[r]) {
			t.Fatalf("row %d: expected %d non-zero entries, got %d (indices=%v)",
				r, len(expected[r]), len(idxs), idxs)
		}
		// check values
		// compare each index in idxs with expected
		for i, idx := range idxs {
			if idx != expected[r][i] {
				t.Errorf("row %d: expected neighbor %v, got %v", r, expected[r], idxs)
			}
		}
	}
}

// case 2: Test fillDirectedKNNWeights on the same 4x4 matrix with k = 2

func TestFillDirectedKNNWeights_K2(t *testing.T) {
	// same distance matrix as above
	distance := mat.NewDense(4, 4, []float64{
		0, 1, 4, 3,
		1, 0, 2, 5,
		4, 2, 0, 1,
		3, 5, 1, 0,
	})

	//this time, we set k = 2 
	k := 2

	// call fillDirectedKNNWeights
	directed := fillDirectedKNNWeights(distance, k)

	//set a tolerance number used to safely compare floating-point numbers.
	eps := 1e-9

	// nearest 2 neighbors per row:
	// row 0: [1,3]
	// row 1: [0,2]
	// row 2: [1,3]
	// row 3: [0,2]
	expected := [][]int{
		{1, 3},
		{0, 2},
		{1, 3},
		{0, 2},
	}

	// check each row
	for r := 0; r < 4; r++ {
		// get non-zero indices in row r
		idxs := nonZeroIndicesInRow(directed, r, eps)

		// check length
		// if the number of non-zero entries does not match expected, report an error
		if len(idxs) != len(expected[r]) {
			t.Fatalf("row %d: expected %d non-zero entries, got %d (indices=%v)",
				r, len(expected[r]), len(idxs), idxs)
		}

		// check values
		// sort both slices to ensure order does not affect comparison
		//consulted chatGPT for sort syntax
		sort.Ints(idxs)
		sort.Ints(expected[r])

		// compare each index in idxs with expected
		// if any index does not match, report an error
		for i := range expected[r] {
			if idxs[i] != expected[r][i] {
				t.Errorf("row %d: expected neighbors %v, got %v", r, expected[r], idxs)
				break
			}
		}
	}
}

// ****** testing cases for topKNeighbors******

// case 1: test case with basic, normal input (k < len(n))
func TestTopKNeighbors_Basic(t *testing.T) {
	neighbors := []Neighbor{
		{Index: 0, Distance: 5.0},
		{Index: 1, Distance: 1.0},
		{Index: 2, Distance: 3.0},
		{Index: 3, Distance: 2.0},
	}

	//select the top 2 neighbors
	k := 2

	// call topKNeighbors
	// we pass a copy of neighbors to avoid modifying the original slice
	result := topKNeighbors(append([]Neighbor(nil), neighbors...), k) 

	// check length
	// if the length of result does not match k, report an error
	if len(result) != k {
		t.Fatalf("expected %d neighbors, got %d", k, len(result))
	}

	// after sorting by distance, we expect index 1 (1.0) and index 3 (2.0)
	expected := []Neighbor{
		{Index: 1, Distance: 1.0},
		{Index: 3, Distance: 2.0},
	}

	// compare result with expected
	eps := 1e-9

	// check each neighbor in result
	for i := range expected {

		// compare index and distance
		// if the Index does not match expected, then report an error
		if result[i].Index != expected[i].Index {
			t.Errorf("at position %d: expected Index %d, got %d", i, expected[i].Index, result[i].Index)
		}

		// if the distance does not match expected (within tolerance), then report an error
		if !almostEqual(result[i].Distance, expected[i].Distance, eps) {
			t.Errorf("at position %d: expected Distance %v, got %v",
				i, expected[i].Distance, result[i].Distance)
		}
	}
}

// case 2:  k = len(n) 
func TestTopKNeighbors_KEqualsLen(t *testing.T) {
	
	//define neighbors slice
	neighbors := []Neighbor{
		{Index: 0, Distance: 4.0},
		{Index: 1, Distance: 2.0},
		{Index: 2, Distance: 3.0},
	}

	// set k equal to len(neighbors)
	k := len(neighbors)

	// call topKNeighbors
	result := topKNeighbors(append([]Neighbor(nil), neighbors...), k)

	// check length
	// if the length of result does not match len(neighbors), report an error
	if len(result) != len(neighbors) {
		t.Fatalf("expected %d neighbors, got %d", len(neighbors), len(result))
	}

	// check that result is sorted by distance
	// iterate through result and ensure each distance is less than or equal to the next
	for i := 1; i < len(result); i++ {
		if result[i].Distance < result[i-1].Distance {
			t.Errorf("result not sorted ascending by distance at position %d: %v < %v",
				i, result[i].Distance, result[i-1].Distance)
		}
	}
}

// case 3: k greater than len(n) 
func TestTopKNeighbors_KGreaterThanLen(t *testing.T) {
	neighbors := []Neighbor{
		{Index: 0, Distance: 10.0},
		{Index: 1, Distance: 7.0},
	}

	// set k greater than len(neighbors)
	k := 10 

	// call topKNeighbors
	// pass a copy of neighbors to avoid modifying the original slice
	result := topKNeighbors(append([]Neighbor(nil), neighbors...), k)

	// check length
	// if the length of result does not match len(neighbors), report an error
	if len(result) != len(neighbors) {
		t.Fatalf("expected len(result) = %d when k > len(n), got %d",
			len(neighbors), len(result))
	}

	// check that result is sorted by distance
	// since there are only 2 neighbors, just check the first two
	if result[0].Distance > result[1].Distance {
		t.Errorf("expected result sorted ascending by distance, got %v", result)
	}
}

//  case 4: k = 0 
func TestTopKNeighbors_KZero(t *testing.T) {

	//define neighbors slice
	neighbors := []Neighbor{
		{Index: 0, Distance: 2.0},
		{Index: 1, Distance: 1.0},
	}

	// set k = 0
	k := 0
	// call topKNeighbors
	result := topKNeighbors(append([]Neighbor(nil), neighbors...), k)

	// check length
	// if the length of result is not 0, report an error
	if len(result) != 0 {
		t.Fatalf("expected empty slice when k=0, got length %d", len(result))
	}
}

// case 5: empty input slice 
func TestTopKNeighbors_EmptySlice(t *testing.T) {
	var neighbors []Neighbor

	k := 3
	result := topKNeighbors(neighbors, k)

	if result == nil {
		// it's okay if it's nil, but usually nicer to return empty slice
		// we only require length 0
	}

	if len(result) != 0 {
		t.Fatalf("expected empty result for empty input slice, got length %d", len(result))
	}

	// also check it equals an empty slice explicitly
	if !reflect.DeepEqual(result, []Neighbor{}) {
		t.Errorf("expected empty slice, got %v", result)
	}
}

//****** testing cases for symmetrizeWeightsUsing ******

//Name :Yinan Zhu 
//date: 12/5/2025
// this function creates an undirected graph with numNodes nodes and some edges for testing purposes.
func newSimpleGraph(numNodes int) *Graph {
    //
	g := &Graph{
        Nodes: numNodes,
        Edges: make(map[int][]Edge),
    }
    // add some edges
	// for simplicity, we create a small graph with a few edges
    if numNodes >= 2 {
        g.Edges[0] = append(g.Edges[0], Edge{To: 1, Weight: 1.0})
        g.Edges[1] = append(g.Edges[1], Edge{To: 0, Weight: 1.0})
    }
    return g
}

// case 1: Empty graph (will return empty community slice)
func TestLeiden_EmptyGraph(t *testing.T) {
   // create an empty graph
	g := &Graph{
        Nodes: 0,
        Edges: make(map[int][]Edge),
    }

	// call Leiden
    res := g.Leiden(1.0, 1.0, 1.0, 10)

	// if not returning empty slice, then report an error

    if len(res) != 0 {
        t.Errorf("expected empty community slice for empty graph, got len=%d", len(res))
    }
}

// case 2: when maxIter = 0, the function should return singleton partition
func TestLeiden_MaxIterZero(t *testing.T) {
    
	// create a simple graph with 4 nodes
	g := newSimpleGraph(4)


	// we prepare the expected singleton partition
    expected := g.InitSingletonPartition()

	// call Leiden with maxIter = 0
    got := g.Leiden(1.0, 1.0, 1.0, 0)

	// compare with expected
	// if not equal, then report an error
    if !reflect.DeepEqual(got, expected) {
        t.Errorf("for maxIter=0, expected singleton partition %v, got %v", expected, got)
    }
}

// Case 3: result length matching number of nodes 
func TestLeiden_ResultLengthMatchesNodes(t *testing.T) {
    // create a simple graph with 5 nodes
	g := newSimpleGraph(5)

	// call Leiden function
    res := g.Leiden(1.0, 1.0, 1.0, 5)

	// check length
	// if length does not match number of nodes, then report an error
    if len(res) != g.Nodes {
        t.Fatalf("expected result slice length %d, got %d", g.Nodes, len(res))
    }
}

// Case 4: labels should be >=0
func TestLeiden_NonNegative(t *testing.T) {
    // create a graph with 5 nodes
	g := newSimpleGraph(5)

	//	 call Leiden function
    res := g.Leiden(1.0, 1.0, 1.0, 5)

	// check length
	// if length does not match number of nodes, then report an error
    if len(res) != g.Nodes {
        t.Fatalf("expected result slice length %d, got %d", g.Nodes, len(res))
    }

	// check each community label
	// if any label <0, then report an error
    for i, c := range res {
        if c < 0 {
            t.Errorf("node %d has negative community label %d", i, c)
        }
    }
}



// ****** testing cases for Leiden clustering ******



// ****** testing cases for FindCandidateClusters ******

// ****** testing cases for FindCandidate Clusters ******

//***** testing cases for ModularityGain******

// ****** testing cases for RefinePartition ******

// ****** testing cases for Aggregate ******

