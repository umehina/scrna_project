package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"gonum.org/v1/gonum/mat"
)

// ============================ Types ============================

// Optional config struct if you ever want a single-arg API.
type UMAPConfig struct {
	NNeighbors     int
	MinDist        float64
	NComponents    int // usually 2
	NEpochs        int
	LearningRate   float64
	NegativeSample int
	Metric         func(a, b []float64) float64
}

// UMAPEdge represents an undirected fuzzy edge between i and j with weight μ_ij.
type UMAPEdge struct {
	I, J   int
	Weight float64 // fuzzy connection strength μ_ij in (0,1]
}

// pair is used as a map key for directed / undirected probabilities.
type pair struct {
	I, J int
}

// ============================ High-level UMAP ============================

// defaultAB returns (a, b) parameters for the low-dimensional UMAP kernel
// Φ(r^2) = 1 / (1 + a * r^(2b))^b.
// These values approximate umap-learn's defaults for spread=1.0, min_dist≈0.1.
func defaultAB(minDist float64) (float64, float64) {
    if minDist <= 0 {
        minDist = 0.1
    }
    // Base values roughly matching umap-learn for min_dist≈0.1
    baseA, baseB := 1.929, 0.7915

    // Heuristic: as minDist gets smaller, make the kernel steeper near 0
    scale := 0.1 / minDist          // minDist<0.1 → scale>1
    a := baseA * scale * scale      // steeper decay for small minDist
    b := baseB

    return a, b
}


// normalizeEmbedding recenters at 0 and rescales so the furthest point
// has radius targetRadius.
func normalizeEmbedding(embedding [][]float64, targetRadius float64) {
	n := len(embedding)
	if n == 0 {
		return
	}
	dim := len(embedding[0])

	// 1) mean-center
	mean := make([]float64, dim)
	for i := 0; i < n; i++ {
		for d := 0; d < dim; d++ {
			mean[d] += embedding[i][d]
		}
	}
	for d := 0; d < dim; d++ {
		mean[d] /= float64(n)
	}
	for i := 0; i < n; i++ {
		for d := 0; d < dim; d++ {
			embedding[i][d] -= mean[d]
		}
	}

	// 2) find max radius
	maxR2 := 0.0
	for i := 0; i < n; i++ {
		var r2 float64
		for d := 0; d < dim; d++ {
			r2 += embedding[i][d] * embedding[i][d]
		}
		if r2 > maxR2 {
			maxR2 = r2
		}
	}
	if maxR2 == 0 {
		return
	}
	scale := targetRadius / math.Sqrt(maxR2)

	// 3) scale
	for i := 0; i < n; i++ {
		for d := 0; d < dim; d++ {
			embedding[i][d] *= scale
		}
	}
}

// UMAP is the main entry point: it constructs the high-dimensional fuzzy graph
// and optimizes a low-dimensional embedding using a UMAP-style objective.
func UMAP(data [][]float64,nNeighbors int,nComponents int,nEpochs int,learningRate float64,negativeSamples int,minDist float64) [][]float64 {
	// ---- 0) Parameter sanity / defaults ----
	if nNeighbors <= 0 {
		nNeighbors = 15
	}
	if nComponents <= 0 {
		nComponents = 2
	}
	if nEpochs <= 0 {
		nEpochs = 200
	}
	if learningRate <= 0 {
		learningRate = 1.0
	}
	if negativeSamples < 0 {
		negativeSamples = 0
	}
	if minDist <= 0 {
		minDist = 0.1
	}

	n := len(data)
	if n == 0 {
		return nil
	}

	// ---- 1) Distance matrix (high-dimensional) ----
	distMtx := computeDistanceMatrix(data, Euclidean)

	// ---- 2) kNN indices + distances ----
	knnIdx, knnDist := BuildKNNForUMAP(distMtx, nNeighbors)

	// ---- 3) Fuzzy graph (high-dimensional probabilities μ_ij) ----
	fuzzyEdges := buildFuzzyGraph(knnIdx, knnDist, nNeighbors)
	// debugFuzzyEdges(fuzzyEdges)

	// ---- 4) Initialize low-dimensional embedding ----
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	embedding := initEmbedding(n, nComponents, rng)

	// ---- 5) Optimize using UMAP-style objective ----
	a, b := defaultAB(minDist)
	repulsionStrength := 1.2

	optimizeUMAP(
		embedding,
		fuzzyEdges,
		nEpochs,
		learningRate,
		negativeSamples,   // negativeSampleRate
		repulsionStrength, // gamma
		a, b,
		rng,
	)

	// ---- 6) Optional post-normalization for "nice" plot scales ----
	normalizeEmbedding(embedding, 10.0)

	return embedding
}

// ============================ Distance / kNN ============================

// computeDistanceMatrix builds an N×N symmetric distance matrix for data.
// Adapted from DistanceMatrix used in clustering kNN.
func computeDistanceMatrix(data [][]float64, metric func([]float64, []float64) float64) *mat.Dense {
	n := len(data)
	m := mat.NewDense(n, n, nil)

	for i := 0; i < n; i++ {
		m.Set(i, i, 0)
		for j := i + 1; j < n; j++ {
			d := metric(data[i], data[j])
			m.Set(i, j, d)
			m.Set(j, i, d)
		}
	}
	return m
}

// BuildKNNForUMAP builds the directed kNN structure needed by UMAP.
// It returns:
//   - knnIdx[i]  = indices of the k nearest neighbors of point i
//   - knnDist[i] = corresponding RAW distances
//
// Amy Ji - 12/06/2025; Vania Halim 11/27/2025; Qinglin Kong 11/29/2025
func BuildKNNForUMAP(distanceMtx *mat.Dense, k int) ([][]int, [][]float64) {
	if distanceMtx == nil {
		return nil, nil
	}
	rows, _ := distanceMtx.Dims()
	if rows == 0 || k <= 0 {
		return nil, nil
	}
	if k >= rows {
		k = rows - 1
	}

	knnIdx := make([][]int, rows)
	knnDist := make([][]float64, rows)

	for r := 0; r < rows; r++ {
		row := distanceMtx.RawRowView(r)
		allNeighbors := getNeighborsRow(row, r)
		knn := topKNeighbors(allNeighbors, k)

		knnIdx[r] = make([]int, len(knn))
		knnDist[r] = make([]float64, len(knn))
		for i, nb := range knn {
			knnIdx[r][i] = nb.Index
			knnDist[r][i] = nb.Distance
		}
	}

	return knnIdx, knnDist
}

// ======================= Rho / Sigma and Probabilities ===================

// computeRhoSigma implements UMAP's local connectivity scaling.
// Given knnDist[i] = distances from point i to its k nearest neighbors,
// it computes:
//   rho[i]   = smallest positive distance, or 0 if none
//   sigma[i] = bandwidth chosen by binary search so that
//              sum_j exp(-max(0, d_ij - rho_i) / sigma_i) ≈ log2(k)
//
// Amy Ji - 12/06/2025
func computeRhoSigma(knnDist [][]float64, k int) (rho []float64, sigma []float64) {
	n := len(knnDist)
	rho = make([]float64, n)
	sigma = make([]float64, n)

	if n == 0 || k <= 0 {
		return rho, sigma
	}

	target := math.Log2(float64(k))
	const (
		maxIter   = 64
		tolerance = 1e-5
	)

	for i, dist := range knnDist {
		if len(dist) == 0 {
			continue
		}

		// 1) rho_i = smallest strictly positive distance, or 0 if none
		minPos := math.Inf(1)
		for _, d := range dist {
			if d > 0 && d < minPos {
				minPos = d
			}
		}
		if math.IsInf(minPos, 1) {
			rho[i] = 0.0
		} else {
			rho[i] = minPos
		}

		// 2) sigma_i via binary search so that sum_j p_{j|i} ≈ log2(k),
		//    where p_{j|i} = exp(-max(0, d_ij - rho_i) / sigma_i).
		allAtOrBelowRho := true
		for _, d := range dist {
			if d > rho[i] {
				allAtOrBelowRho = false
				break
			}
		}
		if allAtOrBelowRho {
			sigma[i] = 1.0
			continue
		}

		lo := 1e-3
		hi := 1e3
		var mid float64

		for iter := 0; iter < maxIter; iter++ {
			mid = 0.5 * (lo + hi)
			sumProbs := 0.0

			for _, d := range dist {
				effective := d - rho[i]
				if effective < 0 {
					effective = 0
				}
				if effective == 0 {
					sumProbs += 1.0
				} else {
					sumProbs += math.Exp(-effective / mid)
				}
			}

			if math.Abs(sumProbs-target) < tolerance {
				break
			}
			if sumProbs > target {
				// probabilities too big → sigma too small → enlarge sigma (increase hi)
				hi = mid
			} else {
				// probabilities too small → sigma too large → shrink sigma (increase lo)
				lo = mid
			}
		}

		sigma[i] = mid
	}

	return rho, sigma
}

// buildDirectedProbs computes directed probabilities p_{j|i} for each edge (i → j).
// It returns a sparse map keyed by (i,j).
//
// Amy Ji - 12/06/2025
func buildDirectedProbs(knnIdx [][]int, knnDist [][]float64, rho, sigma []float64) map[pair]float64 {
	n := len(knnIdx)
	directed := make(map[pair]float64)

	for i := 0; i < n; i++ {
		neighbors := knnIdx[i]
		dists := knnDist[i]
		if len(neighbors) != len(dists) {
			continue
		}

		for m, j := range neighbors {
			if i == j {
				continue
			}
			d := dists[m]
			effective := d - rho[i]
			if effective < 0 {
				effective = 0
			}

			var p float64
			if effective == 0 {
				// distance equals rho_i → closest neighbors get probability 1
				p = 1.0
			} else if sigma[i] == 0 {
				// extremely rare; avoid division by zero
				p = 1.0
			} else {
				p = math.Exp(-effective / sigma[i])
			}

			if p > 0 {
				directed[pair{I: i, J: j}] = p
			}
		}
	}

	return directed
}

// buildFuzzyGraph builds the undirected UMAP fuzzy graph (fuzzy simplicial set)
// from the kNN structure, using:
//   - rho, sigma from computeRhoSigma
//   - directed probabilities p_{j|i}
//   - fuzzy union: μ_ij = p_{i|j} + p_{j|i} - p_{i|j} * p_{j|i}
//
// Returns a flat slice of UMAPEdge, which is fed into the optimizer.
//
// Amy Ji - 12/06/2025
func buildFuzzyGraph(knnIdx [][]int, knnDist [][]float64, nNeighbors int) []UMAPEdge {
	n := len(knnIdx)
	if n == 0 {
		return nil
	}

	// 1) Compute rho and sigma for all points
	rho, sigma := computeRhoSigma(knnDist, nNeighbors)

	// 2) Directed probabilities P_ij
	directed := buildDirectedProbs(knnIdx, knnDist, rho, sigma)

	// 3) Fuzzy union for undirected edges
	undirected := make(map[pair]float64)
	for ij, p_ij := range directed {
		i, j := ij.I, ij.J
		if i > j {
			continue
		}
		p_ji := directed[pair{I: j, J: i}]
		p := p_ij + p_ji - p_ij*p_ji
		if p > 0 {
			undirected[pair{I: i, J: j}] = p
		}
	}

	// 4) Convert map → []UMAPEdge
	edges := make([]UMAPEdge, 0, len(undirected))
	for ij, w := range undirected {
		edges = append(edges, UMAPEdge{
			I:      ij.I,
			J:      ij.J,
			Weight: w,
		})
	}

	return edges
}

// debugFuzzyEdges prints quick stats on fuzzy edge weights.
func debugFuzzyEdges(edges []UMAPEdge) {
	if len(edges) == 0 {
		fmt.Println("No edges")
		return
	}
	minW := edges[0].Weight
	maxW := edges[0].Weight
	sumW := 0.0
	for _, e := range edges {
		if e.Weight < minW {
			minW = e.Weight
		}
		if e.Weight > maxW {
			maxW = e.Weight
		}
		sumW += e.Weight
	}
	meanW := sumW / float64(len(edges))
	fmt.Printf("Fuzzy edges: n=%d, min=%.4g, mean=%.4g, max=%.4g\n",
		len(edges), minW, meanW, maxW)
}

// ============================ Initialization ============================

// initEmbedding initializes a random low-dimensional embedding.
func initEmbedding(n, dim int, rng *rand.Rand) [][]float64 {
	embedding := make([][]float64, n)
	scale := 0.0001

	for i := 0; i < n; i++ {
		embedding[i] = make([]float64, dim)
		for d := 0; d < dim; d++ {
			embedding[i][d] = scale * rng.NormFloat64()
		}
	}
	return embedding
}

// ============================ Optimization ============================

// phi is the UMAP kernel in low-dimensional space:
// Φ(r^2) = 1 / (1 + a * r^(2b))^b.
func phi(dist2, a, b float64) float64 {
	return 1.0 / math.Pow(1.0+a*math.Pow(dist2, b), b)
}

// makeEpochsPerSample returns, for each edge weight, how many epochs should
// elapse between successive samples of that edge.
func makeEpochsPerSample(weights []float64, nEpochs int) []float64 {
	result := make([]float64, len(weights))
	maxW := 0.0
	for _, w := range weights {
		if w > maxW {
			maxW = w
		}
	}
	if maxW == 0 {
		for i := range result {
			result[i] = -1
		}
		return result
	}

	nSamples := make([]float64, len(weights))
	for i, w := range weights {
		nSamples[i] = float64(nEpochs) * (w / maxW)
	}
	for i, ns := range nSamples {
		if ns > 0 {
			result[i] = float64(nEpochs) / ns
		} else {
			result[i] = -1.0 // never sampled
		}
	}
	return result
}

// optimizeUMAP performs UMAP-style stochastic gradient descent on the embedding.
// For each fuzzy edge (i,j), it applies an attractive force, and for each such
// step it samples negative examples k and applies repulsive forces from i to k.
func optimizeUMAP(embedding [][]float64,edges []UMAPEdge,nEpochs int,learningRate float64,negativeSampleRate int,repulsionStrength float64,a, b float64,rng *rand.Rand,) {
	n := len(embedding)
	if n == 0 || len(edges) == 0 {
		return
	}
	dim := len(embedding[0])

	// Prepack edge endpoints and weights
	heads := make([]int, len(edges))
	tails := make([]int, len(edges))
	weights := make([]float64, len(edges))
	for e, edge := range edges {
		heads[e] = edge.I
		tails[e] = edge.J
		weights[e] = edge.Weight
	}

	epochsPerSample := makeEpochsPerSample(weights, nEpochs)
	nextSample := make([]float64, len(edges)) // all 0.0 initially

	lrInitial := learningRate

	for epoch := 0; epoch < nEpochs; epoch++ {
		// linear decay like umap-learn
		lr := lrInitial * (1.0 - float64(epoch)/float64(nEpochs))
		if lr <= 0 {
			continue
		}

		for e := range edges {
			if epochsPerSample[e] <= 0 {
				continue // never sampled
			}
			if float64(epoch) < nextSample[e] {
				continue
			}
			nextSample[e] += epochsPerSample[e]

			i := heads[e]
			j := tails[e]

			yi := embedding[i]
			yj := embedding[j]

			// --- Attractive update (positive edge i-j) ---
			diff := make([]float64, dim)
			dist2 := 0.0
			for d := 0; d < dim; d++ {
				diff[d] = yi[d] - yj[d]
				dist2 += diff[d] * diff[d]
			}
			dist2 += 1e-8 // tiny epsilon to avoid div-by-zero

			// grad coeff approximating umap-learn
			gradCoeff := -2.0 * a * b * math.Pow(dist2, b-1.0)
			gradCoeff /= (a*math.Pow(dist2, b) + 1.0)

			for d := 0; d < dim; d++ {
				grad := gradCoeff * diff[d]
				yi[d] += lr * grad
				yj[d] -= lr * grad
			}

			// --- Negative sampling: repulsive edges from i to random k ---
			for ns := 0; ns < negativeSampleRate; ns++ {
				k := rng.Intn(n)
				if k == i {
					continue
				}
				yk := embedding[k]

				ndiff := make([]float64, dim)
				dist2neg := 0.0
				for d := 0; d < dim; d++ {
					ndiff[d] = yi[d] - yk[d]
					dist2neg += ndiff[d] * ndiff[d]
				}
				dist2neg += 1e-8

				repGradCoeff := 2.0 * repulsionStrength * b
				repGradCoeff /= (0.001 + dist2neg) * (a*math.Pow(dist2neg, b) + 1.0)

				for d := 0; d < dim; d++ {
					grad := repGradCoeff * ndiff[d]
					yi[d] += lr * grad
					yk[d] -= lr * grad
				}
			}
		}
	}
}
