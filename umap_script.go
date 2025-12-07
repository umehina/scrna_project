package main

import (
	"math"
	"math/rand"
	"time"
	"fmt"
	"gonum.org/v1/gonum/mat"
)


type UMAPConfig struct {
	NNeighbors     int
	MinDist        float64
	NComponents    int // usually 2
	NEpochs        int
	LearningRate   float64
	NegativeSample int
	Metric         func(a, b []float64) float64
}

type UMAPEdge struct {
	I, J   int
	Weight float64 // fuzzy connection strength p_ij in (0,1]
}

type pair struct {
	I, J int
}

// UMAP runs the full pipeline on high-dimensional data:
//   data: [][]float64, each row is a vector (e.g., PCs)
//   cfg:   UMAPConfig
// Returns: low-dimensional embedding [][]float64 (n × NComponents)
func UMAP(data [][]float64,nNeighbors int,nComponents int,nEpochs int,learningRate float64,negativeSamples int,minDist float64) [][]float64 {
	// Optional: Gatekeeping. 
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

	// 1) distance matrix from data (using Euclidean)
	distMtx := computeDistanceMatrix(data, Euclidean)

	// 2) kNN for UMAP (directed, raw distances)
	knnIdx, knnDist := BuildKNNForUMAP(distMtx, nNeighbors)

	// 3) fuzzy graph
	fuzzyEdges := buildFuzzyGraph(knnIdx, knnDist, nNeighbors)
	debugFuzzyEdges(fuzzyEdges)


	// 4) initialize embedding
	embedding := initEmbedding(len(data), nComponents)

	// 5) optimize with SGD + negative sampling
	optimize(embedding, fuzzyEdges, nEpochs, learningRate, negativeSamples, minDist)
	normalizeEmbedding(embedding, 10.0) // for fine tunning parameters.
	return embedding
}


// ============================ Distance / kNN ============================

// BuildKNNForUMAP builds the directed kNN structure needed by UMAP.
// Adapted from BuildKNNGraph
// It returns:
//   - knnIdx[i]  = indices of the k nearest neighbors of point i
//   - knnDist[i] = corresponding RAW distances (unlike the cluster knn where we did the inverse?)
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
			knnDist[r][i] = nb.Distance // RAW distance
		}
	}

	return knnIdx, knnDist
}


// rho: distance to the closest non-self neighbor. Each cell will have one rho value. 
// sigma: bandwidth that's chosen (optimized iteratively) so that the sum of probabilities to the k neighbors is approximately log2(k). (Our max iteration is 64.)
// computeRhoSigma implements the UMAP local connectivity scaling.
// Given knnDist[i] = distances from point i to its k nearest neighbors
// it computes rho[i] (smallest positive distance) and sigma[i] (found by binary search so that sum_j exp( -max(0, d_ij - rho_i) / sigma_i ) ≈ log2(k)).
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
		foundPos := false
		for _, d := range dist {
			if d > 0 && d < minPos {
				minPos = d
				foundPos = true
			}
		}
		if !foundPos || math.IsInf(minPos, 1) {
			rho[i] = 0.0
		} else {
			rho[i] = minPos
		}

		// sigma_i via binary search so that sum_j p_{j|i} ≈ log2(k)
		//  where p_{j|i} = exp(-max(0, d_ij - rho_i) / sigma_i)
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
				// we've found the perfect sigma value. 
				break
			}
			if sumProbs > target {
				// sum of probabilities too big -> sigma too small -> shrink hi
				hi = mid
			} else {
				// probabilities too small -> sigma too large -> increase lo
				lo = mid
			}
		}

		sigma[i] = mid
	}

	return rho, sigma
	// rho and sigma correspond 
}


// buildDirectedProbs computes directed probabilities p_{j|i} for each edge (i -> j).
// It returns a sparse map keyed by (i,j).
// Amy Ji - 12/06/2025
func buildDirectedProbs(knnIdx [][]int, knnDist [][]float64, rho, sigma []float64) map[pair]float64 {
	n := len(knnIdx)
	directed := make(map[pair]float64)

	// for each point i, a set of directed edge probbilities p_ij to its neighbors j
	// This is the probability that point X_j is a neighbor of point X_i. 
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
				// this means that dist(i,j) equals rho.
				// which means that this j is the closest neighbor of i
				// it has the greatest probability to be selected as neighbor (p=1)
				p = 1.0
			} else {
				if sigma[i] == 0 {
					// this is to avoid having a sigma == 0 (which is rare, but we absolutely don't want this to happen in any case)
					p = 1.0
				} else {
					// otherwise we calculate p, which is kind of a  probablistic "similarity score".
					p = math.Exp(-effective / sigma[i])
				}
			}
			if p > 0 {
				directed[pair{I: i, J: j}] = p
			}
		}
	}

	return directed
}

// buildFuzzyGraph builds the undirected UMAP fuzzy graph (fuzzy simplicial set)
// from the kNN structure. It uses the UMAP scaling (rho,sigma), directed
// probabilities p_{j|i}, and fuzzy union: p_ij = p_{i|j} + p_{j|i} - p_{i|j} * p_{j|i}
// This behaves like a probabilistic OR: which means that if either direction is strong, the undirected edge is strong. 
// Returns a slice of UMAPEdge, which we will later feed into the optimizer function.
// Amy Ji - 12/06/2025
func buildFuzzyGraph(knnIdx [][]int, knnDist [][]float64, nNeighbors int) []UMAPEdge {
	n := len(knnIdx)
	if n == 0 {
		return nil
	}

	// Compute rho and sigma for all cells
	rho, sigma := computeRhoSigma(knnDist, nNeighbors)
	// Build directed probabilities P_ij
	directed := buildDirectedProbs(knnIdx, knnDist, rho, sigma) // Note that in directed, i is point of interest and j is its neighbors. (P_ji)

	// for each pair of i,j, we store their undirected prob. 
	undirected := make(map[pair]float64)
	for ij, p_ij := range directed {
		i,j := ij.I, ij.J
		if i > j {
			continue
		}

		p_ji := directed[pair{I:j, J:i}] //P_ij
		p := p_ij + p_ji - p_ij*p_ji // compute the new undirected prob.
		if p>0 {
			undirected[pair{I:i, J:j}] = p
		}
	}

	// convert the map into []UMAPEdge. Umap optimizer will work on this flat slice of edges.
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

// initEmbedding initializes a random low-dimensional embedding. (points are randomly placed on low dimension))
// Amy Ji - 12/06/2025
func initEmbedding(n, dim int) [][]float64 {
	embedding := make([][]float64, n)
	rand.Seed(time.Now().UnixNano())
	scale := 0.0001

	for i := 0; i < n; i++ {
		embedding[i] = make([]float64, dim)
		for d := 0; d < dim; d++ {
			embedding[i][d] = scale * rand.NormFloat64()
		}
	}
	return embedding
}

// optimize does two things:
// for each edge(i,j), it pulls them closer
// for each edge, it samples some random points and pushes them away from i.
// This function is generated by chatgpt. 
func optimize(embedding [][]float64,edges []UMAPEdge,nEpochs int,learningRate float64,negativeSamples int,minDist float64) {
	n := len(embedding)
	if n == 0 || len(edges) == 0 {
		return
	}
	dim := len(embedding[0])

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

	// UMAP kernel parameters (for spread=1, min_dist≈0.1 these are the
	// standard values; good enough for our purposes).
	const a = 1.929
	const b = 0.7915

	for epoch := 0; epoch < nEpochs; epoch++ {
		// Shuffle edges for stochasticity
		rand.Shuffle(len(edges), func(i, j int) {
			edges[i], edges[j] = edges[j], edges[i]
		})

		for _, e := range edges {
			i, j := e.I, e.J
			if i < 0 || i >= n || j < 0 || j >= n {
				continue
			}

			yi := embedding[i]
			yj := embedding[j]

			// ---------- POSITIVE (attractive) update ----------
			var dist2 float64
			for d := 0; d < dim; d++ {
				diff := yi[d] - yj[d]
				dist2 += diff * diff
			}
			if dist2 > 0 {
				dist := math.Sqrt(dist2)
				// low-dim similarity q_ij
				r2b := math.Pow(dist, 2.0*b)
				q := 1.0 / (1.0 + a*r2b)

				p := e.Weight // high-dim similarity p_ij in [0,1]
				// gradient magnitude ~ (p - q)
				coef := learningRate * (p - q)

				for d := 0; d < dim; d++ {
					diff := yi[d] - yj[d]
					grad := coef * diff
					yi[d] -= grad
					yj[d] += grad
				}
			}

			// ---------- NEGATIVE SAMPLING (repulsive) ----------
			if negativeSamples == 0 {
				continue
			}

			for s := 0; s < negativeSamples; s++ {
				k := rand.Intn(n)
				if k == i || k == j {
					continue
				}
				yk := embedding[k]

				var dist2Neg float64
				for d := 0; d < dim; d++ {
					diff := yi[d] - yk[d]
					dist2Neg += diff * diff
				}
				if dist2Neg == 0 {
					continue
				}
				distNeg := math.Sqrt(dist2Neg)
				r2bNeg := math.Pow(distNeg, 2.0*b)
				qNeg := 1.0 / (1.0 + a*r2bNeg)

				// For negatives, p = 0, so (p - q) = -q < 0
				coefNeg := learningRate * (0.0 - qNeg)

				for d := 0; d < dim; d++ {
					diff := yi[d] - yk[d]
					grad := coefNeg * diff
					// Since coefNeg < 0, this pushes yi and yk apart
					yi[d] -= grad
					yk[d] += grad
				}
			}
		}
	}
}

func normalizeEmbedding(embedding [][]float64, targetRadius float64) {
    n := len(embedding)
    if n == 0 {
        return
    }
    dim := len(embedding[0])

    // 1. Compute mean per dimension
    mean := make([]float64, dim)
    for i := 0; i < n; i++ {
        for d := 0; d < dim; d++ {
            mean[d] += embedding[i][d]
        }
    }
    for d := 0; d < dim; d++ {
        mean[d] /= float64(n)
    }

    // 2. Center embedding
    for i := 0; i < n; i++ {
        for d := 0; d < dim; d++ {
            embedding[i][d] -= mean[d]
        }
    }

    // 3. Find max radius
    maxR := 0.0
    for i := 0; i < n; i++ {
        var r2 float64
        for d := 0; d < dim; d++ {
            r2 += embedding[i][d] * embedding[i][d]
        }
        if r2 > maxR {
            maxR = r2
        }
    }
    if maxR == 0 {
        return
    }
    scale := targetRadius / math.Sqrt(maxR)

    // 4. Scale
    for i := 0; i < n; i++ {
        for d := 0; d < dim; d++ {
            embedding[i][d] *= scale
        }
    }
}


// computeDistanceMatrix builds an N×N symmetric distance matrix for data. (for UMAP)
// Adapted from func DistanceMatrix for clustering knn. 
// Amy Ji - 12/06/2025; Vania Halim 11/27/2025
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