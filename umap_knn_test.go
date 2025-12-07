// knn_umap_test.go
package main

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

// helper for float comparison
func almostEqual(a, b, eps float64) bool {
	return math.Abs(a-b) <= eps
}

func TestBuildKNNForUMAP_Basic(t *testing.T) {
	// We create 4 points on a line: 0, 1, 3, 10
	// Distances:
	// 0: [0, 1, 3, 10]
	// 1: [1, 0, 2,  9]
	// 2: [3, 2, 0,  7]
	// 3: [10,9, 7,  0]
	data := []float64{
		0, 1, 3, 10,
		1, 0, 2, 9,
		3, 2, 0, 7,
		10, 9, 7, 0,
	}
	distanceMtx := mat.NewDense(4, 4, data)

	k := 2
	knnIdx, knnDist := BuildKNNForUMAP(distanceMtx, k)

	if knnIdx == nil || knnDist == nil {
		t.Fatalf("expected non-nil knnIdx/knnDist, got nil")
	}

	if len(knnIdx) != 4 || len(knnDist) != 4 {
		t.Fatalf("expected 4 rows, got len(knnIdx)=%d len(knnDist)=%d", len(knnIdx), len(knnDist))
	}

	// Each row should have exactly k neighbors
	for i := 0; i < 4; i++ {
		if len(knnIdx[i]) != k {
			t.Fatalf("row %d: expected %d neighbors, got %d", i, k, len(knnIdx[i]))
		}
		if len(knnDist[i]) != k {
			t.Fatalf("row %d: expected %d distances, got %d", i, k, len(knnDist[i]))
		}
	}

	// Check expected neighbors (indices and raw distances)

	// row 0: distances to others = [1, 3, 10] -> nearest 2: idx 1 (1), idx 2 (3)
	expIdx0 := []int{1, 2}
	expDist0 := []float64{1, 3}

	// row 1: distances = [1, 2, 9] -> nearest 2: idx 0 (1), idx 2 (2)
	expIdx1 := []int{0, 2}
	expDist1 := []float64{1, 2}

	// row 2: distances = [3, 2, 7] -> nearest 2: idx 1 (2), idx 0 (3)
	expIdx2 := []int{1, 0}
	expDist2 := []float64{2, 3}

	// row 3: distances = [10, 9, 7] -> nearest 2: idx 2 (7), idx 1 (9)
	expIdx3 := []int{2, 1}
	expDist3 := []float64{7, 9}

	expIdx := [][]int{expIdx0, expIdx1, expIdx2, expIdx3}
	expDist := [][]float64{expDist0, expDist1, expDist2, expDist3}

	for i := 0; i < 4; i++ {
		for j := 0; j < k; j++ {
			if knnIdx[i][j] != expIdx[i][j] {
				t.Errorf("row %d neighbor %d: expected index %d, got %d",
					i, j, expIdx[i][j], knnIdx[i][j])
			}
			if !almostEqual(knnDist[i][j], expDist[i][j], 1e-9) {
				t.Errorf("row %d neighbor %d: expected distance %.4f, got %.4f",
					i, j, expDist[i][j], knnDist[i][j])
			}
		}
	}
}

func TestBuildKNNForUMAP_KTooLarge(t *testing.T) {
	// Same 4×4 distance matrix as above
	data := []float64{
		0, 1, 3, 10,
		1, 0, 2, 9,
		3, 2, 0, 7,
		10, 9, 7, 0,
	}
	distanceMtx := mat.NewDense(4, 4, data)

	// Ask for a k that is too large; function should internally clamp to k = rows-1 = 3
	k := 10
	knnIdx, knnDist := BuildKNNForUMAP(distanceMtx, k)

	if knnIdx == nil || knnDist == nil {
		t.Fatalf("expected non-nil knnIdx/knnDist, got nil")
	}
	if len(knnIdx) != 4 || len(knnDist) != 4 {
		t.Fatalf("expected 4 rows, got len(knnIdx)=%d len(knnDist)=%d", len(knnIdx), len(knnDist))
	}

	for i := 0; i < 4; i++ {
		if len(knnIdx[i]) != 3 {
			t.Fatalf("row %d: expected 3 neighbors (rows-1), got %d", i, len(knnIdx[i]))
		}
		if len(knnDist[i]) != 3 {
			t.Fatalf("row %d: expected 3 distances (rows-1), got %d", i, len(knnDist[i]))
		}
	}
}

func TestBuildKNNForUMAP_BadK(t *testing.T) {
    // Non-empty matrix but k <= 0
    data := []float64{
        0, 1,
        1, 0,
    }
    distanceMtx := mat.NewDense(2, 2, data)

    knnIdx, knnDist := BuildKNNForUMAP(distanceMtx, 0)
    if knnIdx != nil || knnDist != nil {
        t.Fatalf("expected nil for k<=0, got non-nil")
    }
}
