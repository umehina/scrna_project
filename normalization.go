package main

import (
    "math"
    "sort"
)

// Fit a simple log-linear regression: log(y+1) = β0 + β1 * log(s)
func fitGeneCoefficients(y []float64, s []float64) (float64, float64) {
    x := make([]float64, len(y))
    logY := make([]float64, len(y))
    for j := range y {
        if s[j] <= 0 {
            s[j] = 1e-6
        }
        x[j] = math.Log(s[j])
        logY[j] = math.Log(y[j] + 1)
    }

    meanX, meanY := mean(x), mean(logY)
    var num, den float64
    for j := range y {
        num += (x[j] - meanX) * (logY[j] - meanY)
        den += (x[j] - meanX) * (x[j] - meanX)
    }
    beta1 := num / den
    beta0 := meanY - beta1*meanX
    return beta0, beta1
}

func mean(arr []float64) float64 {
    if len(arr) == 0 {
        return 0
    }
    sum := 0.0
    for _, v := range arr {
        sum += v
    }
    return sum / float64(len(arr))
}

func estimateTheta(counts []float64) float64 {
    n := float64(len(counts))
    if n < 2 {
        return 1.0
    }
    sum, sumsq := 0.0, 0.0
    for _, x := range counts {
        sum += x
        sumsq += x * x
    }
    mean := sum / n
    variance := sumsq/n - mean*mean
    if variance <= mean {
        return 1e6 // ~Poisson
    }
    theta := mean * mean / (variance - mean)
    if theta < 1e-3 {
        theta = 1e-3
    }
    return theta
}

// computeSizeFactors calculates per-cell library size normalization factors.
// s_j = n_j / median(total_counts)
func (cm *CountMatrix) computeSizeFactors() []float64 {
    if cm == nil || len(cm.cells) == 0 {
        return nil
    }

    sizes := make([]float64, len(cm.cells))
    for i, c := range cm.cells {
        if c != nil && c.qcMetrics != nil {
            sizes[i] = float64(c.qcMetrics.nCountRNA)
        }
    }

    sorted := append([]float64(nil), sizes...)
    sort.Float64s(sorted)
    median := sorted[len(sorted)/2]
    if median == 0 {
        median = 1.0
    }

    for i := range sizes {
        sizes[i] /= median
    }

    return sizes
}


