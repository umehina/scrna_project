// datatypes.go

package main

type Cell struct {
	// ...
	idx       int
	barcode   string
	features  map[string]int
	qcMetrics QCMetrics
	// ...
}

type QCMetrics struct {
	nFeatureRNA int
	nCountRNA   int
	percentMT   float64 // fraction 0..1
}
