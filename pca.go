package main

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

// PCAResult stores the results of pca: cell embeddings (scores), gene loadings, variance explained per pc, and total variance.
type PCAResult struct {
	scores    *mat.Dense
	loadings  *mat.Dense
	variances []float64
	totalvar  float64

	geneNames []string
	barcodes  []string
}

// GetScores returns the PCA scores matrix (cell x PC)
func (p *PCAResult) GetScores() *mat.Dense {
	return p.scores
}

// ToDense converts the ExpressionMatrix into a row-major *mat.Dense so it can be used by gonum’s PCA routines.
func (em *ExpressionMatrix) ToDense() *mat.Dense {
	n := len(em.data) // number of cells
	if n == 0 {
		return mat.NewDense(0, 0, nil)
	}

	d := len(em.data[0]) // number of genes

	// mat.Dense requires a single contiguous float64 slice in row-major order.
	flat := make([]float64, 0, n*d)
	for i := 0; i < n; i++ {
		flat = append(flat, em.data[i]...)
	}

	return mat.NewDense(n, d, flat)
}

// PCACompute runs pca on a centered n×d matrix by computing eigenvectors/eigenvalues and projecting data onto the top k components.
// data: n x d matrix (rows=cells, columns=genes)
// k: number of principal components to compute
// returns: PCAResult containing scores, loadings, variances, total variance.
// Qinglin Kong 11/26/2025
func PCACompute(data *mat.Dense, k int) *PCAResult {
	_, d := data.Dims()

	// clamp k so it never exceeds number of genes.
	if k > d {
		k = d
	}

	// compute principal components
	var pc stat.PC
	if ok := pc.PrincipalComponents(data, nil); !ok {
		panic("pca failed")
	}

	// loadings is d x d; each column is a principal component (gene weights)
	var loadings mat.Dense
	pc.VectorsTo(&loadings)

	// eigenvalues (length d), sorted largest smallest
	variances := pc.VarsTo(nil)
	variancesk := variances[:k]

	// take only the first k eigenvectors (loadings)
	loadk := loadings.Slice(0, d, 0, k).(*mat.Dense)

	// scores = X * loadings_k
	// scores is n x k matrix (cell embeddings)
	var scores mat.Dense
	scores.Mul(data, loadk)

	// total variance is sum of all eigenvalues
	var total float64
	for _, v := range variances {
		total += v
	}

	return &PCAResult{
		scores:    &scores,
		loadings:  loadk,
		variances: variancesk,
		totalvar:  total,
	}
}

// This is a convenience wrapper to run PCA directly on an ExpressionMatrix.
// Qinglin Kong 11/26/2025
func (em *ExpressionMatrix) PCA(k int) *PCAResult {
	pcaResult := PCACompute(em.ToDense(), k)
	pcaResult.geneNames = em.genes
	pcaResult.barcodes = em.barcodes
	return pcaResult
}

// GetScores returns the PCA scores matrix (cell embeddings)
func (p *PCAResult) GetScores() *mat.Dense {
	return p.scores
}

// GetPC returns the j-th principal component (0-indexed) as a slice of float64.
func (p *PCAResult) GetPC(j int) []float64 {
	rows, _ := p.scores.Dims()
	out := make([]float64, rows)
	for i := 0; i < rows; i++ {
		out[i] = p.scores.At(i, j)
	}
	return out
}

// PlotPCAScatter generates a simple scatter plot of two PCs and saves it as a plot file.
// Qinglin Kong 11/26/2025
func (p *PCAResult) PlotPCAScatter(filename string, pcx, pcy int) error {
	x := p.GetPC(pcx)
	y := p.GetPC(pcy)

	n := len(x)
	pts := make(plotter.XYs, n)
	for i := 0; i < n; i++ {
		pts[i].X = x[i]
		pts[i].Y = y[i]
	}

	plt := plot.New()
	plt.Title.Text = "PCA Scatter Plot"
	plt.X.Label.Text = fmt.Sprintf("PC%d", pcx+1)
	plt.Y.Label.Text = fmt.Sprintf("PC%d", pcy+1)
	s, err := plotter.NewScatter(pts)
	if err != nil {
		return err
	}
	s.Radius = 1.5 * vg.Points(1)
	plt.Add(s)

	return plt.Save(6*vg.Inch, 6*vg.Inch, filename)
}