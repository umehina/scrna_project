package main

import (
	"fmt"
    "gonum.org/v1/gonum/mat"
    "gonum.org/v1/plot"
    "gonum.org/v1/plot/plotter"
    "gonum.org/v1/plot/vg"
)

func PlotTSNE(embedding *mat.Dense, filename string) error {
    r, c := embedding.Dims()
    if c != 2 {
        return fmt.Errorf("TSNE matrix must be Nx2, got %dx%d", r, c)
    }

    // Convert mat.Dense → plotter.XYs
    pts := make(plotter.XYs, r)
    for i := 0; i < r; i++ {
        pts[i].X = embedding.At(i, 0)
        pts[i].Y = embedding.At(i, 1)
    }

    p := plot.New()
    p.Title.Text = "t-SNE Embedding"
    p.X.Label.Text = "t-SNE 1"
    p.Y.Label.Text = "t-SNE 2"

    scatter, err := plotter.NewScatter(pts)
    if err != nil {
        return err
    }

    scatter.GlyphStyle.Radius = vg.Points(2)

    p.Add(scatter)

    return p.Save(6*vg.Inch, 6*vg.Inch, filename)
}
