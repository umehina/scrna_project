package main

import (
    "fmt"

    "gonum.org/v1/gonum/mat"
    "gonum.org/v1/plot"
    "gonum.org/v1/plot/plotter"
    "gonum.org/v1/plot/vg"
)

// PlotTSNE draws a 2D t-SNE embedding stored as an n×2 matrix.
func PlotEmb(embedding *mat.Dense, filename string, title, xaxislabel,yaxislabel string) error {
    r, c := embedding.Dims()
    if c != 2 {
        return fmt.Errorf("PlotTSNE: matrix must be n×2, got %d×%d", r, c)
    }

    // Convert mat.Dense → plotter.XYs
    pts := make(plotter.XYs, r)
    for i := 0; i < r; i++ {
        pts[i].X = embedding.At(i, 0) // t-SNE 1
        pts[i].Y = embedding.At(i, 1) // t-SNE 2
    }

    p := plot.New()
    p.Title.Text = title
    p.X.Label.Text = xaxislabel
    p.Y.Label.Text = yaxislabel

    scatter, err := plotter.NewScatter(pts)
    if err != nil {
        return fmt.Errorf("PlotTSNE: creating scatter: %w", err)
    }
    scatter.GlyphStyle.Radius = vg.Points(2)

    p.Add(scatter)

    if err := p.Save(6*vg.Inch, 6*vg.Inch, filename); err != nil {
        return fmt.Errorf("PlotTSNE: saving plot: %w", err)
    }
    return nil
}

// PlotPCA2D draws a 2D PCA projection stored as an n×2 matrix.
func PlotPCA2D(scores *mat.Dense, filename string) error {
    r, c := scores.Dims()
    if c != 2 {
        return fmt.Errorf("PlotPCA2D: matrix must be n×2, got %d×%d", r, c)
    }

    pts := make(plotter.XYs, r)
    for i := 0; i < r; i++ {
        pts[i].X = scores.At(i, 1) // PC1
        pts[i].Y = scores.At(i, 0) // PC2
    }

    p := plot.New()
    p.Title.Text = "PCA: PC1 vs PC2"
    p.X.Label.Text = "PC2"
    p.Y.Label.Text = "PC1"

    scatter, err := plotter.NewScatter(pts)
    if err != nil {
        return fmt.Errorf("PlotPCA2D: creating scatter: %w", err)
    }
    scatter.GlyphStyle.Radius = vg.Points(2)

    p.Add(scatter)

    if err := p.Save(6*vg.Inch, 6*vg.Inch, filename); err != nil {
        return fmt.Errorf("PlotPCA2D: saving plot: %w", err)
    }
    return nil
}
