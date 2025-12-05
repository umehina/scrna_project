package main

import (
	"fmt"
	"log"
	"strings"
	"flag"
	
)

// config struct for RShiny app
// Amy Ji - 12/04/2025
type PipelineConfig struct {
    DataPath string

    // QC filter params
    minFeatures int
    maxFeatures int
    minCounts   int
    maxCounts   int
    maxPercentMT  float64

    // Normalization / scaling
    NormMethod  string  // "pearson" or "lognorm"
    PearsonK    float64 //theta
    ScaleFactor float64

    // PCA
    UsePCA  bool // Turn PCA ON or OFF.
    NumPCs  int
    PCAX    int
    PCAY    int
    PCAPlot string

    // Embedding
    EmbedMethod string // "tsne" or "umap"

    // t-SNE
    TSNEDims       int
    TSNEPerplexity float64
    TSNETheta      float64
    TSNEIter       int
    TSNEPlot       string

    // UMAP
    UMAPNeighbors int
    UMAPMinDist   float64
    UMAPMetric    string
    UMAPPlot      string

    // Output filenames
    PCAScoresCSV string
    UMAPOutCSV   string
}

// runPipeline added switchcase property to the original main func to support Rshiny integration.
// Amy Ji (and ChatGPT) - 12/04/2025
func runPipeline(cfg PipelineConfig) error {
    dataset, err := ParseCountMatrixFromFile(cfg.DataPath)
    if err != nil {
        return fmt.Errorf("parse count matrix: %w", err)
    }
    dataset.ImportSummary()

    filtered := dataset.FilterBy(
        cfg.minFeatures,
        cfg.maxFeatures,
        cfg.minCounts,
        cfg.maxCounts,
        cfg.maxPercentMT,
    )
    filtered.ImportSummary()

    fmt.Print("Building em...")
    em, _, _ := BuildMatrix(filtered)
    fmt.Println(" done.")

    // ---------- NORMALIZATION SWITCH ----------
    var normalized *ExpressionMatrix
    switch strings.ToLower(cfg.NormMethod) {
    case "pearson":
        fmt.Println("Normalizing with Pearson residuals...")
        normalized = em.Pearson(cfg.PearsonK)
    case "lognorm":
        fmt.Println("Normalizing with log-normalization...")
        normalized = em.LogNormalize(cfg.ScaleFactor)
    default:
        return fmt.Errorf("unknown normalization method: %q", cfg.NormMethod)
    }
    if normalized == nil {
        return fmt.Errorf("normalization produced nil ExpressionMatrix")
    }
    fmt.Println("Normalization done.")

    fmt.Print("Scaling...")
    normalized.ScaleData(cfg.ScaleFactor)
    fmt.Println(" done.")

    // ---------- DIMENSION REDUCTION + EMBEDDING ----------
    if cfg.UsePCA {
        // PCA path
        fmt.Print("Running PCA...")
        pcs := normalized.PCA(cfg.NumPCs)
        fmt.Println(" done.")
        fmt.Println("PC variances:", pcs.variances)

        fmt.Println("Plotting PCA...")
        if err := pcs.PlotPCAScatter(cfg.PCAPlot, cfg.PCAX, cfg.PCAY); err != nil {
            return fmt.Errorf("plot PCA: %w", err)
        }
        fmt.Println("PCA plot saved as", cfg.PCAPlot)

        switch strings.ToLower(cfg.EmbedMethod) {
        case "tsne":
            if err := runTSNEOnDense(pcs.scores, cfg); err != nil {
                return err
            }
        case "umap":
            if err := runUMAPOnDense(pcs.scores, cfg); err != nil {
                return err
            }
        case "both":
            if err := runTSNEOnDense(pcs.scores, cfg); err != nil {
                return err
            }
            if err := runUMAPOnDense(pcs.scores, cfg); err != nil {
                return err
            }
        default:
            return fmt.Errorf("unknown embed method: %q", cfg.EmbedMethod)
        }

    } else {
        // No PCA: work directly on normalized ExpressionMatrix.
        X := normalized.ToDense()

        switch strings.ToLower(cfg.EmbedMethod) {
        case "tsne":
            if err := runTSNEOnDense(X, cfg); err != nil {
                return err
            }
        case "umap":
            if err := runUMAPOnDense(X, cfg); err != nil {
                return err
            }
        case "both":
            if err := runTSNEOnDense(X, cfg); err != nil {
                return err
            }
            if err := runUMAPOnDense(X, cfg); err != nil {
                return err
            }
        default:
            return fmt.Errorf("unknown embed method: %q", cfg.EmbedMethod)
        }
    }

    return nil
}




// Amy Ji (mostly Chatgpt) -12/04/2025
// for Rshiny app
func main() {
    var cfg PipelineConfig

    // ---------- Data / QC ----------
    flag.StringVar(&cfg.DataPath, "data", "data/scRNA_dataset.csv", "path to count matrix CSV")

    flag.IntVar(&cfg.minFeatures, "minFeatures", 200, "minimum features (genes) per cell")
    flag.IntVar(&cfg.maxFeatures, "maxFeatures", 2500, "maximum features (genes) per cell")
    flag.IntVar(&cfg.minCounts, "minCounts", 500, "minimum counts per cell")
    flag.IntVar(&cfg.maxCounts, "maxCounts", 5000, "maximum counts per cell")
    flag.Float64Var(&cfg.maxPercentMT, "maxPercentMT", 0.05, "maximum mitochondrial fraction (0–1)")

    // ---------- Normalization / scaling ----------
    flag.StringVar(&cfg.NormMethod, "norm", "pearson", "normalization method: pearson|lognorm")
    flag.Float64Var(&cfg.PearsonK, "pearsonK", 100.0, "Pearson residual K (theta)")
    flag.Float64Var(&cfg.ScaleFactor, "scaleFactor", 10, "scale factor for scaling")

    // ---------- PCA ----------
    flag.BoolVar(&cfg.UsePCA, "usePCA", true, "whether to run PCA before embedding")
    flag.IntVar(&cfg.NumPCs, "pcs", 50, "number of principal components")
    flag.IntVar(&cfg.PCAX, "pcax", 0, "PCA x-axis PC index (0-based)")
    flag.IntVar(&cfg.PCAY, "pcay", 1, "PCA y-axis PC index (0-based)")
    flag.StringVar(&cfg.PCAPlot, "pcaPlot", "pca.png", "output PCA plot filename")

    // ---------- Embedding choice ----------
    flag.StringVar(&cfg.EmbedMethod, "embed", "tsne", "embedding method: tsne|umap|both")

    // ---------- t-SNE ----------
    flag.IntVar(&cfg.TSNEDims, "tsneDims", 2, "t-SNE output dimensions")
    flag.Float64Var(&cfg.TSNEPerplexity, "tsnePerplexity", 30, "t-SNE perplexity")
    flag.Float64Var(&cfg.TSNETheta, "tsneTheta", 200, "t-SNE learning rate / theta")
    flag.IntVar(&cfg.TSNEIter, "tsneIter", 1000, "t-SNE max iterations")
    flag.StringVar(&cfg.TSNEPlot, "tsnePlot", "tsne.png", "output t-SNE plot filename")

    // ---------- UMAP ----------
    flag.IntVar(&cfg.UMAPNeighbors, "umapNeighbors", 30, "UMAP n_neighbors")
    flag.Float64Var(&cfg.UMAPMinDist, "umapMinDist", 0.3, "UMAP min_dist")
    flag.StringVar(&cfg.UMAPMetric, "umapMetric", "euclidean", "UMAP distance metric")
    flag.StringVar(&cfg.UMAPPlot, "umapPlot", "umap.png", "output UMAP plot filename")

    // CSV paths for UMAP I/O
    flag.StringVar(&cfg.PCAScoresCSV, "pcaScores", "pca_scores.csv", "CSV file for UMAP input")
    flag.StringVar(&cfg.UMAPOutCSV, "umapOut", "umap_out.csv", "CSV file for UMAP output")

    flag.Parse()

    if err := runPipeline(cfg); err != nil {
        log.Fatalf("pipeline error: %v", err)
    }
}



// Old func main (Keep for now)
/*
//Amy Ji - 11/1/2025
func main() {
	// load dataset, returns CountMatrix struct
	dataset, err := ParseCountMatrixFromFile("data/scRNA_dataset.csv")
	if err != nil {
		log.Fatalf("Error: %v", err)
	}
	dataset.ImportSummary()

	// perform QC filtering

	filtered := dataset.FilterBy(200, 2500, 500, 5000, 0.05)
	filtered.ImportSummary()

	fmt.Print("Building em...")
	em, _, _ := BuildMatrix(filtered)
	fmt.Println(" done.")

	fmt.Print("Normalizing...")
	normalized := em.Pearson(100)
	fmt.Println(" done.")

	// fmt.Print("Scaling...")
	em.ScaleData(10)
	// fmt.Println(" done.")

	// ================ Run PCA ====================
	fmt.Print("Running PCA...")
	pcs := normalized.PCA(50)
	fmt.Println(" done.")
	fmt.Println("PC variances:", pcs.variances)

	fmt.Println("Plotting PCA...")
	err = pcs.PlotPCAScatter("pca.png", 0, 1)
	if err != nil {
		panic(err)
	}
	fmt.Println("PCA plot saved as pca.png")

	// ================= Run TSNE ====================
	fmt.Println("Running t-SNE")
	tsneResult := pcs.TSNE(2,30,200,1000) // to cluster after tSNE, call variable name tsneResult
	if err != nil {
		log.Fatal(err)
	}
	t_title:="tsne"
	t_xlabel:="tsne_1"
	t_ylabel:="tsne_2"
	PlotEmb(tsneResult.scores,"tsne.png",t_title,t_xlabel,t_ylabel )
	fmt.Println("TSNE plot saved as tsne.png")
	
	// ================== Run UMAP ====================
	fmt.Println("converting scores to csv file")
	SavePCAScoresForR(pcs,"pca_scores.csv")

	fmt.Println("Running UMAP in R")
	// set UMAP parameters here!
	if err := RunRUMAP("umap_script.R", "pca_scores.csv", "umap_out.csv", 30, 0.3, "euclidean"); err != nil {
    	log.Fatal(err)
	}

	// Read UMAP embedding back into Go
	// umapCells are the last column, checkout umap_out.csv file.
	// It is omitted here, but we may need the cell barcode info in the future.
	umapResult,_, err := LoadUMAPFromCSV("umap_out.csv") // to cluster after umap, call variable name umapResult
	if err != nil {
    	log.Fatal(err)
	}

	fmt.Println("Plotting UMAP")
	u_title:="umap"
	u_xlabel:="umap_1"
	u_ylabel:="umap_2"
	PlotEmb(umapResult,"umap.png", u_title,u_xlabel,u_ylabel)

}
*/