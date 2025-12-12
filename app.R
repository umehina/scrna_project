# ChatGpt- 12/04/2025
library(shiny)

# Path to your compiled Go binary
# Make sure this path is correct relative to where you run the app
GO_BIN <- "./sc_pipeline"   # e.g. "sc_pipeline.exe" on Windows

ui <- fluidPage(
  titlePanel("Single-cell Pipeline (Go + R Shiny)"),
  
  sidebarLayout(
    sidebarPanel(
      h3("Inputs"),
      
      # --- Data ---
      textInput("data_path", "Data path (CSV)", 
                value = "data/scRNA_dataset.csv"),
      
      tags$hr(),
      h4("QC Filters"),
      numericInput("min_features", "Min features per cell", 200),
      numericInput("max_features", "Max features per cell", 2500),
      numericInput("min_counts",   "Min counts per cell", 500),
      numericInput("max_counts",   "Max counts per cell", 5000),
      numericInput("max_percent_mt", "Max percent mitochondrial (0–1)", 
                   value = 0.05, step = 0.01),
      
      tags$hr(),
      h4("Normalization & Scaling"),
      selectInput("norm", "Normalization method", 
                  choices = c("Pearson residuals" = "pearson",
                              "Log-normalization" = "lognorm"),
                  selected = "pearson"),
      numericInput("pearson_k", "Pearson K (theta)", 100),
      numericInput("scale_factor", "Scale factor", 10),
      
      tags$hr(),
      h4("PCA"),
      checkboxInput("use_pca", "Use PCA before embedding", value = TRUE),
      numericInput("n_pcs", "Number of PCs", 50, min = 2, step = 1),
      numericInput("pca_x", "PCA X-axis PC index (0-based)", 0, step = 1),
      numericInput("pca_y", "PCA Y-axis PC index (0-based)", 1, step = 1),
      
      tags$hr(),
      h4("Embedding"),
      selectInput("embed", "Embedding method",
                  choices = c("t-SNE only" = "tsne",
                              "UMAP only" = "umap",
                              "Both t-SNE and UMAP" = "both"),
                  selected = "tsne"),
      
      tags$hr(),
      h4("t-SNE parameters"),
      numericInput("tsne_dims", "Output dimensions", 2, min = 2, max = 3),
      numericInput("tsne_perp", "Perplexity", 30),
      numericInput("tsne_theta", "Learning rate / theta", 200),
      numericInput("tsne_iter", "Max iterations", 1000),
      
      tags$hr(),
      h4("UMAP parameters"),
      numericInput("umap_neighbors", "n_neighbors", 30, min = 2),
      numericInput("umap_mindist", "min_dist", 0.3, step = 0.05),
      textInput("umap_metric", "Metric", value = "euclidean"),
      
      tags$hr(),
      actionButton("run", "Run pipeline", class = "btn btn-primary")
    ),
    
    mainPanel(
      h3("Outputs"),
      
      conditionalPanel(
        condition = "input.use_pca == true",
        h4("PCA plot"),
        imageOutput("pca_plot")
      ),
      
      conditionalPanel(
        condition = "input.embed == 'tsne' || input.embed == 'both'",
        h4("t-SNE plot"),
        imageOutput("tsne_plot")
      ),
      
      conditionalPanel(
        condition = "input.embed == 'umap' || input.embed == 'both'",
        h4("UMAP plot"),
        imageOutput("umap_plot")
      ),
      
      tags$hr(),
      h4("Go pipeline log"),
      verbatimTextOutput("go_log")
    )
  )
)

server <- function(input, output, session) {
  go_log <- reactiveVal("")
  
  observeEvent(input$run, {
    # Build CLI args to match Go flags exactly
    args <- c(
      "-data",          input$data_path,
      "-minFeatures",   input$min_features,
      "-maxFeatures",   input$max_features,
      "-minCounts",     input$min_counts,
      "-maxCounts",     input$max_counts,
      "-maxPercentMT",  input$max_percent_mt,
      
      "-norm",          input$norm,
      "-pearsonK",      input$pearson_k,
      "-scaleFactor",   input$scale_factor,
      
      "-usePCA",        if (isTRUE(input$use_pca)) "true" else "false",
      "-pcs",           input$n_pcs,
      "-pcax",          input$pca_x,
      "-pcay",          input$pca_y,
      "-pcaPlot",       "pca.png",
      
      "-embed",         input$embed,
      
      "-tsneDims",       input$tsne_dims,
      "-tsnePerplexity", input$tsne_perp,
      "-tsneTheta",      input$tsne_theta,
      "-tsneIter",       input$tsne_iter,
      "-tsnePlot",       "tsne.png",
      
      "-pcaScores",      "pca_scores.csv",
      "-umapOut",        "umap_out.csv",
      "-umapNeighbors",  input$umap_neighbors,
      "-umapMinDist",    input$umap_mindist,
      "-umapMetric",     input$umap_metric,
      "-umapPlot",       "umap.png"
    )
    
    # Run the Go binary and capture stdout+stderr
    out <- NULL
    withProgress(message = "Running Go pipeline...", value = 0, {
      incProgress(0.3)
      out <- tryCatch(
        system2(GO_BIN, args = args, stdout = TRUE, stderr = TRUE),
        error = function(e) paste("Error running Go binary:", e$message)
      )
      incProgress(0.7)
    })
    
    go_log(paste(out, collapse = "\n"))
    
    # Render logs
    output$go_log <- renderText({
      go_log()
    })
    
    # Render PCA plot (if PCA is used and file exists)
    output$pca_plot <- renderImage({
      if (!isTRUE(input$use_pca) || !file.exists("pca.png")) {
        return(NULL)
      }
      list(
        src = "pca.png",
        contentType = "image/png",
        alt = "PCA plot"
      )
    }, deleteFile = FALSE)
    
    # Render t-SNE plot
    output$tsne_plot <- renderImage({
      if (!(input$embed %in% c("tsne", "both")) || !file.exists("tsne.png")) {
        return(NULL)
      }
      list(
        src = "tsne.png",
        contentType = "image/png",
        alt = "t-SNE plot"
      )
    }, deleteFile = FALSE)
    
    # Render UMAP plot
    output$umap_plot <- renderImage({
      if (!(input$embed %in% c("umap", "both")) || !file.exists("umap.png")) {
        return(NULL)
      }
      list(
        src = "umap.png",
        contentType = "image/png",
        alt = "UMAP plot"
      )
    }, deleteFile = FALSE)
  })
}

shinyApp(ui, server)
