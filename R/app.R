library(shiny)
library(ggplot2)
library(dplyr)

options(shiny.maxRequestSize = 200 * 1024^2)  # 50 MB

ui <- fluidPage(
  titlePanel("scRNA-seq pipeline (Go backend + Shiny)"),
  
  sidebarLayout(
    sidebarPanel(
      fileInput(
        "datafile",
        "Upload count matrix CSV (optional – otherwise use default in data/)",
        accept = c(".csv")
      ),
      
      radioButtons(
        "norm",
        "Normalization method",
        choices = c(
          "Pearson residuals" = "pearson",
          "LogNormalize"      = "lognorm"
        ),
        selected = "pearson"
      ),
      
      sliderInput(
        "npcs",
        "Number of principal components",
        min   = 2,
        max   = 50,
        value = 30,
        step  = 1
      ),
      
      radioButtons(
        "embed",
        "Embedding method",
        choices = c(
          "UMAP" = "umap",
          "t-SNE" = "tsne"
        ),
        selected = "umap"
      ),
      
      hr(),
      h4("Leiden clustering"),
      sliderInput(
        "k",
        "k (neighbors in KNN graph)",
        min   = 5,
        max   = 100,
        value = 25,
        step  = 1
      ),
      sliderInput(
        "resolution",
        "Resolution",
        min   = 0.1,
        max   = 2.0,
        value = 0.5,
        step  = 0.1
      ),
      
      hr(),
      h4("UMAP parameters"),
      helpText("Only used when embedding method is UMAP."),
      sliderInput(
        "umap_n_neighbors",
        "UMAP n_neighbors",
        min   = 5,
        max   = 100,
        value = 30,
        step  = 1
      ),
      sliderInput(
        "umap_lr",
        "UMAP learning rate",
        min   = 0.01,
        max   = 1.0,
        value = 0.1,
        step  = 0.01
      ),
      
      hr(),
      h4("t-SNE parameters"),
      helpText("Only used when embedding method is t-SNE."),
      sliderInput(
        "tsne_perp",
        "t-SNE perplexity",
        min   = 5,
        max   = 100,
        value = 30,
        step  = 1
      ),
      sliderInput(
        "tsne_lr",
        "t-SNE learning rate",
        min   = 10,
        max   = 500,
        value = 100,
        step  = 10
      ),
      sliderInput(
        "tsne_iter",
        "t-SNE iterations",
        min   = 250,
        max   = 5000,
        value = 1000,
        step  = 250
      ),
      
      hr(),
      actionButton("run", "Run pipeline"),
      br(), br(),
      downloadButton("downloadPlot", "Download plot (PNG)")
    ),
    
    mainPanel(
      verbatimTextOutput("status"),
      plotOutput("embedPlot", height = "600px", width = "600px")
    )
  )
)

server <- function(input, output, session) {
  # Infer project root (folder that contains data/ and output/)
  infer_project_root <- function() {
    cwd <- getwd()
    if (file.exists(file.path(cwd, "data", "scRNA_dataset.csv"))) {
      return(normalizePath(cwd))
    }
    parent <- dirname(cwd)
    if (file.exists(file.path(parent, "data", "scRNA_dataset.csv"))) {
      return(normalizePath(parent))
    }
    warning("Could not confidently infer project root; using current working directory.")
    normalizePath(cwd)
  }
  
  project_root <- infer_project_root()
  go_binary <- file.path(project_root, "scrna_project")
  
  if (!file.exists(go_binary)) {
    warning("Go binary not found at: ", go_binary,
            "\nBuild with: go build -o scrna_project .")
  }
  
  rv <- reactiveValues(
    status   = "Idle. Upload data (optional), set parameters, and click 'Run pipeline'.",
    combined = NULL,
    xcol     = NULL,
    ycol     = NULL
  )
  
  output$status <- renderText(rv$status)
  
  make_embedding_plot <- function(df, xcol, ycol, embed_method) {
    method_label <- ifelse(embed_method == "umap", "UMAP", "t-SNE")
    x_label <- if (embed_method == "umap") "UMAP1" else "t-SNE1"
    y_label <- if (embed_method == "umap") "UMAP2" else "t-SNE2"
    
    ggplot(df, aes_string(x = xcol, y = ycol, color = "cluster")) +
      geom_point(size = 0.8, alpha = 0.8) +
      coord_equal() +
      theme_minimal(base_size = 14) +
      labs(
        title = paste(method_label, "embedding with Leiden clusters"),
        x = x_label,
        y = y_label,
        color = "Cluster"
      )
  }
  
  observeEvent(input$run, {
    rv$status <- "Running Go pipeline..."
    
    # Decide which data path to use
    if (!is.null(input$datafile)) {
      data_path <- input$datafile$datapath  # uploaded temp file
    } else {
      data_path <- file.path(project_root, "data", "scRNA_dataset.csv")
    }
    
    if (!file.exists(data_path)) {
      rv$status <- paste("Data file not found:", data_path)
      rv$combined <- NULL
      rv$xcol <- rv$ycol <- NULL
      return(NULL)
    }
    
    # Build CLI args for Go (MUST match flags in main.go)
    args <- c(
      sprintf("-data=%s", data_path),
      sprintf("-norm=%s", input$norm),
      sprintf("-npcs=%d", input$npcs),
      sprintf("-embed=%s", input$embed),
      sprintf("-k=%d", input$k),
      sprintf("-resolution=%f", input$resolution),
      sprintf("-umap_neighbors=%d", input$umap_n_neighbors),
      sprintf("-umap_lr=%f", input$umap_lr),
      sprintf("-tsne_perp=%f", input$tsne_perp),
      sprintf("-tsne_lr=%f", input$tsne_lr),
      sprintf("-tsne_iter=%d", input$tsne_iter)
    )
    
    cat("Project root:", project_root, "\n")
    cat("Go binary   :", go_binary, "\n")
    cat("Args        :", paste(args, collapse = " "), "\n")
    
    # Run Go from project root so relative paths (output/, etc.) work
    old_wd <- getwd()
    setwd(project_root)
    on.exit(setwd(old_wd), add = TRUE)
    
    exit_status <- system2(
      go_binary,
      args   = args,
      stdout = "",
      stderr = "",
      wait   = TRUE
    )
    
    if (is.numeric(exit_status) && exit_status != 0) {
      rv$status <- sprintf(
        "Go pipeline failed (exit code %d). Check console / Go logs.",
        exit_status
      )
      rv$combined <- NULL
      rv$xcol <- rv$ycol <- NULL
      return(NULL)
    }
    
    # Decide which embedding CSV to read
    embed_file <- if (input$embed == "umap") {
      file.path(project_root, "output", "umap.csv")
    } else {
      file.path(project_root, "output", "tsne.csv")
    }
    label_file <- file.path(project_root, "output", "leiden_export_labels.csv")
    
    if (!file.exists(embed_file)) {
      rv$status <- paste("Pipeline finished, but embedding file not found:", embed_file)
      rv$combined <- NULL
      rv$xcol <- rv$ycol <- NULL
      return(NULL)
    }
    if (!file.exists(label_file)) {
      rv$status <- paste("Pipeline finished, but labels file not found:", label_file)
      rv$combined <- NULL
      rv$xcol <- rv$ycol <- NULL
      return(NULL)
    }
    
    embed_df  <- read.csv(embed_file, stringsAsFactors = FALSE)
    labels_df <- read.csv(label_file, stringsAsFactors = FALSE)
    
    if (!("node" %in% names(embed_df)) || !("node" %in% names(labels_df))) {
      rv$status <- "Embedding or labels CSV missing 'node' column."
      rv$combined <- NULL
      rv$xcol <- rv$ycol <- NULL
      return(NULL)
    }
    
    combined <- embed_df %>%
      inner_join(labels_df, by = "node")
    
    combined$cluster <- as.factor(combined$cluster)
    
    # Pick x/y columns based on method
    if (input$embed == "umap") {
      xcol <- "UMAP1"; ycol <- "UMAP2"
    } else {
      xcol <- "TSNE1"; ycol <- "TSNE2"
    }
    
    # Fallback if columns named differently
    if (!(xcol %in% names(combined)) || !(ycol %in% names(combined))) {
      numeric_cols <- names(combined)[sapply(combined, is.numeric)]
      numeric_cols <- setdiff(numeric_cols, c("node", "cluster"))
      if (length(numeric_cols) >= 2) {
        xcol <- numeric_cols[1]
        ycol <- numeric_cols[2]
      } else {
        rv$status <- "Could not infer embedding columns to plot."
        rv$combined <- NULL
        rv$xcol <- rv$ycol <- NULL
        return(NULL)
      }
    }
    
    rv$combined <- combined
    rv$xcol <- xcol
    rv$ycol <- ycol
    
    rv$status <- sprintf(
      "Success. Norm=%s, PCs=%d, Embed=%s, k=%d, resolution=%.2f, cells=%d.",
      input$norm, input$npcs, input$embed, input$k, input$resolution, nrow(combined)
    )
  })
  
  # On-screen plot
  output$embedPlot <- renderPlot({
    req(rv$combined, rv$xcol, rv$ycol)
    make_embedding_plot(rv$combined, rv$xcol, rv$ycol, input$embed)
  })
  
  # Download handler for PNG
  output$downloadPlot <- downloadHandler(
    filename = function() {
      if (input$embed == "umap") {
        "embedding_umap.png"
      } else {
        "embedding_tsne.png"
      }
    },
    content = function(file) {
      req(rv$combined, rv$xcol, rv$ycol)
      
      p <- make_embedding_plot(rv$combined, rv$xcol, rv$ycol, input$embed)
      
      ggsave(
        filename = file,
        plot     = p,
        width    = 6,
        height   = 6,
        dpi      = 300
      )
    }
  )
  
}

shinyApp(ui, server)
