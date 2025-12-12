library(shiny)
library(ggplot2)
library(dplyr)

ui <- fluidPage(
  titlePanel("scRNA-seq pipeline (Go backend + Shiny frontend)"),
  
  sidebarLayout(
    sidebarPanel(
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
      
      actionButton("run", "Run pipeline")
    ),
    
    mainPanel(
      verbatimTextOutput("status"),
      plotOutput("embedPlot", height = "600px")
    )
  )
)

server <- function(input, output, session) {
  # Infer project root (folder that contains data/ and output/)
  infer_project_root <- function() {
    cwd <- getwd()
    
    # If cwd has data/scRNA_dataset.csv, assume cwd is root
    if (file.exists(file.path(cwd, "data", "scRNA_dataset.csv"))) {
      return(normalizePath(cwd))
    }
    
    # If ../data/scRNA_dataset.csv exists, assume parent is root
    parent <- dirname(cwd)
    if (file.exists(file.path(parent, "data", "scRNA_dataset.csv"))) {
      return(normalizePath(parent))
    }
    
    warning("Could not confidently infer project root; using current working directory.")
    normalizePath(cwd)
  }
  
  project_root <- infer_project_root()
  setwd(project_root)  # ensure WD is the project root for Go's relative paths
  go_binary <- file.path(project_root, "scrna_project")
  
  if (!file.exists(go_binary)) {
    warning("Go binary not found at: ", go_binary,
            "\nBuild it with: go build -o scrna_project .  in the project root.")
  }
  
  rv <- reactiveValues(
    status   = "Idle. Choose options and click 'Run pipeline'.",
    combined = NULL,
    xcol     = NULL,
    ycol     = NULL
  )
  
  output$status <- renderText(rv$status)
  
  observeEvent(input$run, {
    rv$status <- "Running Go pipeline..."
    
    # Build CLI args
    args <- c(
      sprintf("-norm=%s", input$norm),
      sprintf("-npcs=%d", input$npcs),
      sprintf("-embed=%s", input$embed)
    )
    
    cat("Project root:", project_root, "\n")
    cat("Go binary   :", go_binary, "\n")
    cat("Args        :", paste(args, collapse = " "), "\n")
    
    # Run Go with working directory set to project_root
    exit_status <- system2(
      go_binary,
      args   = args,
      stdout = "",
      stderr = "",
      wait   = TRUE,
    )
    
    # system2 returns 0 on success; non-zero indicates failure
    if (is.numeric(exit_status) && exit_status != 0) {
      rv$status <- sprintf(
        "Go pipeline failed (exit code %d). Check console / Go logs.",
        exit_status
      )
      rv$combined <- NULL
      rv$xcol <- rv$ycol <- NULL
      return(NULL)
    }
    
    # Decide which embedding to read
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
    
    # Fallback if names differ
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
      "Success. Normalization = %s, PCs = %d, Embedding = %s. Cells = %d.",
      input$norm, input$npcs, input$embed, nrow(combined)
    )
  })
  
  output$embedPlot <- renderPlot({
    req(rv$combined, rv$xcol, rv$ycol)
    
    method_label <- ifelse(input$embed == "umap", "UMAP", "t-SNE")
    
    ggplot(rv$combined, aes_string(x = rv$xcol, y = rv$ycol, color = "cluster")) +
      geom_point(size = 0.8, alpha = 0.8) +
      coord_equal() +
      theme_minimal(base_size = 14) +
      labs(
        title = paste(method_label, "embedding with Leiden clusters"),
        x = rv$xcol,
        y = rv$ycol,
        color = "Cluster"
      )
  })
}

shinyApp(ui, server)
