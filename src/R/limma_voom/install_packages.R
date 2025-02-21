packages <- c(
  "limma",
  "edgeR",
  "clusterProfiler",
  "BiocManager"
)

install.packages(setdiff(packages, rownames(installed.packages())), dependencies = TRUE)

if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")

bioc_packages <- c("org.Hs.eg.db")
BiocManager::install(bioc_packages, update = FALSE, ask = FALSE)
