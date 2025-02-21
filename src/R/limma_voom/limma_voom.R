library(limma)
library(edgeR)

args <- commandArgs(trailingOnly = TRUE)
input_file <- args[1]
output_file <- args[2]
plot_file <- args[3]

df <- read.csv(input_file, row.names=1, check.names=FALSE)

metadata <- df[, c("status", "overall_survival", "cluster")]
expr_data <- df[, !(colnames(df) %in% c("status", "overall_survival", "cluster"))]

metadata$cluster <- as.factor(metadata$cluster)

if (length(unique(metadata$cluster)) < 2) {
  stop("Error: Clustering variable must have at least two unique values for contrasts.")
}

expr_data <- t(expr_data)

dge <- DGEList(counts=as.matrix(expr_data))

levels(metadata$cluster) <- paste0("Cluster", levels(metadata$cluster))

design <- model.matrix(~ 0 + metadata$cluster)
colnames(design) <- levels(metadata$cluster)

pdf(plot_file)
voom_data <- voom(dge, design, plot=TRUE)
dev.off()

fit <- lmFit(voom_data, design)
contrast_matrix <- makeContrasts(Cluster1 - Cluster0, levels=design)
fit <- contrasts.fit(fit, contrast_matrix)
fit <- eBayes(fit)

results <- topTable(fit, coef=1, adjust="fdr", number=Inf)
results$Gene <- rownames(results)

write.csv(results, file=output_file, row.names=FALSE)

print("Differential Expression Analysis Completed. Results saved to limma_voom_results.csv")