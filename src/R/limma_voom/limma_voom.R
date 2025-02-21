library(limma)
library(edgeR)
library(clusterProfiler)

args <- commandArgs(trailingOnly = TRUE)
input_file <- args[1]
output_dir <- args[2]
plot_file <- args[3]
go_dir <- args[4]

df <- read.csv(input_file, row.names=1, check.names=FALSE)

metadata <- df[, c("status", "overall_survival", "cluster")]
expr_data <- df[, !(colnames(df) %in% c("status", "overall_survival", "cluster"))]

metadata$cluster <- as.factor(metadata$cluster)

if (length(unique(metadata$cluster)) < 4) {
  stop("Error: Clustering variable must have at least four unique values for contrasts.")
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
contrast_matrix <- makeContrasts(
  High_vs_Low = Cluster3 - Cluster0,
  LowMiddle_vs_Low = Cluster1 - Cluster0,
  HighMiddle_vs_Low = Cluster2 - Cluster0,
  High_vs_LowMiddle = Cluster3 - Cluster1,
  High_vs_HighMiddle = Cluster3 - Cluster2,
  HighMiddle_vs_LowMiddle = Cluster2 - Cluster1,
  levels = design
)
fit <- contrasts.fit(lmFit(voom_data, design), contrast_matrix)
fit <- eBayes(fit)

perform_GO_enrichment <- function(results, comparison_name) {
  significant_genes <- results[results$P.Value < 0.01, "Gene"]
  
  if (length(significant_genes) > 0) {
    enrich_result <- enrichGO(
      gene = significant_genes,
      OrgDb = "org.Hs.eg.db",
      keyType = "SYMBOL",
      ont = "BP",
      pAdjustMethod = "BH",
      pvalueCutoff = 0.05
    )
    
    enrich_df <- as.data.frame(enrich_result)
    
    output_go_file <- paste0(go_dir, "GO_", comparison_name, ".csv")
    write.csv(enrich_df, file=output_go_file, row.names=FALSE)
    
    print(paste("Top GO Terms for", comparison_name))
    print(head(enrich_df, 5))
  } else {
    print(paste("No significant genes for GO enrichment in", comparison_name))
  }
}

results_high_vs_low <- topTable(fit, coef="High_vs_Low", adjust="fdr", number=Inf)
results_lowMiddle_vs_low <- topTable(fit, coef="LowMiddle_vs_Low", adjust="fdr", number=Inf)
results_highMiddle_vs_low <- topTable(fit, coef="HighMiddle_vs_Low", adjust="fdr", number=Inf)
results_high_vs_lowMiddle <- topTable(fit, coef="High_vs_LowMiddle", adjust="fdr", number=Inf)
results_high_vs_highMiddle <- topTable(fit, coef="High_vs_HighMiddle", adjust="fdr", number=Inf)
results_highMiddle_vs_lowMiddle <- topTable(fit, coef="HighMiddle_vs_LowMiddle", adjust="fdr", number=Inf)

results_high_vs_low$Gene <- rownames(results_high_vs_low)
results_lowMiddle_vs_low$Gene <- rownames(results_lowMiddle_vs_low)
results_highMiddle_vs_low$Gene <- rownames(results_highMiddle_vs_low)
results_high_vs_lowMiddle$Gene <- rownames(results_high_vs_lowMiddle)
results_high_vs_highMiddle$Gene <- rownames(results_high_vs_highMiddle)
results_highMiddle_vs_lowMiddle$Gene <- rownames(results_highMiddle_vs_lowMiddle)

write.csv(results_high_vs_low, file=paste0(output_dir, "High_vs_Low.csv"), row.names=FALSE)
write.csv(results_lowMiddle_vs_low, file=paste0(output_dir, "LowMiddle_vs_Low.csv"), row.names=FALSE)
write.csv(results_highMiddle_vs_low, file=paste0(output_dir, "HighMiddle_vs_Low.csv"), row.names=FALSE)
write.csv(results_high_vs_lowMiddle, file=paste0(output_dir, "High_vs_LowMiddle.csv"), row.names=FALSE)
write.csv(results_high_vs_highMiddle, file=paste0(output_dir, "High_vs_HighMiddle.csv"), row.names=FALSE)
write.csv(results_highMiddle_vs_lowMiddle, file=paste0(output_dir, "HighMiddle_vs_LowMiddle.csv"), row.names=FALSE)

perform_GO_enrichment(results_high_vs_low, "High_vs_Low")
perform_GO_enrichment(results_lowMiddle_vs_low, "LowMiddle_vs_Low")
perform_GO_enrichment(results_highMiddle_vs_low, "HighMiddle_vs_Low")
perform_GO_enrichment(results_high_vs_lowMiddle, "High_vs_LowMiddle")
perform_GO_enrichment(results_high_vs_highMiddle, "High_vs_HighMiddle")
perform_GO_enrichment(results_highMiddle_vs_lowMiddle, "HighMiddle_vs_LowMiddle")