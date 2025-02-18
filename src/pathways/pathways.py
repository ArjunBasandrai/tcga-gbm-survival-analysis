import gseapy as gp

def get_enriched_pathways(mutation_df):
    gene_cols = mutation_df.columns[1:].to_list()
    df = mutation_df[gene_cols]
    cluster_groups = df.groupby("cluster").mean()

    significance_level = 0.05

    enriched_pathways = []
    for cluster, row in cluster_groups.iterrows():
        gene_list = row[row > 0].index.tolist()
        gsea_results = gp.enrichr(gene_list=gene_list, 
                                gene_sets='KEGG_2019_Human', 
                                organism='human', 
                                outdir=None)
        enriched_pathways.append(gsea_results.results.query(f"`Adjusted P-value` < {significance_level}"))
        print(f"Top enriched pathways for Cluster {cluster}:")
        print(gsea_results.results[["Term", "Adjusted P-value"]].head())

    return enriched_pathways