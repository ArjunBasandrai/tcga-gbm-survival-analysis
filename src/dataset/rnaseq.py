from .datasets import load_df

def load_rna_data(path):
    rna_df = load_df(path)
    return rna_df

def append_survival_data(rna_df, clinical_df):
    merged_df = rna_df.merge(
        clinical_df[['patient_id', 'status', 'overall_survival']],
        on="patient_id",
        how="inner"
    )
    merged_df['status'] = merged_df['status'].astype(int)
    merged_df['overall_survival'] = merged_df['overall_survival'].astype(int)

    merged_df['cluster'] = (merged_df['overall_survival'] > merged_df['overall_survival'].median()).astype(int)
    merged_df.to_csv("processed/GeneExpressionData.csv", index=False)