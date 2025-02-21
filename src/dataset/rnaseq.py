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

    q25 = merged_df['overall_survival'].quantile(0.25)
    q50 = merged_df['overall_survival'].quantile(0.50)
    q75 = merged_df['overall_survival'].quantile(0.75)

    merged_df['cluster'] = 2
    merged_df.loc[merged_df['overall_survival'] <= q25, 'cluster'] = 0
    merged_df.loc[(merged_df['overall_survival'] > q25) & (merged_df['overall_survival'] <= q50), 'cluster'] = 1
    merged_df.loc[(merged_df['overall_survival'] > q50) & (merged_df['overall_survival'] <= q75), 'cluster'] = 2
    merged_df.loc[merged_df['overall_survival'] > q75, 'cluster'] = 3

    merged_df.to_csv("processed/GeneExpressionData.csv", index=False)