from .datasets import load_df

def load_rna_data(path):
    rna_df = load_df(path)
    return rna_df