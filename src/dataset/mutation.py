from .datasets import load_df

def load_mutation_data(path):
    mutation_df = load_df(path)
    mutation_df[mutation_df.columns[1:]]=mutation_df[mutation_df.columns[1:]].astype(int)
    
    return mutation_df
