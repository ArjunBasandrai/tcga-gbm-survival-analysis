import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer

class ClinicalData:
    def __init__(self, path: str):
        self.df: pd.DataFrame = self.load_and_transpose_df(path)
        self.label_mappings: dict = {}

    def load_and_transpose_df(self, dataset: str) -> pd.DataFrame:
        df = pd.read_csv(dataset).T
        df.columns = df.iloc[1]
        df = df[2:]
        df = df.reset_index()
        df = df.rename(columns={'index': 'patient_id'})
        df.columns.name = None
        return df
    
    def select_cols(self) -> None:
        self.df.drop(['overallsurvival', 'ethnicity', 'radiation_therapy', 'race'], axis=1, inplace=True)
    
    def drop_na_columns(self) -> None:
        self.df.dropna(inplace=True)    
         
    def label_encode_clinical_data(self) -> None:
        categorical_columns = ['histological_type', 'gender']
    
        for column in categorical_columns:
            if column in self.df.columns:
                unique_values = [value for value in self.df[column].unique() if pd.notna(value)]
                self.label_mappings[column] = {value: idx for idx, value in enumerate(sorted(unique_values))}
    
        def to_id(row: pd.Series) -> pd.Series:
            for column in categorical_columns:
                if column in row.index and pd.notna(row[column]):
                    row[column] = self.label_mappings[column][row[column]]
            return row
    
        self.df = self.df.apply(to_id, axis=1)
    
    def preprocess_data(self) -> None:
        self.select_cols()
        self.drop_na_columns()
        self.label_encode_clinical_data()

    def get_label_decoded(self) -> pd.DataFrame:
        df_copy = self.df.copy(deep=True)
        for column, mapping in self.label_mappings.items():
            inv_mapping = {v: k for k, v in mapping.items()}
            df_copy[column] = df_copy[column].apply(
                lambda x: inv_mapping[x] if (pd.notnull(x) and x in inv_mapping) else x
            )
        return df_copy

    def __call__(self) -> pd.DataFrame:
        return self.df