from .datasets import load_df
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_clinical_data(path):
    encoder = LabelEncoder()

    clinical_df = load_df(path)
    clinical_df.drop(['overallsurvival', 'race', 'ethnicity'], axis=1, inplace=True)
    
    clinical_df = clinical_df.dropna()
    clinical_df['radiation_therapy'] = clinical_df['radiation_therapy'].map({"yes": 1, "no": 0})
    clinical_df['gender'] = clinical_df['gender'].map({"male": 1, "female": 0})
    clinical_df['histological_type'] = encoder.fit_transform(clinical_df['histological_type'])
    clinical_df[['radiation_therapy', 'gender', 'years_to_birth', 'overall_survival', 'status']] = clinical_df[['radiation_therapy', 'gender', 'years_to_birth', 'overall_survival', 'status']].astype(np.int16)

    clinical_df = clinical_df.rename({"years_to_birth": "age"}, axis=1)

    return clinical_df, encoder