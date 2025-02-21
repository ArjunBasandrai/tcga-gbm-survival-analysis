import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt

import os

def predict_patient_survival():
    if os.path.exists("models/mutation_cox_model.pkl") is False:
        raise ValueError("Please train the Cox Regression model first")
    
    with open('models/mutation_cox_model.pkl', 'rb') as f:
        model = pkl.load(f)

    predict = True
    print("press Enter after a prediction to make another prediction. Type 'exit' to exit")
    while predict:
        patient_name = input("Enter patient name: ")
        print("For the following genes, enter (0/1) if the gene is mutated/not mutated")
        data = {
            'CREB3L3': 0,
            'PIK3R2': 0,
            'PTEN': 0,
            'NR1H2': 0,
            'PPARGC1B': 0,
            'SLC27A6': 0,
        }

        for gene in data.keys():
            data[gene] = int(input(f"{gene}: "))
        
        processed_name = "_".join(patient_name.split(" "))
        patient_dir = f"predictions/mutation/{processed_name}"
        os.makedirs(patient_dir, exist_ok=True)
        
        data = pd.DataFrame(data, index=[0])
        data.to_csv(f"predictions/mutation/{processed_name}/mutations_data.csv", index=False)

        _, ax = plt.subplots(figsize=(10, 5))
        prediction = model.predict_survival_function(data)
        prediction.plot(ax=ax, color='teal')
        ax.get_legend().remove()
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Survival Probability", fontsize=12)
        ax.set_title(f"Survival Curve for {patient_name}", fontsize=14)
        plt.savefig(f"predictions/mutation/{processed_name}/survival_plot.png")
        plt.show()

        inp = input("\n> ")
        if inp.lower() == 'exit':
            predict = False
        print()

