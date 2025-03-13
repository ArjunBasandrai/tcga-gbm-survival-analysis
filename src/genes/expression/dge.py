import os
import subprocess

def install_R_packages():
    packages_r_script = "src/R/limma_voom/install_packages.R"
    cmd = ["Rscript", packages_r_script]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error installing R packages: {e}")

def perform_dge_go_analysis():
    r_script = "src/R/limma_voom/limma_voom.R"
    input_csv = "processed/GeneExpressionData.csv"
    output_dir = "results/genes/expression/feature_selection/limma_voom/"
    go_path = "results/genes/expression/feature_selection/genetic_ontology_enrichment/"
    plot_path = os.path.join(output_dir, "mean_variance_trend.pdf")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(go_path, exist_ok=True)

    cmd = ["Rscript", r_script, input_csv, output_dir, plot_path, go_path]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running R script: {e}")
        print("If some R packages are missing, please install them manually using python main.py --install-r-packages")
