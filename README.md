# Molecular Property Prediction: Comparing Tabular and Graph-Based Models for `pGI50`

## Project Overview

This project focuses on developing and comparing machine learning models for predicting `pGI50`, a critical biological activity metric for chemical compounds. The `pGI50` value of a drug is calculated as:  

$$pGI50 = -\log_{10}(\text{GI50 in Molars (M)})$$

where `GI50` is **_the concentration of a drug required to inhibit the growth of cancer cells by 50%_**. **Lower `GI50` values (and conversely, HIGHER `pGI50` values) indicate more potency** as the drug is required in comparatively lower amounts.

The central hypothesis explored is whether models explicitly designed to leverage the inherent **graph structure of molecular data (Graph Neural Networks - GNNs)** offer a significant advantage over more generalized machine learning approaches (XGBoost and Multi-Layer Perceptrons - MLPs) that rely on engineered tabular features.

This repository demonstrates a complete end-to-end machine learning pipeline, from raw data acquisition and rigorous feature engineering to hyperparameter optimization, model training, and a detailed comparative performance analysis.

## Key Features

-   **Data-Driven Approach:** Utilizes publicly available molecular activity [data from ChEMBL](https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/).
-   **Comprehensive Feature Engineering:** Explores both traditional RDKit descriptors/fingerprints and graph-based representations.
-   **Diverse Modeling Strategies:** Implements:
    -   **XGBoost:** A powerful gradient boosting framework for tabular data.
    -   **Multi-Layer Perceptron (MLP):** A foundational neural network for tabular data.
    -   **Graph Neural Network (GNN):** A specialized deep learning architecture capable of directly processing molecular graph structures.
-   **Hyperparameter Optimization:** Utilizes Optuna for systematic and efficient tuning of all model architectures.
-   **Rigorous Evaluation:** Compares models based on standard regression metrics (RMSE, R2) on an unseen test set.
-   **Reproducibility:** Project setup includes environment configurations and Git commit hash tracking for models.
-   **Modular Codebase:** Organized into clear Jupyter notebooks and a `src` directory for custom MLP and GNN class architetures.

## Project Structure

The repository is organized into the following main directories and notebooks:

```
├── notebooks/
│   ├── 00_Build_Clean_Dataset.ipynb                # Data Cleaning, Initial EDA
│   ├── 01_Engineer_Molecular_Features.ipynb        # Tabular Feature Engineering
│   ├── 02_Split_Data_For_Model_Training.ipynb      # Universal Data Splitting (X, y for all models)
│   ├── 03_Build_XGB_Model.ipynb
│   ├── 04_Build_MLP_Model.ipynb
│   ├── 05_Build_GNN_Model.ipynb                    # Outputs PyG graph objects here as well
│   └── 06_Compare_Final_Model_Performances.ipynb
├── src/
│   └── models/
│       ├── mlp_models.py                           # Custom MLP model class
│       └── gnn_models.py                           # Custom GNN model class
├── data/
│   ├── raw/                                        # Raw downloaded data
│   ├── processed/                                  # Cleaned and preprocessed data (output of notebook 00)
│   ├── features/                                   # Fully feature-engineered tabular data (output of notebook 01)
│   ├── splits/                                     # Universal train/validation/test splits (output of notebook 02)
│   └── pyg_data_graphs/                            # PyTorch Geometric Data graph objects (output of notebook 05)
├── models/
│   ├── gnn/                                        # Trained GNN model state dict (.pt)
│   ├── mlp/                                        # Trained MLP model state dict (.pt)
│   └── xgb/                                        # Trained XGBoost model (.joblib)
├── studies/
│   ├── gnn_study/                                  # GNN Optuna study database
│   ├── mlp_study/                                  # MLP Optuna study database
│   └── xgboost_study/                              # XGBoost Optuna study database
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup and Running the Project

Follow these steps to set up the project environment and run the notebooks:

### 1.  **Clone the repository:**

```bash
git clone https://github.com/Shirshak52/Drug-pGI50-Prediction.git

cd Drug-pGI50-Prediction
```

### 2.  **Create a virtual environment:**

```bash
python -m venv venv

# On Windows
.\venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3.  **Install dependencies:**

```bash
pip install -r requirements.txt
```

**Note on Specific Library Installations:**  
Some libraries, particularly RDKit, PyTorch, and PyTorch Geometric, can sometimes present unique installation challenges due to their dependencies (e.g., CUDA versions, specific `conda` requirements, etc.). If you encounter any issues during `pip install -r`, please refer to the official documentation for:

-   [RDKit Installation Guide](https://www.rdkit.org/docs/Install.html)
-   [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
-   [PyTorch Geometric Installation Guide](https://pytorch-geometric.readthedocs.io/en/2.6.1/install/installation.html)

### 4. **Install external dependencies (for high-resolution molecular visualization):**  
Some external, system-level dependencies must be installed that are used for high-quality plots with crisp molecular images and properly formatted text (such as inline bolding)

#### 4.1. For Crisp Molecule Images (via `cairosvg`):
The `cairosvg` Python library relies on the **Cairo graphics library**.

* **Windows:**
    The most reliable way to get Cairo and its dependencies is often by using [MSYS2](https://www.msys2.org/).
    1.  Download and install MSYS2.
    2.  Open an MSYS2 MinGW 64-bit terminal.
    3.  Install GTK3 (which includes Cairo): `pacman -S mingw-w64-x86_64-gtk3`

    Alternatively, you can try direct GTK+ Runtime installers (e.g., from [gtkd.org/download.html](https://gtkd.org/download.html) for GTK+ 3.24.8 runtime, or the [GTK+ for Windows Runtime Environment Installer GitHub fork](https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer)). Ensure the installation directory's `bin` folder is added to your system's PATH.

* **macOS:**
    Install Cairo via Homebrew (if you don't have Homebrew, install it first from [brew.sh](https://brew.sh/)):
    ```bash
    brew install cairo
    # GTK3 is often implicitly installed or needed for full Cairo functionality:
    # brew install gtk+3
    ```

* **Linux (Debian/Ubuntu-based):**
    Install the Cairo development libraries:
    ```bash
    sudo apt-get update
    sudo apt-get install libcairo2-dev libgirepository1.0-dev
    ```
    For other Linux distributions (e.g., Fedora/RHEL), use your distribution's package manager (e.g., `dnf install cairo-devel gobject-introspection-devel`).

#### 4.2. For Formatted Text and Inline Bolding (via LaTeX):
Matplotlib uses a LaTeX distribution to render complex text and inline formatting (like bolding) when `usetex` is enabled.

* **Windows:**
    Install [MiKTeX](https://miktex.org/download). Choose the "Basic Installer." During installation, ensure it's added to your system's PATH. After the initial installation, MiKTeX should automatically prompt to install necessary packages, or you may need to manually update packages (e.g., `type1cm`, `textcomp`, `underscore`) via the MiKTeX Console.

* **macOS:**
    Install [MacTeX](https://www.tug.org/mactex/) (a comprehensive distribution) or [BasicTeX](https://www.tug.org/mactex/morepackages.html) (a smaller distribution, then install additional packages as needed). After installation, ensure its `bin` directory is in your system's PATH.

* **Linux:**
    Install [TeX Live](https://www.tug.org/texlive/acquire-linux.html). The most straightforward way is often through your distribution's package manager (a full install is usually recommended for simplicity with Matplotlib):
    ```bash
    # For Debian/Ubuntu-based systems
    sudo apt-get update
    sudo apt-get install texlive-full dvipng
    # If 'texlive-full' is too large, you can try 'texlive-base'
    # and then manually install packages like 'texlive-latex-extra',
    # 'texlive-fonts-recommended', 'texlive-pictures', and ensure 'dvipng'
    # or 'dvisvgm' is installed for Matplotlib's image conversion.
    ```
    For other Linux distributions, use their respective package managers (e.g., `sudo dnf install texlive-scheme-full texlive-dvipng` for Fedora/RHEL).

### 5.  **Run Jupyter Notebooks:**
```bash
jupyter notebook
```
Navigate to the `notebooks/` directory and execute the notebooks in sequential order (00 to 06). Each notebook builds upon the outputs of the previous ones.

## Results and Conclusion

The comparative analysis revealed insightful findings regarding the strengths of different modeling approaches for `pGI50` prediction:

| Model   | RMSE   | R2     |
| :------ | :----- | :----- |
| XGBoost | 0.6957 | 0.4951 |
| MLP     | 0.6408 | 0.5715 |
| GNN     | 0.6114 | 0.6100 |

The **Graph Neural Network (GNN) model emerged as the top performer**, achieving the lowest RMSE (0.6114) and the highest R2 score (0.6100) on the unseen test data. This result strongly supports the hypothesis that models specifically designed to leverage the inherent graph structure of molecular data can offer a distinct advantage in property prediction. By directly processing atom-bond connectivity, GNNs effectively capture complex relational patterns that are more challenging for generalized models relying on flattened feature vectors.

While the Multi-Layer Perceptron (MLP) also demonstrated strong performance (RMSE 0.6408, R2 0.5715), outperforming the XGBoost model (RMSE 0.6957, R2 0.4951), the GNN's consistent edge highlights the power of its inherent bias for graph-structured data. This approach is particularly promising for generalization to new/unseen chemical entities, where handcrafted tabular features might not fully capture the relevant structural nuances.

## Future Work

-   **Explore More Advanced GNN Architectures:** Investigate more sophisticated GNN layers (e.g., attention mechanisms, message passing variants) or deeper GNN models.
-   **Incorporate 3D Molecular Information:** Integrate 3D structural data (e.g., bond angles) into the GNN input.
-   **Expand Hyperparameter Search:** Conduct more extensive hyperparameter optimization, especially for deep learning models, with broader search spaces or longer training times.
-   **Ensemble Modeling:** Develop ensemble methods that combine predictions from diverse model types (XGBoost, MLP, GNN) for potentially enhanced robustness and accuracy.
-   **Model Interpretability:** Investigate techniques to enhance the interpretability of the GNN model, providing deeper chemical insights into which molecular substructures or interactions drive the `pGI50` predictions.

## License

This project is licensed under the [MIT License](LICENSE) - see the `LICENSE` file for details.

## Contact

Shirshak Aryal - shirshakaryal52@gmail.com
