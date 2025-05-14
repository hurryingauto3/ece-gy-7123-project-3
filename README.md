
# Deep Learning Project 3: Jailbreaking Deep Models

This repository contains the code for Deep Learning Project 3, focusing on adversarial attacks against pre-trained image classification models. The goal is to generate subtle perturbations to images that cause state-of-the-art models to misclassify them, degrading their performance.

We implement and evaluate pixel-wise ($L_\infty$) attacks (FGSM, PGD) and a patch-based ($L_0$) attack against a pre-trained ResNet-34 model on a subset of ImageNet-1K. We also analyze the transferability of these attacks to a pre-trained DenseNet-121 model.

## Project Tasks

The project addresses the following tasks:

1.  **Basics:** Load dataset and pre-trained ResNet-34, evaluate baseline accuracy.
2.  **Pixel-wise Attacks (FGSM):** Implement and evaluate the Fast Gradient Sign Method ($L_\infty$) attack.
3.  **Improved Attacks (PGD):** Implement and evaluate the Projected Gradient Descent ($L_\infty$) attack.
4.  **Patch Attacks:** Implement and evaluate a PGD-based patch ($L_0$) attack, including an ablation study to find effective configurations.
5.  **Transferring Attacks:** Evaluate all generated adversarial datasets on a different model (DenseNet-121).

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [your_github_repo_link_here]
    cd [your_repo_name]
    ```

2.  **Prerequisites:**
    *   Python 3.7+
    *   pip or conda

3.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install torch torchvision numpy pandas tqdm matplotlib Pillow
    # If using CUDA, make sure to install the correct PyTorch version
    # e.g., pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```

4.  **Dataset:**
    Obtain the `TestDataSet` directory and the `labels_list.json` file. These were provided as part of the project. Place them in a location accessible to the code. Update the `DATASET_PATH` variable in the configuration section of the main notebook/script to point to your dataset location.
    The code will automatically generate the `imagenet_class_index.json` file required for mapping class indices.

5.  **Device:**
    The code is set up to use a GPU if available (`cuda`) and falls back to CPU. A GPU is highly recommended for faster attack generation and evaluation.

## How to Run the Code

The main project execution is typically done through a Jupyter Notebook or a Python script.

1.  **Using Jupyter Notebook (Recommended):**
    *   Start Jupyter Notebook or JupyterLab:
        ```bash
        jupyter notebook
        ```
    *   Open the main notebook file (e.g., `ece-7123-deeplearning-project-3.ipynb`).
    *   Run all cells sequentially. The notebook will perform data loading, baseline evaluation, attack generation and evaluation for each task, the patch attack ablation study, and transferability analysis.

The code will print status updates and final accuracy results to the console. Adversarial datasets for each task will be saved locally in directories like `./AdversarialTestSet1_FGSM`, `./AdversarialTestSet2_ImprovedLinf`, `./AdversarialTestSet3_Patch`, and directories for ablation variants. Note that these are saved as PyTorch `.pt` files to preserve floating-point perturbations.

## Project Structure

```
.
├── ece-7123-deeplearning-project-3.ipynb  # Main script/notebook for execution
├── README.md                   # This file
└── TestDataSet               # Directory containing the dataset (needs to be acquired)
    [labels_list.json
    
```

## Deliverables

*   **Project Report (PDF)
*   **Project Codebase (GitHub Repository)

## Team Members

*   Ali Hamza
*   Saad Zubairi

## Citations

Relevant research papers and resources used are cited within the code comments and the project report.
