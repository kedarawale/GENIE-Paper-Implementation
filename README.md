# GENIE-Paper-Implementation

This repository contains a Python implementation related to the GENIE paper, focusing on Graph Neural Network (GNN) watermarking techniques and evaluating their robustness against various attacks.

## Code Structure

Make sure to keep the dataset files in the correct folder before running the scripts.

```
GENIE-Paper-Implementation/
│
├── astroph/                      # Astro Physics Collaboration dataset (CA-AstroPh)
│   ├── CA-AstroPh.txt            # Dataset file
│   ├── gcn.py                    # GCN model implementation for AstroPh
│   ├── graphsage.py              # GraphSAGE model implementation for AstroPh
│   ├── neognn.py                 # NeoGNN model implementation for AstroPh
│   ├── seal.py                   # SEAL model implementation for AstroPh
│   └── experiments/              # Robustness experiments for AstroPh
│       ├── extraction.py
│       ├── finetuning.py
│       ├── knowledgedistillation.py
│       ├── pruning.py
│       ├── quantization.py
│       ├── rtal.py
│       └── timing.py
│
├── biogrid/                      # BioGRID Protein-Protein Interaction dataset
│   ├── BIOGRID.txt               # Dataset file (Assumed name, check file)
│   ├── gcn.py                    # GCN model implementation for BioGRID
│   ├── biogrid_graphsage.py      # GraphSAGE model implementation for BioGRID
│   └── experiments/              # Robustness experiments for BioGRID
│       ├── extraction.py
│       ├── finetuning.py
│       ├── knowledgedistillation.py
│       ├── pruning.py
│       ├── quantization.py
│       ├── rtal.py
│       └── timing.py
│
├── inf-power/                   # US Power Grid dataset
│   ├── inf-power.mtx           # Dataset file
│   ├── gcn.py                    # GCN model implementation for Power Grid
│   ├── neognn.py                 # NeoGNN model implementation for Power Grid
│   ├── seal.py                   # SEAL model implementation for Power Grid
│   └── experiments/              # Robustness experiments for Power Grid
│       └── ...
│
├── usair97/                      # USAir97 Flight Network dataset
│   ├── USAir97.mtx               # Dataset file
│   ├── USAir97_coord.mtx         # Coordinate file (Used by some models)
│   ├── USAir97_nodename.txt      # Node name file (Used by some models)
│   ├── usair97_gcn.py            # GCN model implementation for USAir97
│   ├── usair97_graphsage.py      # GraphSAGE model implementation for USAir97
│   ├── usair97_neognn.py         # NeoGNN model implementation for USAir97
│   ├── usair97_seal.py           # SEAL model implementation for USAir97
│   └── experiments/              # Robustness experiments for USAir97
│       └── ...
│
└── yeast/                        # Yeast Protein-Protein Interaction dataset
    ├── yeast.edges               # Dataset file
    ├── gcn.py                    # GCN model implementation for Yeast
    ├── neognn.py                 # NeoGNN model implementation for Yeast
    ├── seal.py                   # SEAL model implementation for Yeast
    └── experiments/              # Robustness experiments for Yeast
        └── ...
```

- **Dataset Directories (e.g., `yeast/`, `usair97/`)**: Each contains the raw data file(s) and Python scripts for specific GNN models (GCN, GraphSAGE, NeoGNN, SEAL) adapted for that dataset.  
- **Model Scripts (e.g., `yeast/gcn.py`)**: These scripts typically handle data loading, Node2Vec embedding generation (if not pre-computed), model definition, training a watermarked model, and basic evaluation. Running these directly often trains the base watermarked model for the respective dataset and GNN architecture.  
- **Experiment Subdirectories (e.g., `yeast/experiments/`)**: Contain scripts to run specific attack evaluations (Extraction, Fine-tuning, Pruning, etc.) against the pre-trained watermarked models.  

## Running the Code

### General Prerequisites

Ensure you have Python and necessary libraries installed. Based on the imports observed, you'll likely need:

- PyTorch (`torch`)  
- PyTorch Geometric (`torch_geometric`)  
- NetworkX (`networkx`)  
- NumPy (`numpy`)  
- SciPy (`scipy`)  
- Scikit-learn (`sklearn`)  
- Node2Vec (`node2vec` - likely a specific local or fork version is used, check requirements if provided elsewhere)  
- TQDM (`tqdm`)  

### Running a Base Model Training

To train a specific watermarked GNN model on a dataset, navigate to the dataset's directory and run the corresponding model script. The scripts typically save intermediate files like embeddings (`*.npy`) and data splits (`*.pkl`) to avoid re-computation.

**Example: Train GCN on Yeast Dataset**

```bash
cd GENIE-Paper-Implementation/yeast
python gcn.py
```

(Repeat similarly for other models like `graphsage.py`, `neognn.py`, `seal.py` within their respective dataset directories.)

### Running Experiments

To run a specific robustness experiment (attack evaluation), navigate to the `experiments` subdirectory within the desired dataset's folder and run the corresponding experiment script. These scripts usually load a pre-trained watermarked model (trained by the base model script) and evaluate its watermark under the specific attack.

**Example: Run Extraction Experiment on Yeast GCN Model**

```bash
cd GENIE-Paper-Implementation/yeast/experiments
python extraction.py
```

**Example: Run Fine-tuning Experiment on USAir97 GCN Model**

```bash
cd GENIE-Paper-Implementation/usair97/experiments
python finetuning.py
```

(Follow this pattern for other experiments like `knowledgedistillation.py`, `pruning.py`, etc., within the relevant experiments directory.)
