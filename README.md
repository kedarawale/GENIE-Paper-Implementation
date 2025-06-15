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
- **Experiment Subdirectories (e.g., `yeast/experiments/`)**: Contain scripts to run specific attack evaluations (Extraction, Fine-tuning, Pruning, etc.) against the 
pre-trained watermarked models.  

#### Experiments Scripts Location

All the experiment scripts are located inside the `experiments/` subdirectory for each dataset:

```
GENIE-Paper-Implementation/
├── yeast/experiments/
├── usair97/experiments/
├── biogrid/experiments/
├── astroph/experiments/
└── inf-power/experiments/

```

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
**Note : Run all the experiments for 10 times and get the average of them to get the expected results in the paper**

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

### Experiment - Model Extraction Attack (Table 6 & Table 7)


#### How to Run the Scripts

```bash
# Yeast Dataset
cd yeast/experiments
python extraction.py

# USAir97 Dataset
cd ../../usair97/experiments
python extraction.py

# BioGRID Dataset
cd ../../biogrid/experiments
python extraction.py

# AstroPh Dataset
cd ../../astroph/experiments
python extraction.py

# Power Grid Dataset
cd ../../inf-power/experiments
python extraction.py
```

#### Output Format

```bash
--- Model Extraction ---
No attack        → Dtest=97.44%, Dwm=96.54%
Extract   Soft   → Dtest=88.87%, Dwm=18.41%
Extract   Hard   → Dtest=88.96%, Dwm=16.13%
Extract Double   → Dtest=87.54%, Dwm=56.22%

```


### Experiment - Timing Evaluation (Watermark Overhead)

#### How to Run the Scripts

```bash
# Yeast Dataset
cd yeast/experiments
python timing.py

# USAir97 Dataset
cd usair97/experiments
python timing.py

# BioGRID Dataset
cd biogrid/experiments
python timing.py

# AstroPh Dataset
cd astroph/experiments
python timing.py

# Power Grid Dataset
cd inf-power/experiments
python timing.py

```

#### Output Format

```bash
--- Training Time Comparison ---
Standard Training Time : 1.6s
Watermark Training Time: 3.2s
Overhead Factor        : 1.99x
```

### Experiment - Model Fine-Tuning Attack (Table 9)

#### How to Run the Scripts

```bash
# Yeast Dataset
cd yeast/experiments
python finetuning.py
python rtal.py

# USAir97 Dataset
cd usair97/experiments
python finetuning.py
python rtal.py

# BioGRID Dataset
cd biogrid/experiments
python finetuning.py
python rtal.py

# AstroPh Dataset
cd astroph/experiments
python finetuning.py
python rtal.py

# Power Grid Dataset
cd inf-power/experiments
python finetuning.py
python rtal.py
```
#### Output Format

```bash
--- Model Fine-Tuning ---
No tuning   → Dtest=96.21%, Dwm=100.00%
Tune  FTLL → Dtest=97.15%, Dwm=100.00%
Tune  RTLL → Dtest=97.45%, Dwm=55.07%
Tune  FTAL → Dtest=95.77%, Dwm=84.45%
Tune  RTAL → Dtest=95.27%, Dwm=37.29%
```

### Experiment - Knowledge Distillation Attack (Table 8)

#### How to Run the Scripts

```bash
# Yeast Dataset
cd yeast/experiments
python knowledgedistillation.py

# USAir97 Dataset
cd usair97/experiments
python knowledgedistillation.py

# BioGRID Dataset
cd biogrid/experiments
python knowledgedistillation.py

# AstroPh Dataset
cd astroph/experiments
python knowledgedistillation.py

# Power Grid Dataset
cd inf-power/experiments
python knowledgedistillation.py
```
#### Output Format 

```bash
--- Knowledge Distillation ---
Distilled   → Dtest=95.90%, Dwm= 8.63%
Knowledge Distillation: Test AUC=0.9513, WM AUC=0.0407
```


### Experiment - Model Pruning Attack (Table 10)

#### How to Run the Scripts

```bash
# Yeast Dataset
cd yeast/experiments
python pruning.py

# USAir97 Dataset
cd usair97/experiments
python pruning.py

# BioGRID Dataset
cd biogrid/experiments
python pruning.py

# AstroPh Dataset
cd astroph/experiments
python pruning.py

# Power Grid Dataset
cd inf-power/experiments
python pruning.py
```

#### Output Format

```bash
--- Model Pruning ---
Prune   0%  → Dtest=96.99%, Dwm=99.95%
Prune  20% → Dtest=97.00%, Dwm=99.95%
Prune  40% → Dtest=96.90%, Dwm=99.92%
Prune  60% → Dtest=96.61%, Dwm=99.87%
Prune  80% → Dtest=95.94%, Dwm=99.63%
```

### Experiment - Weight Quantization Attack

#### How to Run the Scripts

```bash
# Yeast Dataset
cd yeast/experiments
python quantization.py

# USAir97 Dataset
cd ../../usair97/experiments
python quantization.py

# BioGRID Dataset
cd ../../biogrid/experiments
python quantization.py

# AstroPh Dataset
cd ../../astroph/experiments
python quantization.py

# Power Grid Dataset
cd ../../inf-power/experiments
python quantization.py
```

#### Output Format

```bash
--- Weight Quantization --- 
Quantized   → Dtest=96.07%, Dwm=99.97%
```


### Experiment - Fine-Pruning with RTAL (Table 9)

#### How to Run the Scripts

```bash
# Yeast Dataset
cd yeast/experiments
python rtal.py

# USAir97 Dataset
cd ../../usair97/experiments
python rtal.py

# BioGRID Dataset
cd ../../biogrid/experiments
python rtal.py

# AstroPh Dataset
cd ../../astroph/experiments
python rtal.py

# Power Grid Dataset
cd ../../inf-power/experiments
python rtal.py
```

#### Output Format

```bash
--- Fine-Pruning (RTAL) ---
P+RTAL  20% → Dtest=96.01%, Dwm=12.75%
P+RTAL  40% → Dtest=94.03%, Dwm=29.63%
P+RTAL  60% → Dtest=95.71%, Dwm=26.18%
P+RTAL  80% → Dtest=94.27%, Dwm=16.21%
```

### Architecture Code Execution : Watermarked Model Training & End-to-End Evaluation (Tables 4–10)

For eg : The script "usair97_gcn.py" trains a watermarked GCN model on a given dataset and runs **end-to-end evaluation**, including:

- Watermark embedding
- Watermark verification (ownership check)
- Robustness attacks (extraction, distillation, fine-tuning, pruning, quantization)
- Timing overhead measurement

#### How to Run (USAir97 example)

```bash
cd usair97
python usair97_gcn.py
```

#### Output Format
```bash
Final Train AUC: 0.9981, Val AUC: 0.9493, Test AUC: 0.9700
Watermark Verification AUC on WM Data: 0.9957
Judge: Registered model for USAir_GCNModelOwner at timestamp=2025-02-15 19:58:45

Fine-tuning attack ...
Judge: Found record for USAir_GCNModelOwner, verifying watermark ...

Watermark Verification AUC = 0.9960
Dynamic Threshold = 0.7040
Ownership Verified : True
Smoothed bootstrap p-value = 0.3198
Judge: Verified ownership for USAir_GCNModelOwner, p-value=0.3198
```
