# FedAvgIm

## Federated Learning for Multi-Site Impedance Identification of Inverter-Based Resources

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Federated%20Learning-ee4c2c.svg)](https://pytorch.org/)
[![MATLAB](https://img.shields.io/badge/MATLAB-Impedance%20Modeling-orange.svg)](https://www.mathworks.com/products/matlab.html)
[![Research](https://img.shields.io/badge/Status-Research%20Code-green.svg)](#project-status)

**FedAvgIm** is a research-oriented framework for privacy-preserving, multi-site impedance identification of inverter-based resources (IBRs).

The repository combines:

* Physics-based impedance modeling in MATLAB.
* Data-driven impedance identification using neural networks.
* Federated Averaging (FedAvg) across multiple IBR sites.
* Comparisons with centralized and local-only training.
* Transfer-learning experiments for adapting the federated model to previously unseen IBRs.
* Publication-quality visualization and performance evaluation.

The primary objective is to identify the frequency-domain impedance or admittance characteristics of multiple grid-connected converters without requiring each site to share its raw measurement data.

---

## Table of Contents

* [Motivation](#motivation)
* [Method Overview](#method-overview)
* [Repository Structure](#repository-structure)
* [Experimental Scenarios](#experimental-scenarios)
* [Requirements](#requirements)
* [Installation](#installation)
* [Data Preparation](#data-preparation)
* [Running the Experiments](#running-the-experiments)
* [Evaluation](#evaluation)
* [MATLAB Impedance Models](#matlab-impedance-models)
* [Expected Outputs](#expected-outputs)
* [Reproducibility Notes](#reproducibility-notes)
* [Citation](#citation)
* [License](#license)
* [Contact](#contact)

---

## Motivation

Impedance-based analysis is widely used to assess the small-signal stability and dynamic interactions of converter-dominated power systems.

Conventional data-driven impedance identification commonly requires measurement data from all converter sites to be transferred to a centralized server. In practical multi-owner or geographically distributed power systems, this may be undesirable because of:

* Data privacy requirements.
* Communication constraints.
* Proprietary converter information.
* Cybersecurity concerns.
* Differences in operating conditions and local datasets.

FedAvgIm addresses these issues using federated learning. Each IBR trains a local impedance-identification model using its own dataset. Only model parameters are communicated to the coordinating server, where they are aggregated to construct a global model.

---

## Method Overview

The general workflow is illustrated below:

```text
Local measurements at each IBR
              │
              ▼
    Local dataset construction
              │
              ▼
      Local neural-network model
              │
              ▼
      Local parameter updates
              │
              ├───────────────┐
              ▼               │
      Federated server        │
              │               │
   Weighted parameter         │
      aggregation             │
         using FedAvg         │
              │               │
              ▼               │
       Updated global model ──┘
              │
              ▼
  Multi-site impedance prediction
```

For (K) participating clients, the standard FedAvg update is

[
\mathbf{w}^{(r+1)}
==================

\sum_{k=1}^{K}
\frac{n_k}{\sum_{j=1}^{K} n_j}
\mathbf{w}^{(r+1)}_k,
]

where:

* (\mathbf{w}^{(r+1)}_k) is the locally trained model from client (k).
* (n_k) is the number of training samples available at client (k).
* (r) denotes the federated communication round.

The global model is repeatedly distributed, locally updated and aggregated until the selected convergence criterion or maximum number of communication rounds is reached.

---

## Key Features

* **Multi-site federated training:** Trains a common impedance-identification model across multiple IBRs.
* **Raw-data privacy:** Local measurement datasets remain at their respective sites.
* **Non-IID evaluation:** Supports clients with different converter parameters, operating points and data distributions.
* **Centralized baseline:** Compares FedAvg with conventional centralized training.
* **Local-only baseline:** Quantifies the benefit of collaboration relative to isolated site-specific models.
* **Transfer learning:** Adapts the global federated representation to an unseen or data-limited IBR.
* **Complex-valued impedance prediction:** Supports the real and imaginary components of frequency-dependent impedance or admittance.
* **Publication-ready figures:** Generates high-resolution PDF and SVG plots suitable for technical papers.

---

## Repository Structure

```text
FedAvgIm-Federated-Learning-for-Multi-Site-Impedance-Identification-of-Inverter-Based-Resources/
│
├── CCM_based_Impedance.m
├── CC_PLL_GFLI.m
├── CC_PLL_GFLI_Plot.m
│
├── CodeImplement/
│   ├── FL_SCENARIO_AE.ipynb
│   ├── FL_SCENARIO_B.ipynb
│   ├── Run_Scenario_D.ipynb
│   ├── FL_vs_Centralized.py
│   ├── FL_vs_LocalOnly.py
│   │
│   ├── fedavg_scenario_ae_results/
│   ├── fedavg_scenario_ae_compare/
│   │
│   └── Former_code/
│
├── .gitignore
└── README.md
```

### Main files

| File                                 | Description                                                                                                    |
| ------------------------------------ | -------------------------------------------------------------------------------------------------------------- |
| `CCM_based_Impedance.m`              | MATLAB state-space formulation of the grid-following inverter and its small-signal impedance/admittance model. |
| `CC_PLL_GFLI.m`                      | Grid-following inverter model including current control, computational delay, LCL filter and PLL dynamics.     |
| `CC_PLL_GFLI_Plot.m`                 | MATLAB impedance calculation and frequency-response visualization.                                             |
| `CodeImplement/FL_SCENARIO_AE.ipynb` | Main FedAvg experiment and associated evaluations.                                                             |
| `CodeImplement/FL_SCENARIO_B.ipynb`  | Alternative federated-learning scenario for evaluating client or data heterogeneity.                           |
| `CodeImplement/Run_Scenario_D.ipynb` | Transfer-learning or adaptation experiment for an unseen/data-limited IBR.                                     |
| `CodeImplement/FL_vs_Centralized.py` | Evaluation and visualization comparing FedAvg with centralized training.                                       |
| `CodeImplement/FL_vs_LocalOnly.py`   | Evaluation and visualization comparing FedAvg with independently trained local models.                         |
| `CodeImplement/Former_code/`         | Archived scripts, notebooks and earlier experimental outputs.                                                  |

Generated figures and trained model files may be stored in scenario-specific result directories.

---

## Experimental Scenarios

The repository contains several experiment groups.

### Scenario A/E — Multi-Site Federated Impedance Identification

This scenario trains a global model using datasets distributed across multiple IBR clients.

The typical process is:

1. Load the local datasets for all participating IBRs.
2. Divide each local dataset into training and testing subsets.
3. Initialize a global neural-network model.
4. Distribute the global model to participating clients.
5. Train each local model for a specified number of local epochs.
6. Aggregate the local parameters using FedAvg.
7. Evaluate the global model on the local test datasets.
8. Compare the final model with centralized and local-only baselines.

The resulting global model is typically saved as:

```text
scenario_ae_fedavg_global_model.pt
```

### Scenario B — Heterogeneous Client Evaluation

Scenario B is intended to investigate federated-learning performance when clients have heterogeneous local conditions.

Possible sources of heterogeneity include:

* Different converter controller parameters.
* Different operating points.
* Different quantities of local data.
* Different local frequency samples.
* Different noise levels.
* Non-identically distributed datasets.

Use the configuration cells inside `FL_SCENARIO_B.ipynb` to define the clients and experimental settings.

### Scenario D — Transfer to an Unseen IBR

Scenario D evaluates whether a model trained collaboratively across existing IBRs can be transferred to a new IBR with limited local data.

A typical transfer-learning workflow is:

1. Pretrain the representation using federated learning.
2. Initialize the new IBR model using the federated parameters.
3. Freeze or partially freeze the shared feature-extraction layers.
4. Fine-tune the selected layers using a small local dataset.
5. Compare zero-shot prediction, fine-tuned prediction and local training from scratch.

This scenario is useful for evaluating:

* Data efficiency.
* Generalization to unseen converters.
* Fine-tuning convergence.
* Transfer-learning accuracy.
* Reduction in local data requirements.

---

## Requirements

### Python environment

Recommended:

* Python 3.10 or later
* JupyterLab or Jupyter Notebook
* PyTorch
* NumPy
* Pandas
* SciPy
* scikit-learn
* Matplotlib

Install the primary dependencies with:

```bash
pip install torch numpy pandas scipy scikit-learn matplotlib jupyter
```

A dedicated virtual environment is recommended.

### MATLAB environment

Recommended MATLAB products:

* MATLAB
* Control System Toolbox
* Simulink, when simulation models are used
* Signal Processing Toolbox, when additional frequency-domain preprocessing is required

The exact toolbox requirements may depend on the MATLAB release and the selected scripts.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/ManhqhUMich12/FedAvgIm-Federated-Learning-for-Multi-Site-Impedance-Identification-of-Inverter-Based-Resources.git
```

Enter the repository:

```bash
cd FedAvgIm-Federated-Learning-for-Multi-Site-Impedance-Identification-of-Inverter-Based-Resources
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install --upgrade pip
pip install torch numpy pandas scipy scikit-learn matplotlib jupyter
```

Start Jupyter:

```bash
jupyter lab
```

Then open the notebooks under:

```text
CodeImplement/
```

---

## Data Preparation

Large datasets are intentionally excluded from version control. The repository `.gitignore` excludes common generated-data formats such as:

```text
*.mat
*.csv
*.xlsx
```

Therefore, users must generate or provide the required local impedance datasets before running the federated-learning notebooks.

### Recommended data organization

A suggested structure is:

```text
CodeImplement/
└── data/
    ├── IBR1/
    │   ├── train.*
    │   └── test.*
    ├── IBR2/
    │   ├── train.*
    │   └── test.*
    ├── IBR3/
    │   ├── train.*
    │   └── test.*
    └── ...
```

Each client dataset should contain the input features and target impedance/admittance values required by the selected notebook.

A typical supervised sample can be expressed as

[
\mathbf{x}
==========

\left[
f,,
\boldsymbol{\theta}*{\mathrm{op}},,
\boldsymbol{\theta}*{\mathrm{ctrl}}
\right],
]

[
\mathbf{y}
==========

\left[
\operatorname{Re}{Y_{dd}},
\operatorname{Im}{Y_{dd}},
\operatorname{Re}{Y_{dq}},
\operatorname{Im}{Y_{dq}},
\operatorname{Re}{Y_{qd}},
\operatorname{Im}{Y_{qd}},
\operatorname{Re}{Y_{qq}},
\operatorname{Im}{Y_{qq}}
\right],
]

depending on the model configuration.

The exact feature and target definitions should remain consistent across:

* Training datasets.
* Testing datasets.
* Centralized baseline.
* Local-only baseline.
* Federated-learning clients.
* Transfer-learning experiments.

### Data normalization

Normalization statistics must be calculated from the training data only.

For an input feature (x_j),

[
\hat{x}_j
=========

\frac{x_j-\mu_j}{\sigma_j},
]

where (\mu_j) and (\sigma_j) are the training-set mean and standard deviation.

The same normalization parameters must be reused for validation, testing and deployment.

---

## Running the Experiments

### 1. Generate or validate the analytical impedance model

Open MATLAB and run one of the impedance-model scripts:

```matlab
run('CCM_based_Impedance.m')
```

or:

```matlab
run('CC_PLL_GFLI.m')
```

To calculate and plot the corresponding frequency responses:

```matlab
run('CC_PLL_GFLI_Plot.m')
```

Before running the scripts, verify:

* DC-link voltage.
* Grid voltage and frequency.
* LCL-filter parameters.
* Current-controller gains.
* PLL gains.
* Digital delay.
* Converter operating point.
* Frequency range used for the Bode response.

### 2. Run the main FedAvg experiment

Open:

```text
CodeImplement/FL_SCENARIO_AE.ipynb
```

Execute the notebook sequentially from the first cell.

The notebook should perform:

* Data loading.
* Train/test preparation.
* Model definition.
* Client initialization.
* Local training.
* FedAvg aggregation.
* Global-model evaluation.
* Result visualization.
* Model checkpoint export.

### 3. Compare FedAvg with centralized training

The comparison script expects the main experiment to have already created the relevant models, datasets and learning-history variables.

Run it from the notebook environment using:

```python
%run FL_vs_Centralized.py
```

The script evaluates quantities such as:

* Training MSE.
* Testing MSE.
* Convergence speed.
* Final prediction error.
* Error distributions.
* Per-client accuracy.

### 4. Compare FedAvg with local-only training

After running the required model and dataset cells, execute:

```python
%run FL_vs_LocalOnly.py
```

This comparison measures whether clients benefit from federated collaboration relative to training solely on their local datasets.

### 5. Run the heterogeneous-client scenario

Open and execute:

```text
CodeImplement/FL_SCENARIO_B.ipynb
```

Review the client configuration and local dataset paths before starting the experiment.

### 6. Run the transfer-learning scenario

Open:

```text
CodeImplement/Run_Scenario_D.ipynb
```

Configure:

* Source federated checkpoint.
* Target IBR dataset.
* Fine-tuning data percentage.
* Frozen and trainable layers.
* Number of fine-tuning epochs.
* Learning rate.
* Random seed.

Run the notebook sequentially to compare the federated initialization with alternative training strategies.

---

## Configuration

The main hyperparameters are generally defined near the beginning of each notebook.

Important settings include:

| Parameter         | Meaning                                              |
| ----------------- | ---------------------------------------------------- |
| `NUM_ROUNDS`      | Number of federated communication rounds.            |
| `LOCAL_EPOCHS`    | Number of local training epochs per round.           |
| `BATCH_SIZE`      | Local mini-batch size.                               |
| `LEARNING_RATE`   | Optimizer learning rate.                             |
| `CLIENT_FRACTION` | Fraction of clients participating in each round.     |
| `HIDDEN_GFLI`     | Hidden-layer configuration of the neural network.    |
| `SEED`            | Random seed used for reproducibility.                |
| `device`          | CPU or CUDA execution device.                        |
| `input_dim`       | Number of input features.                            |
| `output_dim`      | Number of predicted impedance/admittance components. |

For a fair comparison, the centralized, local-only and federated models should use compatible:

* Architectures.
* Optimizers.
* Loss functions.
* Data splits.
* Normalization parameters.
* Evaluation metrics.
* Random seeds.

---

## Evaluation

The primary regression metric is mean squared error:

[
\mathrm{MSE}
============

\frac{1}{N}
\sum_{i=1}^{N}
\left|
\hat{\mathbf{y}}_i-\mathbf{y}_i
\right|_2^2.
]

Additional metrics may include:

### Mean absolute error

[
\mathrm{MAE}
============

\frac{1}{N}
\sum_{i=1}^{N}
\left|
\hat{\mathbf{y}}_i-\mathbf{y}_i
\right|_1.
]

### Absolute percentage error

[
\mathrm{APE}_i
==============

\frac{
\left|\hat{y}_i-y_i\right|
}{
\max\left(\left|y_i\right|,\epsilon\right)
}
\times 100%.
]

### Overfitting gap

[
\Delta_{\mathrm{overfit}}
=========================

## \mathrm{MSE}_{\mathrm{test}}

\mathrm{MSE}_{\mathrm{train}}.
]

### Client fairness

Client-level performance should be evaluated separately to determine whether the global model benefits all participating IBRs rather than only improving average accuracy.

Useful fairness indicators include:

* Mean client test error.
* Worst-client test error.
* Standard deviation of client errors.
* Improvement relative to local-only training.
* Fraction of clients benefiting from federated learning.

---

## MATLAB Impedance Models

The MATLAB files construct a small-signal model of a grid-following inverter containing components such as:

* Inner current-control loop.
* Computational and modulation delay.
* LCL filter.
* Phase-locked loop.
* Operating-point-dependent coupling.
* (dq)-domain impedance/admittance formulation.

The resulting frequency-domain model generally has the form

[
\begin{bmatrix}
\Delta i_d \
\Delta i_q
\end{bmatrix}
=============

*

\begin{bmatrix}
Y_{dd}(s) & Y_{dq}(s) \
Y_{qd}(s) & Y_{qq}(s)
\end{bmatrix}
\begin{bmatrix}
\Delta v_d \
\Delta v_q
\end{bmatrix}.
]

The scripts calculate and visualize:

* (Y_{dd})
* (Y_{dq})
* (Y_{qd})
* (Y_{qq})

over a selected frequency range.

These analytical or simulation-derived responses can be used to construct the datasets for federated impedance identification.

---

## Expected Outputs

Depending on the selected scenario, the scripts may produce:

### Model checkpoints

```text
*.pt
```

### Publication-quality figures

```text
*.pdf
*.svg
```

### Typical plots

* Federated training and testing curves.
* FedAvg versus centralized convergence.
* FedAvg versus local-only performance.
* Per-client MSE comparison.
* Error cumulative distribution functions.
* Client fairness plots.
* Real-part impedance/admittance responses.
* Imaginary-part impedance/admittance responses.
* Zero-shot versus fine-tuned performance.
* Transfer-learning data-efficiency curves.

Generated files should preferably be placed in a dedicated result directory rather than in the source-code directories.

---

## Recommended Reproduction Order

For a complete reproduction, use the following order:

```text
1. Validate the MATLAB converter impedance model
2. Generate the local datasets
3. Configure data paths and normalization
4. Run FL_SCENARIO_AE.ipynb
5. Run FL_vs_Centralized.py
6. Run FL_vs_LocalOnly.py
7. Run FL_SCENARIO_B.ipynb
8. Run Run_Scenario_D.ipynb
9. Export and inspect all figures and metrics
```

---

## Reproducibility Notes

For reproducible results:

1. Set the same random seed for Python, NumPy and PyTorch.
2. Save the exact train/test split indices.
3. Record the PyTorch, CUDA and MATLAB versions.
4. Use identical normalization statistics across comparison methods.
5. Store experiment parameters in a configuration file.
6. Save the global model after every important communication round.
7. Report results over multiple independent random seeds.
8. Clearly distinguish client-weighted and unweighted average metrics.
9. Avoid using test data for normalization or model selection.
10. Document any excluded clients or failed training runs.

Example seed configuration:

```python
import random
import numpy as np
import torch

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
```

For stricter deterministic execution:

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

Deterministic execution may reduce training speed and may not be available for every PyTorch operation.

---

## Known Limitations

The current repository is research code and may require manual configuration before execution.

Current limitations may include:

* Dataset files are not included in the repository.
* Some analysis scripts depend on variables created by previously executed notebook cells.
* Experiment paths may need to be updated for the local environment.
* Archived figures and scripts are retained under `Former_code/`.
* Hardware, software-version and random-seed differences may affect exact numerical reproduction.
* The current implementation is intended primarily for offline experimental studies rather than production deployment.

---

## Suggested Future Improvements

Potential extensions include:

* Moving experiment settings into YAML configuration files.
* Adding a standalone command-line training interface.
* Adding a `requirements.txt` or `environment.yml`.
* Providing a small public example dataset.
* Adding automated unit tests.
* Adding continuous integration.
* Supporting partial client participation.
* Investigating communication-efficient aggregation.
* Incorporating differential privacy or secure aggregation.
* Evaluating robustness against noisy or malicious clients.
* Extending the framework to additional grid-forming and grid-following converter types.

---

## Citation

A formal citation will be added when the associated publication is publicly available.

For the current repository, use:

```bibtex
@misc{fedavgim,
  author       = {Manh Hoang et al},
  title        = {FedAvgIm: Federated Learning for Multi-Site Impedance
                  Identification of Inverter-Based Resources},
  year         = {2026},
  howpublished = {\url{https://github.com/ManhqhUMich12/FedAvgIm-Federated-Learning-for-Multi-Site-Impedance-Identification-of-Inverter-Based-Resources}},
  note         = {Research software}
}
```

Please replace the provisional author list and citation metadata with the final publication information when available.

---

## License

No open-source license is currently specified.

Before redistributing or reusing the code, please contact the repository owner for permission. A standard license such as MIT, BSD-3-Clause or Apache-2.0 should be added if public reuse is intended.

---

## Contact

**Manh Hoang**

GitHub: [@ManhqhUMich12](https://github.com/ManhqhUMich12)

For questions, bug reports or research collaboration, please open a GitHub issue in this repository.

---

## Acknowledgments

This repository was developed for research on federated learning, impedance identification and stability analysis of inverter-based power systems.

Contributions, technical discussions and reproducibility feedback are welcome.
