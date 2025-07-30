# Census Income Data Exploration & Preprocessing

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
  <img src="https://img.shields.io/badge/Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white"/>
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white"/>
</p>

## Overview

This repository provides a modular walkthrough of data filtering, balancing, and type inspection techniques on the UCI Census Income dataset. Each notebook focuses on a distinct preprocessing task that is foundational for real-world data science and machine learning workflows.

## Contents

### 1. applying_filters.ipynb

Learn how to filter, combine conditions, and sample data using pandas and NumPy.

Topics covered:
- Filtering rows using Boolean conditions
- Combining filters with logical operators (&, |)
- Random sampling with NumPy and pandas
- Analyzing subsets using basic statistics

### 2. building_a_balanced_dataset.ipynb

Address class imbalance in the `sex_selfID` and `income` columns by selectively upsampling underrepresented groups.

Techniques used:
- Analyzing class distributions with value_counts and groupby
- Calculating imbalance ratios
- Filtering and sampling from underrepresented classes
- Merging new samples with the original dataset

### 3. obtaining_data_types.ipynb

Explore data types in the Census dataset using multiple pandas utilities.

Key methods:
- `df.dtypes` for direct inspection
- `df.describe(include='all')` to summarize all columns
- `pd.api.types.infer_dtype()` for precise type inference

## Getting Started

Clone the repository and install required dependencies:

```bash
git clone https://github.com/aditi-dheer/btt-mit-ai.git
cd Data\ Preprocessing/Manage\ Data\ in\ ML/
pip install pandas numpy jupyter
```

---  
## Usage
Each notebook is self-contained. Navigate to the module directory and launch Jupyter:

```bash
jupyter notebook applying_filters.ipynb
```
```bash
jupyter notebook building_a_balanced_dataset.ipynb
```

```bash
jupyter notebook obtaining_data_types.ipynb
```
