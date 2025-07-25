# Applying Filters on Census Income Dataset

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
  <img src="https://img.shields.io/badge/Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white"/>
</p>

## Overview

This notebook demonstrates how to:

- Load and inspect the **Census Income dataset**
- Perform **random sampling** using both NumPy and pandas
- **Filter data** based on specific column values
- Combine multiple conditions for filtering
- Perform basic **data analysis** such as computing mean values for filtered subsets

The dataset contains census information from **1994** with **7000 rows and 15 columns**.

---

## Key Steps

1. **Inspecting the Data**
   - Load the CSV file into a pandas DataFrame
   - Explore the first few rows using `head()`
   - Get dataset shape (`df.shape`)

2. **Random Sampling**
   - Randomly select 30% of rows using:
     - NumPy (`np.random.choice` with `loc`)
     - Pandas built-in `sample()` method

3. **Filtering Data**
   - Filter rows where `workclass == 'Private'`
   - Count the number of rows that meet the condition

4. **Combining Conditions**
   - Filter rows where:
     - `workclass == 'Local-gov'`
     - AND `hours-per-week > 40`
   - Combine conditions using `&` operator

6. **Basic Data Analysis**
   - Compute mean age for specific groups (e.g., self-reported females)
   - Find number of rows matching multiple criteria

---

## Installation

Clone the repository and install required dependencies:

```bash
git clone https://github.com/aditi-dheer/btt-mit-ai.git
cd Manage\ Data\ in\ ML
pip install pandas numpy jupyter
```

---  
## Usage

Run the project:

```bash
jupyter notebook applying_filters.ipynb
```

It will output the **ROC-AUC score** on the test set.

---

## Project Structure

- **data/censusData** – Census Income dataset (1994)
- **applying_filters.ipynb** – Main script
- **README.md** – Project documentation  
