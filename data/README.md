# Sparkov Credit Card Fraud Detection Dataset

## Overview

This project uses the **Sparkov Credit Card Fraud Detection** dataset, a large-scale
simulated credit card transaction dataset generated using the Sparkov Data Generation
tool by Brandon Harris.

Key statistics:
- **1.8 million** simulated transactions
- **1,000** unique customers
- **800** merchants
- Covers transactions from **January 2019 to December 2020**
- Fraud rate is approximately **0.58%** (realistic class imbalance)

Dataset source: <https://www.kaggle.com/datasets/kartik2112/fraud-detection>

## Download Instructions

### Option A: Kaggle CLI (recommended)

1. Install the Kaggle CLI if you have not already:

   ```bash
   pip install kaggle
   ```

2. Configure your Kaggle API credentials. Place your `kaggle.json` file at
   `~/.kaggle/kaggle.json` (Linux/macOS) or `C:\Users\<user>\.kaggle\kaggle.json`
   (Windows). You can generate this file from your Kaggle account settings page.

3. Download and extract the dataset into this directory:

   ```bash
   kaggle datasets download -d kartik2112/fraud-detection -p data/ --unzip
   ```

### Option B: Manual Download

1. Visit <https://www.kaggle.com/datasets/kartik2112/fraud-detection>.
2. Click the **Download** button (requires a free Kaggle account).
3. Extract the ZIP archive into this `data/` directory.

## Expected File Structure

After downloading, this directory should contain:

```
data/
  README.md          <- this file
  fraudTrain.csv     <- training set (~1.3M transactions, 2019-01 to 2020-06)
  fraudTest.csv      <- test set (~0.5M transactions, 2020-07 to 2020-12)
```

## Note on Version Control

The `data/` directory is included in `.gitignore`. The CSV files are large
(fraudTrain.csv is roughly 490 MB, fraudTest.csv roughly 180 MB) and must not
be committed to the repository. Each collaborator should download the data
independently using the instructions above.
