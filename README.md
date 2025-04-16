# Multi-Label Classification Model Performance (30 epochs)

## Trained models with GUI that predicts Lung X-Rays    
![image](https://github.com/user-attachments/assets/6612ce4e-0a6d-4c26-a868-01c9d42ba02f)
![image](https://github.com/user-attachments/assets/ad130c86-eda0-4c3c-b26d-7c212ac64a47)
      

## Overall Models Performance & Comparison

| Metric | EfficientNet | ResNet50 | MobileNet_v2 |
| --- | --- | --- | --- |
| Accuracy | 0.4752 | 0.5263 | 0.4441 |
| Jaccard Similarity | 0.5347 | 0.5671 | 0.5169 |
| Precision | 0.5329 | 0.5773 | 0.4959 |
| Recall | 0.5211 | 0.5242 | 0.5238 |
| F1 Score | 0.5269 | 0.5495 | 0.5095 |
| Hamming Loss | 0.0924 | 0.0849 | 0.0996 |

*Note: Accuracy values may be misleading.*

## EfficientNet Per-label Metrics

| Label | Precision | Recall | F1-score | Support |
| --- | --- | --- | --- | --- |
| 0 | 0.35 | 0.14 | 0.20 | 814 |
| 1 | 0.16 | 0.13 | 0.14 | 369 |
| 2 | 0.23 | 0.25 | 0.24 | 1266 |
| 3 | 0.35 | 0.22 | 0.27 | 342 |
| 4 | 0.36 | 0.19 | 0.25 | 139 |
| 5 | 0.63 | 0.11 | 0.19 | 166 |
| 6 | 0.42 | 0.39 | 0.40 | 914 |
| 7 | 0.17 | 0.10 | 0.12 | 228 |
| 8 | 0.33 | 0.22 | 0.26 | 251 |
| 9 | 0.48 | 0.13 | 0.21 | 389 |
| 10 | 0.33 | 0.11 | 0.17 | 501 |
| 11 | 0.66 | 0.84 | 0.74 | 5184 |
| micro avg | 0.53 | 0.52 | 0.53 | 10563 |
| macro avg | 0.37 | 0.24 | 0.27 | 10563 |
| weighted avg | 0.49 | 0.52 | 0.49 | 10563 |
| samples avg | 0.55 | 0.58 | 0.56 | 10563 |

## ResNet50 Per-label Metrics

| Label | Precision | Recall | F1-score | Support |
| --- | --- | --- | --- | --- |
| 0 | 0.26 | 0.07 | 0.11 | 814 |
| 1 | 0.18 | 0.02 | 0.04 | 369 |
| 2 | 0.30 | 0.12 | 0.17 | 1266 |
| 3 | 0.47 | 0.08 | 0.14 | 342 |
| 4 | 0.62 | 0.15 | 0.24 | 139 |
| 5 | 0.00 | 0.00 | 0.00 | 166 |
| 6 | 0.50 | 0.31 | 0.38 | 914 |
| 7 | 0.09 | 0.02 | 0.03 | 228 |
| 8 | 0.31 | 0.14 | 0.20 | 251 |
| 9 | 0.43 | 0.07 | 0.12 | 389 |
| 10 | 0.29 | 0.06 | 0.10 | 501 |
| 11 | 0.62 | 0.95 | 0.75 | 5184 |
| micro avg | 0.58 | 0.52 | 0.55 | 10563 |
| macro avg | 0.34 | 0.16 | 0.19 | 10563 |
| weighted avg | 0.47 | 0.52 | 0.45 | 10563 |
| samples avg | 0.58 | 0.60 | 0.58 | 10563 |

## MobileNet_v2 Per-label Metrics

| Label | Precision | Recall | F1-score | Support |
| --- | --- | --- | --- | --- |
| 0 | 0.24 | 0.18 | 0.21 | 814 |
| 1 | 0.13 | 0.09 | 0.11 | 369 |
| 2 | 0.27 | 0.31 | 0.29 | 1266 |
| 3 | 0.32 | 0.27 | 0.29 | 342 |
| 4 | 0.26 | 0.19 | 0.22 | 139 |
| 5 | 0.25 | 0.07 | 0.10 | 166 |
| 6 | 0.39 | 0.41 | 0.40 | 914 |
| 7 | 0.07 | 0.04 | 0.05 | 228 |
| 8 | 0.35 | 0.27 | 0.31 | 251 |
| 9 | 0.18 | 0.17 | 0.18 | 389 |
| 10 | 0.23 | 0.16 | 0.19 | 501 |
| 11 | 0.66 | 0.82 | 0.73 | 5184 |
| micro avg | 0.50 | 0.52 | 0.51 | 10563 |
| macro avg | 0.28 | 0.25 | 0.26 | 10563 |
| weighted avg | 0.46 | 0.52 | 0.48 | 10563 |
| samples avg | 0.53 | 0.58 | 0.54 | 10563 |

## Installation

To set up the environment, run the following commands:

```bash
conda activate pytorch_env
pip install tkinterdnd2
```
