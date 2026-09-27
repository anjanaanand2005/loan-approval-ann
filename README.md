# Loan Approval Prediction — PyTorch ANN

A feed-forward neural network that predicts whether a loan application will be approved, based on applicant financial and demographic features.

## Dataset
~20,000-record "Financial Risk for Loan Approval" dataset (Kaggle), 34 features per applicant.

## Approach
- Built a feed-forward ANN in PyTorch (256 → 128 → 64 → 1, ~51K parameters) with BatchNorm and Dropout for regularization.
- Handled class imbalance using a weighted loss function.
- Trained for 30 epochs using AdamW optimizer with cosine annealing learning rate scheduling.

## Result
**94% test accuracy, 0.99 AUC** — with strong precision/recall on both approval and rejection classes.

## Files
- `loan_approval_ann.py` — full training and evaluation pipeline
