# Loan Repayment Propensity Prediction

## Objective
Predict the probability that a borrower will pay any outstanding loan amount within the next 30 days.

## Approach
- Behaviorally realistic synthetic data (until real data is available)
- Interpretable baseline + gradient boosting model
- Probability calibration for reliability
- Actionable customer segmentation for collections strategy

## Target Variable
paid_within_30_days (1 = payment made, 0 = no payment)

## Folder Notes
- data/raw → synthetic data
- data/real → real data drop-in (same schema)
- notebooks → end-to-end analysis
- outputs → plots, models, tables shown to judges
