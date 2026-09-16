# Credit Card Default Prediction

MSc Artificial Intelligence — Introduction to Artificial Intelligence
Practical Skills Assessment (Predictive Modelling Using Machine Learning)

## Overview

This project predicts whether a credit card client will default on their payment
next month (`default.payment.next.month`), using a dataset of 34,788 clients
based on the "Default of Credit Card Clients" dataset (Yeh and Lien, 2009),
extended with additional derived columns (`CITY`, `RISK_RATING`, `BILL_AMT_SUM`,
`LIMIT_BAL_LOG`, `risk_leak`).

The notebook covers, in order:

1. **Exploratory Data Analysis** — class imbalance, descriptive statistics,
   distributions, correlations, and payment-history patterns.
2. **Data Preparation** — missing-value handling, encoding, scaling, and a
   before/after example, all inside a leakage-safe scikit-learn pipeline.
3. **Model Training** — five classifiers trained with default hyperparameters:
   Logistic Regression, Support Vector Machine, Decision Tree, Random Forest,
   K-Nearest Neighbors.
4. **Model Evaluation & Interpretation** — accuracy, precision, recall,
   F1-score and ROC-AUC for every model; confusion matrices and ROC curves;
   hyperparameter tuning of the best model with `RandomizedSearchCV`; and
   SHAP-based interpretation of the tuned model's predictions.
5. **Conclusion** — see the accompanying report for the full discussion,
   including an important finding around likely data leakage via `risk_leak`.

## Files

| File | Description |
|---|---|
| `Credit_Card_Default_Prediction.ipynb` | Full analysis notebook (code + outputs) |
| `Credit_Card.csv` | Dataset used (34,788 rows, 30 features) |
| `README.md` | This file |

## How to Run

1. Install the required packages:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn shap joblib
   ```
2. Make sure `Credit_Card.csv` is in the same folder as the notebook.
3. Open the notebook and run all cells top to bottom:
   ```bash
   jupyter notebook Credit_Card_Default_Prediction.ipynb
   ```

## Key Result

Random Forest was the best-performing model by ROC-AUC (0.9984 before tuning,
0.9987 after tuning with `RandomizedSearchCV`). However, the near-perfect
scores from the tree-based models (Decision Tree: 99.8% accuracy, 99.25%
recall) are unusually high for a real-world credit-default problem and are
discussed in the report as strong evidence of data leakage, most likely via
the `risk_leak` feature.

## References

- Baesens, B., Van Gestel, T., Viaene, S., Stepanova, M., Suykens, J. and
  Vanthienen, J. (2003) 'Benchmarking state-of-the-art classification
  algorithms for credit scoring', *Journal of the Operational Research
  Society*, 54(6), pp. 627-635.
- Chawla, N.V., Bowyer, K.W., Hall, L.O. and Kegelmeyer, W.P. (2002) 'SMOTE:
  Synthetic minority over-sampling technique', *Journal of Artificial
  Intelligence Research*, 16, pp. 321-357.
- Lessmann, S., Baesens, B., Seow, H.-V. and Thomas, L.C. (2015)
  'Benchmarking state-of-the-art classification algorithms for credit
  scoring: An update of research', *European Journal of Operational
  Research*, 247(1), pp. 124-136.
- Lundberg, S.M. and Lee, S.-I. (2017) 'A unified approach to interpreting
  model predictions', *Advances in Neural Information Processing Systems*,
  30, pp. 4765-4774.
- Yeh, I.-C. and Lien, C.-H. (2009) 'The comparisons of data mining
  techniques for the predictive accuracy of probability of default of
  credit card clients', *Expert Systems with Applications*, 36(2),
  pp. 2473-2480.
