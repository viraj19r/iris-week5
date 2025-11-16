Iris Data Poisoning Experiment

This project demonstrates how data poisoning affects machine-learning model performance on the Iris dataset. Random noise is injected into the training data at different poisoning levels, and the model is trained and logged using MLflow.

What is Data Poisoning?

Data poisoning is an attack where incorrect or noisy data is added to the training set.
This causes the model to learn wrong patterns and reduces its accuracy.

How Poisoning is Added

A percentage of training rows (5%, 10%, 50%) are selected and their feature values are replaced with random numbers in the range 1–8.

Example:

df_poisoned.loc[idx, feature_cols] = np.random.uniform(1, 8, size=len(feature_cols))


Only the training data is poisoned.
The test dataset remains clean for proper evaluation.

Results
Poison Level	Accuracy
0%	0.9048
5%	0.8571
10%	0.7619
50%	0.8095

Cross-validation scores also drop as poisoning increases.

Conclusion:
More poisoned data → Lower accuracy and unstable validation results.

Mitigation

Use data validation checks

Anomaly detection before training

Collect more clean data when quality is low

Use robust models or noise-resistant training techniques