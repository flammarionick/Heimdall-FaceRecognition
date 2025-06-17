Heimdall Face Classification: Neural Network vs Classical ML

Problem Statement
This project explores the implementation of supervised learning models on the FacePix dataset to classify grayscale facial images of individuals.
The goal was to evaluate the performance of baseline and optimized neural networks versus classical machine learning models.
We applied optimization techniques such as dropout, early stopping, and learning rate tuning.
We also performed hyperparameter tuning using GridSearchCV to improve the accuracy of classical models.

Dataset
Source: Local RecordIO-based dataset (FacePix)

Images: 60×51 grayscale face images

Classes: 30 unique individuals

Total samples: Over 3,000 images with labels

Preprocessing: Image normalization, label encoding, and flattening into 1D arrays

Findings & Result Summary
Model Instance	Optimizer	Epochs	EarlyStopping	Loss	Accuracy	Precision	Recall	F1-score
Baseline NN (instance_1)	Adam	10	No	2.9906	0.0785	0.0310	0.0790	0.0303
Optimized NN (instance_2)	RMSprop	30	Yes	2.7163	0.1939	0.1859	0.1927	0.1371
Deep NN (instance_3)	Adam	40	Yes	3.3983	0.0331	0.0012	0.0333	0.0023
Shallow NN (instance_4)	SGD	50	Yes	3.3996	0.0368	0.0041	0.0370	0.0061
Logistic Regression	N/A	N/A	No	0.0143	0.9988	0.9989	0.9988	0.9988
Logistic Regression (Tuned)	N/A	N/A	No	0.0143	0.9988	0.9989	0.9988	0.9988

Summary of Best Combination
The best-performing combination was Logistic Regression with hyperparameter tuning, achieving ~99.88% accuracy, near-perfect precision, recall, and F1-score. This model used:

solver='lbfgs'

C=1.0

multi_class='multinomial'

max_iter=1000

This significantly outperformed all neural network implementations, including those with optimization.

Neural Networks vs. Classical ML: What Worked Better?
While neural networks showed moderate improvements after optimization (especially instance_2 with RMSprop), they were still vastly outperformed by Logistic Regression. The classical ML approach worked better for this structured, low-resolution dataset due to its simplicity and well-separated class boundaries.

Key Observations:
NN models were sensitive to optimizer and depth, often underperforming due to overfitting or vanishing gradients.

Logistic Regression remained robust across all hyperparameter ranges and handled multiclass separation very well after tuning.

GridSearchCV helped identify the best combination of C and solver, making Logistic Regression not only effective but extremely efficient.

How to Run the Notebook
Clone this repository:

git clone https://github.com/yourusername/heimdall-facepix-classifier.git
cd heimdall-facepix-classifier
Ensure the following packages are installed:

numpy, pandas, opencv-python

scikit-learn

tensorflow

joblib

Open the Jupyter Notebook:

jupyter notebook heimdall_facepix_classification.ipynb
Ensure the following files and folders are in place:

Dataset directory: FacePix/

Metadata CSV: facepix_metadata_named.csv

Saved models will be stored in: saved_models/

Load Best Saved Model
To use the best performing model:


import joblib
model = joblib.load("saved_models/logistic_regression_model.pkl")
predictions = model.predict(X_test)
