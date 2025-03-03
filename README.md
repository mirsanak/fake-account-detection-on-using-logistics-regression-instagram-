# fake-account-detection-on-using-logistics-regression-
Fake Account Detection on Instagram using Logistic Regression

Description

This project implements Logistic Regression to detect fake social media accounts on Instagram. By analyzing account features, the model predicts whether an account is real or fake.

Logistic Regression is a supervised machine learning algorithm used for binary classification tasks, making it a suitable choice for fake account detection. In this scenario, it predicts whether an Instagram account is fake (1) or real (0) based on selected features.

1. How Logistic Regression Works in Fake Account Detection
Feature Extraction:

Collect Instagram account data, such as:
Number of followers
Number of followings
Post count
Engagement rate (likes/comments per post)
Account age
Preprocessing the Data:

Normalize the features for better performance.
Split data into training and testing sets.
Building the Logistic Regression Model:

Applies the sigmoid function to map predictions between 0 (real) and 1 (fake).
Calculates the probability of an account being fake.
Training the Model:

Uses a dataset of labeled fake and real accounts.
Optimizes using Gradient Descent to adjust weights.
Prediction & Evaluation:

If the probability is >0.5, classify as a fake account.
Evaluate accuracy, precision, recall, and F1-score.

Installation

Prerequisites

Python 3.x

Jupyter Notebook

Scikit-Learn for Logistic Regression

Setup

Clone this repository:

git clone https://github.com/yourusername/fake-account-detection-logistic.git
cd fake-account-detection-logistic

Install dependencies:

pip install -r requirements.txt

Open the Jupyter Notebook:

jupyter notebook

Run logistic.ipynb to train and test the model.

Usage

Open logistic.ipynb in Jupyter Notebook.

Run all cells to load the dataset, train the model, and evaluate performance.

Check accuracy and classification metrics for results.

Dataset

The dataset contains Instagram account features, including:

Number of followers

Number of followings

Post count

Engagement rate (likes/comments per post)

Account age

Source: Custom dataset (or specify if using a public dataset)

Model & Approach

Machine Learning Model: Logistic Regression

Working Principle:

Uses a sigmoid function to calculate the probability of an account being fake.

If probability > 0.5, classify as a fake account.

Trained using labeled fake and real accounts.

Performance Metrics:

Accuracy, Precision, Recall, and F1-score

