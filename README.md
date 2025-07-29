# ReviewRanker
Predictive model to classify and rank helpful book reviews using NLP and metadata features.

## 📌 Project Overview

**ReviewRanker** is a machine learning pipeline developed to predict and rank the helpfulness of book reviews on an e-commerce platform. By identifying the most helpful reviews, I aim to improve user experience by surfacing insightful feedback first.

## 📂 Dataset

**Source:** [Amazon Book Reviews Dataset (Kaggle)](https://www.kaggle.com/datasets/cynthiarempel/amazon-us-customer-reviews-dataset?select=amazon_reviews_us_Books_v1_02.tsv)

- Subset: `amazon_reviews_us_Books_v1_02.tsv`
- Filtered to reviews from `2003` to `2005`.

## ⚙️ Project Structure

ReviewRanker/
│
├── data/ # Raw and cleaned datasets (if any)
├── notebooks/
│ └── helpful_review_model.ipynb # Main analysis and modeling notebook
├── models/ # Saved models or vectorizers (if any)
├── utils/ # Utility functions (optional)
├── requirements.txt # Python dependencies
├── README.md # Project overview (this file)
└── results/ # Evaluation reports, figures


