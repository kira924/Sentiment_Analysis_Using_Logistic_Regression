# Sentiment Analysis Using Logistic Regression

## Overview
This project performs sentiment analysis on the Sentiment140 dataset, 
classifying tweets as either positive or negative based on their content.
It demonstrates foundational machine learning techniques, including data preprocessing,
feature extraction, model training, and evaluation.

## Features
- **Data Extraction**: Automatically downloads and loads the Sentiment140 dataset using the Kaggle API.
- **Data Preprocessing**: Cleans and prepares text data for analysis.
- **Feature Extraction**: Converts text data into numerical features using the TF-IDF vectorization technique.
- **Model Training**: Trains a Logistic Regression model for binary sentiment classification.
- **Evaluation**: Evaluates the model with metrics like accuracy and a confusion matrix.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   ```

2. Navigate to the project directory:
   ```bash
   cd your-repo-name
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements
   ```

4. Set up Kaggle credentials to access the Sentiment140 dataset. Follow the [Kaggle API setup guide](https://www.kaggle.com/docs/api).

## Usage

1. Run the notebook file `Enhanced_Sentiment_Analysis_Project.ipynb` in Jupyter Notebook or JupyterLab.
2. Follow the workflow:
   - Download and load the dataset.
   - Preprocess the data.
   - Extract features using TF-IDF.
   - Train the Logistic Regression model.
   - Evaluate the model's performance.

## Results
The model achieves accuracy suitable for foundational sentiment analysis tasks.
Detailed results, including the confusion matrix and accuracy score, are displayed in the notebook.

## Future Enhancements
- Add more advanced machine learning models like SVMs or neural networks for performance comparison.
- Deploy the model as a web app using frameworks like Flask or Streamlit.
