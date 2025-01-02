This is a short program made to kick me off on my New Year's resolution to actually start to build up some personal projects.
# Titanic Survival Feature Importance

This project predicts survival outcomes for passengers on the Titanic using machine learning. It involves data preprocessing, exploratory data analysis, and model training with Logistic Regression and Random Forest.

## Overview
The Titanic dataset provides information on passengers, such as age, sex, class, and whether they survived. This project demonstrates a machine learning pipeline, from data cleaning to model evaluation, to predict survival based on passenger features.

### Goals
- Handle missing values in the dataset.
- Perform feature encoding for categorical variables.
- Train and evaluate Logistic Regression and Random Forest models.
- Visualize feature importance for interpretability.

## Technologies Used
- Python 3.11
- Libraries:
  - Pandas
  - NumPy
  - Scikit-learn
  - Seaborn
  - Matplotlib

## Setup and Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/titanic-survival-feature-importance.git
   
2. Navigate to the project directory:
   cd titanic-survival-feature-importance

3. Install the required packages:
   pip install -r requirements.txt

4. Run the project:
   python test.py

## Usage
The script test.py contains all the code for data preprocessing, model training, and evaluation. Modify or extend the script to test different models or preprocessing techniques.

## Project Structure
titanic-survival-prediction/
├── test.py               # Main script

├── requirements.txt      # Dependencies

├── README.md             # Project documentation

└── .gitignore            # Ignored files for Git
## Features
Fills missing values for age and embarked.
Trains Logistic Regression and Random Forest models.
Evaluates models using accuracy, confusion matrix, and classification report.
Visualizes feature importance for the Random Forest model.
## Results
Logistic Regression Accuracy: ~[81.0%]
Random Forest Accuracy: ~[82.1%]
Feature importance visualization provides insights into which features most influenced the predictions.
![Feature_Importance](https://github.com/user-attachments/assets/b9ee8f4b-0db0-45da-b77c-e41b13fa37a0)

## Acknowledgments
Titanic dataset courtesy of Seaborn.
Inspired by data science and machine learning courses.


