
# Titanic Survival Prediction with Logistic Regression

This project uses a Logistic Regression model to predict the survival of passengers aboard the Titanic based on selected features from the Titanic dataset.

## 📂 Dataset

- Dataset used: `Titanic-Dataset.csv`
- Ensure the dataset file is placed in the same directory as the script.

## 🔍 Features Used

The model uses the following features for prediction:

- `Pclass` (Passenger Class)
- `Sex` (Gender - encoded)
- `Age` (Numerical - missing values filled with mean)
- `Embarked` (Port of Embarkation - encoded and missing values filled with mode)

## 📦 Libraries Used

- `pandas` for data handling
- `sklearn` for model training and evaluation
  - `train_test_split` for data splitting
  - `LogisticRegression` for model
  - `accuracy_score` for model evaluation
  - `LabelEncoder` for converting categorical variables

## ⚙️ Preprocessing Steps

1. Load and preview the dataset.
2. Fill missing `Age` values with the column mean.
3. Fill missing `Embarked` values with the mode.
4. Encode categorical columns: `Sex` and `Embarked` using `LabelEncoder`.

## 🧠 Model Training

- A Logistic Regression model is trained with the selected features.
- The dataset is split with 90% training and 10% testing.
- The model is trained with a maximum of 300 iterations to ensure convergence.

## 📈 Evaluation

- The model's performance is evaluated using **accuracy score** on the test set.
- A sample of actual vs predicted values is displayed.

## 💻 Example Output

```
Initial Data Preview:
   PassengerId  Survived  Pclass  ...    Fare Cabin  Embarked
0            1         0       3  ...  7.2500   NaN         S
...

Model Accuracy: 0.800000

   Actual  Predicted
0       0          0
1       1          1
...
```

## ✅ How to Run

Make sure you have Python installed. Then, install the required packages (if not already installed):

```bash
pip install pandas scikit-learn
```

Run the script:

```bash
python titanic_logistic.py
```

## 📌 Notes

- This is a basic model and does not include advanced feature engineering or hyperparameter tuning.
- Accuracy may vary depending on data splits and preprocessing choices.
