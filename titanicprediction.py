import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('Titanic-Dataset.csv')


print("Initial Data Preview:")
print(df.head())
print("\nData Summary:")
print(df.info())

df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])


label_encoder = LabelEncoder()
df['Sex'] = label_encoder.fit_transform(df['Sex'])        
df['Embarked'] = label_encoder.fit_transform(df['Embarked'])
features = ['Pclass', 'Sex', 'Age', 'Embarked']
target = 'Survived'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=10)

model = LogisticRegression(max_iter=300)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.6f}")

results_df = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred})
print(results_df.head())
