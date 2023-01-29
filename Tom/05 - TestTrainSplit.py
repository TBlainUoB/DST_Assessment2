from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("Model_W2V.csv")
rating = df['IMDbRating']
xdf = df.drop('IMDbRating', axis=1, inplace=False)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(xdf, rating, test_size=0.2, random_state=42)

X_train_df = pd.DataFrame(X_train)
X_test_df = pd.DataFrame(X_test)
y_train_df = pd.DataFrame(y_train)
y_test_df = pd.DataFrame(y_test)

# Save the dataframes to csv files
X_train_df.to_csv('X_train.csv', index=False)
X_test_df.to_csv('X_test.csv', index=False)
y_train_df.to_csv('y_train.csv', index=False)
y_test_df.to_csv('y_test.csv', index=False)