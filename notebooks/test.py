import pandas as pd

train = pd.read_csv("data/processed/ratings_train.csv")
test = pd.read_csv("data/processed/ratings_test.csv")

print(train.head())
print(test.head())
print(len(train), len(test))
