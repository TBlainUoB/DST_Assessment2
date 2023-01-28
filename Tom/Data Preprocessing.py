import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

# Load your data into a pandas DataFrame
df = pd.read_csv("Movie_database_BritishAmerican2000-2021.csv")

# Create a list of unique genres in the dataset
genres = list(set(df["Genre"].tolist()))
print(genres)

genres = list(filter(lambda x: isinstance(x, str), genres))
genres = [triplet.split(',') for triplet in genres]
splitgenre = list(set([word.replace(" ", "") for triplet in genres for word in triplet]))
print(splitgenre)

film_genres = df['Genre'].str.split(',', expand=True).replace(" ", "")
film_genres = film_genres.apply(lambda x: x.str.strip())

# One-hot encode the new genre columns and append to the film dataframe
genre_dummies = pd.get_dummies(film_genres.stack()).groupby(level=0).sum()
film_df = pd.concat([df, genre_dummies], axis=1)

# drop original genre column
film_df.drop('Genre', axis=1, inplace=True)

# using label encoder
film_actors = df['Actors'].str.split(',', expand=True).replace(" ", "")
film_actors = film_actors.apply(lambda x: x.str.strip())
le = LabelEncoder()
film_actors = film_actors.apply(le.fit_transform)
# Concatenate actors with original dataframe
film_df = pd.concat([film_df, film_actors], axis=1)

# drop original actors column
film_df.drop('Actors', axis=1, inplace=True)


film_df.to_csv("MovieData_OHE_actors.csv", index=False)
