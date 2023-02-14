import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras.losses import binary_crossentropy


def RemoveNA(df):
    df = df.replace("N/A", np.nan)
    df = df.dropna()
    return df


def RemoveDuplicates(df):
    df = df.drop_duplicates()
    return df


def OHE_Genre(df):
    genres = list(set(df["Genre"].tolist()))

    film_genres = df['Genre'].str.split(',', expand=True).replace(" ", "")
    film_genres = film_genres.apply(lambda x: x.str.strip())

    # One-hot encode the new genre columns and append to the film dataframe
    genre_dummies = pd.get_dummies(film_genres.stack()).groupby(level=0).sum()
    film_df = pd.concat([df, genre_dummies], axis=1)

    # drop original genre column
    film_df.drop('Genre', axis=1, inplace=True)
    return film_df

'''
def VAE_actors(df):
    # Get all unique actors in the dataset
    film_actors = df['Actors'].str.split(',', expand=True).replace(" ", "")
    film_actors = film_actors.apply(lambda x: x.str.strip())

    actor_dummies = pd.get_dummies(film_actors.stack()).groupby(level=0).sum()

    # Sum the number of movies each actor has appeared in
    actor_counts = actor_dummies.sum(axis=0)

    # Filter out actors with less than 5 movie appearances
    actor_counts = actor_counts[actor_counts >= 5]

    # Select only the one-hot encoded columns for actors with more than 5 movie appearances
    actor_dummies = actor_dummies[actor_counts.index]
    df = pd.concat([df, actor_dummies], axis=1)
    df.drop('Actors', axis=1, inplace=True)

    # Create VAE
    latent_dim = 20
    encoder_inputs = Input(shape=(actor_dummies.shape[1],), name='encoder_input')
    encoder = Dense(64, activation='relu')(encoder_inputs)
    encoder = Dense(32, activation='relu')(encoder)
    encoder_outputs = Dense(latent_dim, activation='linear')(encoder)

    decoder_inputs = Input(shape=(latent_dim,), name='decoder_input')
    decoder = Dense(32, activation='relu')(decoder_inputs)
    decoder = Dense(64, activation='relu')(decoder)
    decoder_outputs = Dense(actor_dummies.shape[1], activation='sigmoid')(decoder)

    encoder = Model(encoder_inputs, encoder_outputs, name='encoder')
    decoder = Model(decoder_inputs, decoder_outputs, name='decoder')

    # VAE model
    outputs = decoder(encoder(encoder_inputs))
    vae = Model(encoder_inputs, outputs, name='vae')

    # Compile
    vae.compile(optimizer='adam', loss='binary_crossentropy')

    # Fit VAE
    vae.fit(actor_dummies, actor_dummies, epochs=50, batch_size=32)

    # Predict on OHE actors data
    encoded_actors = encoder.predict(actor_dummies)

    # Add the encoded actors to the original dataframe
    for i in range(latent_dim):
        df['encoded_actor_' + str(i)] = encoded_actors[:, i]

    return df
'''
df = pd.read_csv("Movie_database_BritishAmerican2000-2021.csv")
df = RemoveDuplicates(df)
df = RemoveNA(df)
df = OHE_Genre(df)
#df = VAE_actors(df)

df.to_csv("PreProcessedData.csv", index=False)
