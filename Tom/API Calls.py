import pandas as pd
import numpy as np
import requests
from APIKEY import api

# Set the API endpoint URL
url = "http://www.omdbapi.com/"

# Set the API key and search parameters
api_key = api()
search_term = "lego batman"
year = ""

# Make the GET request to the API
response = requests.get(url, params={
    "apikey": api_key,
    "t": search_term,
    # "y": year,
    "plot": "full",
    "type": "movie"
})

# Get the JSON data from the response
data = response.json()

# Create a list to store the movie data
movie_data = []

# Iterate over the movie results and store them in the list
movie_data.append({
    "IMDbRating": data["imdbRating"],
    "Title": data["Title"],
    "Year": data["Year"],
    "Genre": data["Genre"],
    "Plot": data["Plot"],
    "Actors": data["Actors"],
})

# Create a DataFrame from the movie data
df = pd.DataFrame(movie_data)
df.to_csv("Movie_database.csv", index=False)

# Print the DataFrame
print(df)
