import pandas as pd
import numpy as np
import requests
from APIKEY import api

# Set the API endpoint URL
url = "http://www.omdbapi.com/"
api_key = api()

# Set the API key and search parameters
titles_british = pd.read_csv("MovieTitles2000-2021_British.csv")["0"].to_list()
titles_american = pd.read_csv("MovieTitles2000-2021_American.csv")["0"].to_list()
titles = titles_british + titles_american
#titles = list(titles)
#print(titles)
movie_data = []
count = 0
for i in titles:
    response = requests.get(url, params={
        "apikey": api_key,
        "t": i,
        # "y": year,
        "plot": "full",
        "type": "movie"
    })
    data = response.json()
    count += 1
    if count % 10 == 0:
        print(data["Response"], count)

    if data["Response"] == 'True':
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
df.to_csv("Movie_database_BritishAmerican2000-2021.csv", index=False)

# Print the DataFrame
#610