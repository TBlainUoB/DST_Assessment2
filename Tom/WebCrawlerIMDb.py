import requests
import pandas as pd
from bs4 import BeautifulSoup

# Set the website URL
url = "https://www.imdb.com/chart/top/?ref_=nv_mv_250"

# Make a GET request to the website
response = requests.get(url)

# Parse the HTML content of the website
soup = BeautifulSoup(response.content, "html.parser")

#print(soup)

# Find all elements with the class "movie-title"
movie_titles = soup.find_all(class_="titleColumn")

# Extract the text from the elements and store them in a list
titles = []
for title in movie_titles:
    titles.append(title.get_text())

cleaned_titles = [titles.split('\n')[2][6:] for titles in titles]

# Print the list of movie titles
#print(cleaned_titles)

df = pd.DataFrame(cleaned_titles)
df.to_csv("MovieTitles2.csv", index = False)