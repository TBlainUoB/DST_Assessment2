import requests
import pandas as pd
from bs4 import BeautifulSoup

filmdata = []
# Set the website URL
year = ['00','01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22']
for i in year:
    url = (f"https://en.wikipedia.org/wiki/List_of_American_films_of_20{i}")

    # Make a GET request to the website
    response = requests.get(url)

    # Parse the HTML content of the website
    soup = BeautifulSoup(response.content, "html.parser")
    # print(soup)
    movie_titles = []
    # Find all elements with the class "movie-title"
    rows = soup.find_all('tr')
    for row in rows:
        title = row.find('a')
        if title:
            movie_titles.append(title.text)
    print(movie_titles)
    movie_titles = movie_titles[29:]
    movie_titles = movie_titles[:len(movie_titles) - 30]
    print(movie_titles)
    for i in movie_titles:
        filmdata.append(i)

# print(movie_titles)
# Print the list of movie titles
# print(cleaned_titles)

df = pd.DataFrame(filmdata)
df.to_csv("MovieTitles2000-2021_American.csv", index=False)
