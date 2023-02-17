# Introduction

IMDb (Internet Movie Database) is one of the leading communities surrounding film reviews.
This gives rise to lots of available data. An IMDb page will typically list out all of the cast, directors, runtime, awards etc.
As well as having a score which is measured as a weighted average user ratings.

This gives motivation to our project. Can we train a model which can take in features pulled from IMDb to then predict the movie rating.
This system could be designed to rely heavily on the plot of the film for information on its prediction.
A model like this might allow us to predict ratings of future film releases, and learn more about what might make a movie score highly.

The outline of the project is as follows

We will use a web crawler to gather a list of movie titles which we can then use to pull the data from IMDb.
We then conduct EDA to find insights into our data. We then use LDA topic modelling to utalise the movie plots for prediction, and use an XGBoost model to predict a movie rating out of 10.