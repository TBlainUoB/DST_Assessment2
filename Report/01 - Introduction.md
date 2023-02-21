# Introduction

IMDb (Internet Movie Database) is one of the leading communities surrounding film reviews.
This gives rise to lots of available data. An IMDb page will typically list out all of the cast, directors, runtime, awards etc.
As well as having a score which is measured as a weighted average user ratings.

This gives motivation to our project. Can we train a model which can take in features pulled from IMDb to then predict the movie rating.
Specifically, we will look at the summary of the film's plot given by IMDb. These are generally about a small to mid length paragraph of text summing up the film, which we will try to use to create a topic model to be used in our wider prediction model. This is strongly motivated by the fact that these plot summaries are of a workable size to use, and by their nature, are already full of significant keywords to describe the film as concisely as possible. A model like this might allow us to predict ratings of future film releases, and learn more about what plot elements or keywords might make a movie score highly.

The outline of the project is as follows

 - We will use a web crawler to gather a list of movie titles which we can then use to pull the data from IMDb.
 - We then conduct EDA to find insights into our data.
 - We then preprocess the text, and use LDA topic modelling to utilise the movie plots for prediction
 - We implement an XGBoost model to predict a movie rating out of 10.
