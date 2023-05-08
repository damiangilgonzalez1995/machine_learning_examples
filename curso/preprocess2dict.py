# https://udemy.com/recommender-systems
# https://deeplearningcourses.com/recommender-systems
from __future__ import print_function, division
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

# load in the data
# https://www.kaggle.com/grouplens/movielens-20m-dataset
df = pd.read_csv('../data/very_small_rating.csv')

N = df.userId.max() + 1 # number of users
M = df.movie_idx.max() + 1 # number of movies

# split into train and test
df = shuffle(df)
cutoff = int(0.8*len(df))
df_train = df.iloc[:cutoff]
df_test = df.iloc[cutoff:]


def user2movie_movie2user(df, col_user, col_movies):
  user2movie = {}
  movie2user = {}
  usermovie2rating = {}

  for id_user in df[col_user].unique().tolist():
    df_aux = df[df[col_user] == id_user]
    list_movies_rating = df_aux[[col_movies, "rating"]].values.tolist()
    user2movie[id_user] = [elem[0] for elem in list_movies_rating]

    for elem in list_movies_rating:
      id_movies = int(elem[0])

      usermovie2rating[(id_user, id_movies)] = elem[1]

      if id_movies not in movie2user:
        movie2user[id_movies] = [id_user]
      else:
        movie2user[id_movies].append(id_user)

  return user2movie, movie2user, usermovie2rating


user2movie, movie2user, usermovie2rating = user2movie_movie2user(df_train, "userId", "movie_idx")


def update_usermovie2rating(df, col_user, col_movies):
  usermovie2rating_update = {}
  for index, row in df.iterrows():
    usermovie2rating_update[(row[col_user], row[col_movies])] = row.rating

  return usermovie2rating_update


usermovie2rating_test = update_usermovie2rating(df_test, "userId", "movie_idx")

# note: these are not really JSONs
with open('user2movie.json', 'wb') as f:
  pickle.dump(user2movie, f)

with open('movie2user.json', 'wb') as f:
  pickle.dump(movie2user, f)

with open('usermovie2rating.json', 'wb') as f:
  pickle.dump(usermovie2rating, f)

with open('usermovie2rating_test.json', 'wb') as f:
  pickle.dump(usermovie2rating_test, f)
