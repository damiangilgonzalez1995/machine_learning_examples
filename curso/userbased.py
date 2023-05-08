# TEORIA INTERESANTE
# https://towardsdatascience.com/collaborative-filtering-based-recommendation-systems-exemplified-ecbffe1c20b1
# NOTEBOOK IMPORTANTE
# https://github.com/csaluja/JupyterNotebooks-Medium/blob/master/CF%20Recommendation%20System-Examples.ipynb

from __future__ import print_function, division
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from datetime import datetime
from sortedcontainers import SortedList

# load in the data
import os

if not os.path.exists('user2movie.json') or \
        not os.path.exists('movie2user.json') or \
        not os.path.exists('usermovie2rating.json') or \
        not os.path.exists('usermovie2rating_test.json'):
    import preprocess2dict

with open('user2movie.json', 'rb') as f:
    user2movie = pickle.load(f)

with open('movie2user.json', 'rb') as f:
    movie2user = pickle.load(f)

with open('usermovie2rating.json', 'rb') as f:
    usermovie2rating = pickle.load(f)

with open('usermovie2rating_test.json', 'rb') as f:
    usermovie2rating_test = pickle.load(f)

N = np.max(list(user2movie.keys())) + 1
# the test set may contain movies the train set doesn't have data on
m1 = np.max(list(movie2user.keys()))
m2 = np.max([m for (u, m), r in usermovie2rating_test.items()])
M = max(m1, m2) + 1
print("N:", N, "M:", M)

if N > 10000:
    print("N =", N, "are you sure you want to continue?")
    print("Comment out these lines if so...")
    exit()

# to find the user similarities, you have to do O(N^2 * M) calculations!
# in the "real-world" you'd want to parallelize this
# note: we really only have to do half the calculations, since w_ij is symmetric

print(len(user2movie))

def calculate_weight(user2movie, usermovie2rating):
    count = 0
    neighbors = {}  # store neighbors in this list
    K = 25  # number of neighbors we'd like to consider
    limit = 5  # number of common movies users must have in common in order to consider

    for i in user2movie.keys():

        print(100 * (count / len(user2movie)))
        count += 1

        sl = SortedList()

        movies_i = user2movie[i]
        movies_i_set = set(movies_i)

        for j in user2movie.keys():

            if i != j:
                movies_j = user2movie[j]
                movies_j_set = set(movies_j)

                common_movies = (movies_i_set & movies_j_set)  # intersection
                if len(common_movies) > limit:
                    ratings_i = {movie: usermovie2rating[(i, movie)] for movie in common_movies}
                    avg_i = np.mean(list(ratings_i.values()))
                    dev_i = {movie: (rating - avg_i) for movie, rating in ratings_i.items()}
                    dev_i_values = np.array(list(dev_i.values()))
                    sigma_i = np.sqrt(dev_i_values.dot(dev_i_values))

                    # # save these for later use
                    # averages.append(avg_i)
                    # deviations.append(dev_i)

                    ratings_j = {movie: usermovie2rating[(j, movie)] for movie in common_movies}
                    avg_j = np.mean(list(ratings_j.values()))
                    dev_j = {movie: (rating - avg_j) for movie, rating in ratings_j.items()}
                    dev_j_values = np.array(list(dev_j.values()))
                    sigma_j = np.sqrt(dev_j_values.dot(dev_j_values))

                    numerator = sum(dev_i[m] * dev_j[m] for m in common_movies)
                    w_ij = numerator / (sigma_i * sigma_j)

                    sl.add((-w_ij, j))

                    if len(sl) > K:
                        del sl[-1]

                # store the neighbors
                neighbors[i] = sl

    return neighbors


def calculate_rate(i, m, neighbors, usermovie2rating, user2movie):
    neighborns_i = neighbors[i]
    movies_i = user2movie[i]
    averages_i = np.mean([usermovie2rating[(i, movie)] for movie in movies_i])
    numerator = 0
    denominator = 0

    for w_ij, j in neighborns_i:

        try:
            movies_j = user2movie[j]
            ratings_j = {movie: usermovie2rating[(j, movie)] for movie in movies_j}
            avg_j = np.mean(list(ratings_j.values()))

            dev_j = usermovie2rating[(j, m)] - avg_j
            numerator += dev_j * (- w_ij)
            denominator += abs(w_ij)

        except:
            pass

    if denominator == 0:
        prediction = averages_i
    else:
        prediction = numerator / denominator + averages_i
        prediction = min(5, prediction)
        prediction = max(0.5, prediction)  # min rating is 0.5

    return prediction


neighbors = calculate_weight(user2movie, usermovie2rating)

with open('neighbors.json', 'wb') as f:
  pickle.dump(neighbors, f)

train_predictions = []
train_targets = []
for (i, m), target in usermovie2rating.items():
  # calculate the prediction for this movie
  prediction = calculate_rate(i, m, neighbors, usermovie2rating, user2movie)

  # save the prediction and target
  train_predictions.append(prediction)
  train_targets.append(target)

test_predictions = []
test_targets = []
# same thing for test set
for (i, m), target in usermovie2rating_test.items():
  # calculate the prediction for this movie
  prediction = calculate_rate(i, m, neighbors, usermovie2rating, user2movie)

  # save the prediction and target
  test_predictions.append(prediction)
  test_targets.append(target)


# calculate accuracy
def mse(p, t):
  p = np.array(p)
  t = np.array(t)
  return np.mean((p - t)**2)

print('train mse:', mse(train_predictions, train_targets))
print('test mse:', mse(test_predictions, test_targets))