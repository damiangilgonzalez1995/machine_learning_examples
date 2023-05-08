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
from datetime import datetime

# load in the data
import os
# if not os.path.exists('user2movie.json') or \
#    not os.path.exists('movie2user.json') or \
#    not os.path.exists('usermovie2rating.json') or \
#    not os.path.exists('usermovie2rating_test.json'):
#    import preprocess2dict

path = "../"
with open(path+'user2movie.json', 'rb') as f:
  user2movie = pickle.load(f)

with open(path+'movie2user.json', 'rb') as f:
  movie2user = pickle.load(f)

with open(path+'usermovie2rating.json', 'rb') as f:
  usermovie2rating = pickle.load(f)

with open(path+'usermovie2rating_test.json', 'rb') as f:
  usermovie2rating_test = pickle.load(f)


N = np.max(list(user2movie.keys())) + 1
# the test set may contain movies the train set doesn't have data on
m1 = np.max(list(movie2user.keys()))
m2 = np.max([m for (u, m), r in usermovie2rating_test.items()])
M = max(m1, m2) + 1
print("N:", N, "M:", M)


# initialize variables
K = 10 # latent dimensionality
W = np.random.randn(N, K)
b = np.zeros(N)
U = np.random.randn(M, K)
c = np.zeros(M)
mu = np.mean(list(usermovie2rating.values()))

# prediction[i,j] = W[i].dot(U[j]) + b[i] + c.T[j] + mu


def calculate_user_parameters(user2movie, usermovie2rating, U, W, b, mu, c, reg):

  for i in range(len(user2movie)):
    part1 = np.eye(K) * reg
    part2 = np.zeros(K)
    i = int(i)
    bi = 0
    for j in user2movie[i]:
      j = int(j)
      part1 += np.outer(U[j], U[j])
      part2 += (usermovie2rating[(i, j)] - b[i] - c[j] - mu) * U[j]
      bi += (usermovie2rating[(i, j)] - W[i].dot(U[j]) - c[j] - mu)

    W[i] = np.linalg.solve(part1, part2)
    b[i] = bi / (len(user2movie[i]) + reg)

    if i % (len(user2movie)//10) == 0:
      print("i:", i, "N:", N)

  return W, b


def calculate_movie_parameters(movie2user, usermovie2rating, U, W, b, mu, c, reg):


  for j in range(len(movie2user)):
    j = int(j)
    part1 = np.eye(K) * reg
    part2 = np.zeros(K)
    cj = 0
    try:
      for i in movie2user[j]:
        i = int(i)
        part1 += np.outer(W[i], W[i])
        part2 += (usermovie2rating[(i, j)] - b[i] - c[j] - mu) * W[j]
        cj += (usermovie2rating[(i, j)] - W[i].dot(U[j]) - b[i] - mu)

        U[j] = np.linalg.solve(part1, part2)
        c[j] = cj / (len(movie2user[j]) + reg)

    except KeyError:
      # possible not to have any ratings for a movie
      pass

    if j % (len(movie2user) // 10) == 0:
      print("j:", j, "M:", M)

  return U, c


def get_loss(d, U, W, b, mu, c):
  # d: (user_id, movie_id) -> rating
  N = float(len(d))
  sse = 0
  for k, r in d.items():
    try:
      i, j = k
      i = int(i)
      j = int(j)

      p = W[i].dot(U[j]) + b[i] + c[j] + mu
      sse += (p - r)*(p - r)
    except:
      print(i, j)
  return sse / N


# train the parameters
epochs = 7
reg =20. # regularization penalty
train_losses = []
test_losses = []
for epoch in range(epochs):
  print("epoch:", epoch)
  epoch_start = datetime.now()

  # update W and b
  W, b = calculate_user_parameters(user2movie, usermovie2rating, U, W, b, mu, c, reg)

  # update U and c
  U, c = calculate_movie_parameters(movie2user, usermovie2rating, U, W, b, mu, c, reg)

  train_losses.append(get_loss(usermovie2rating, U, W, b, mu, c))
  # store test loss
  test_losses.append(get_loss(usermovie2rating_test, U, W, b, mu, c))
  print("train loss:", train_losses[-1])
  print("test loss:", test_losses[-1])


print("train losses:", train_losses)
print("test losses:", test_losses)

# plot losses
plt.plot(train_losses, label="train loss")
plt.plot(test_losses, label="test loss")
plt.legend()
plt.show()
