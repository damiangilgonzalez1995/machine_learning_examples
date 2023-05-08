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
from copy import deepcopy

# load in the data
import os



path = "../"
with open(path+'user2movie.json', 'rb') as f:
  user2movie = pickle.load(f)

with open(path+'movie2user.json', 'rb') as f:
  movie2user = pickle.load(f)

with open(path+'usermovie2rating.json', 'rb') as f:
  usermovie2rating = pickle.load(f)

with open(path+'usermovie2rating_test.json', 'rb') as f:
  usermovie2rating_test = pickle.load(f)


def get_loss(m2u, U, W, b, mu, c):
  # d: movie_id -> (user_ids, ratings)
  N = 0.
  sse = 0
  for j, (u_ids, r) in m2u.items():
    j = int(j)
    u_ids = [int(u) for u in u_ids]
    try:
      p = W[u_ids].dot(U[j]) + b[u_ids] + c[j] + mu
      delta = p - r
      sse += delta.dot(delta)
      N += len(r)
    except:
      print(i, j)

  return sse / N


N = np.max(list(user2movie.keys())) + 1
# the test set may contain movies the train set doesn't have data on
m1 = np.max(list(movie2user.keys()))
m2 = np.max([m for (u, m), r in usermovie2rating_test.items()])
M = max(m1, m2) + 1
print("N:", N, "M:", M)


# convert user2movie and movie2user to include ratings
print("converting...")
user2movierating = {}
for i, movies in user2movie.items():
    movies = [int(movie) for movie in movies]
    r = np.array([usermovie2rating[(i,j)] for j in movies])
    user2movierating[i] = (movies, r)
movie2userrating = {}
for j, users in movie2user.items():
  users = [int(user) for user in users]
  r = np.array([usermovie2rating[(i,j)] for i in users])
  movie2userrating[j] = (users, r)

# create a movie2user for test set, since we need it for loss
movie2userrating_test = {}
for (i, j), r in usermovie2rating_test.items():
  if j not in movie2userrating_test:
    movie2userrating_test[j] = [[i], [r]]
  else:
    movie2userrating_test[j][0].append(i)
    movie2userrating_test[j][1].append(r)
for j, (users, r) in movie2userrating_test.items():
  movie2userrating_test[j][1] = np.array(r)
print("conversion done")

# initialize variables
K = 10 # latent dimensionality
W = np.random.randn(N, K)
b = np.zeros(N)
U = np.random.randn(M, K)
c = np.zeros(M)
mu = np.mean(list(usermovie2rating.values()))

# train the parameters
epochs = 25
reg = 20. # regularization penalty
train_losses = []
test_losses = []

for epoch in range(epochs):
  print("epoch:", epoch)

  for i in range(N):
    i = int(i)
    bi = 0

    m_ids, r = user2movierating[i]

    matrix = U[m_ids].T.dot(U[m_ids]) + np.eye(K) * reg
    vector = (r - b[i] - c[m_ids] - mu).dot(U[m_ids])
    bi = (r - U[m_ids].dot(W[i]) - c[m_ids] - mu).sum()

    W[i] = np.linalg.solve(matrix, vector)
    b[i] = bi / (len(user2movie[i]) + mu)

    if i % (N//10) == 0:
      print("i:", i, "N:", N)

  for j in range(M):
    j = int(j)
    cj = 0

    u_ids, r = movie2userrating[j]

    matrix = W[u_ids].T.dot(W[u_ids]) + np.eye(K) * reg
    vector = (r - b[u_ids] - c[j] - mu).dot(W[u_ids])
    cj = (r - W[u_ids].dot(U[j]) - b[u_ids] - mu).sum()

    U[j] = np.linalg.solve(matrix, vector)
    c[j] = cj / (len(movie2user[j]) + mu)

    if j % (M//10) == 0:
      print("j:", j, "M:", M)


  train_losses.append(get_loss(movie2userrating, U, W, b, mu, c))

  # store test loss
  test_losses.append(get_loss(movie2userrating_test, U, W, b, mu, c))
  print("train loss:", train_losses[-1])
  print("test loss:", test_losses[-1])


print("train losses:", train_losses)
print("test losses:", test_losses)

# plot losses
plt.plot(train_losses, label="train loss")
plt.plot(test_losses, label="test loss")
plt.legend()
plt.show()
