import pandas as pd

df = pd.read_csv('../data/rating.csv')

df.userId = df.userId - 1

# create a mapping for movie ids
unique_movie_ids = set(df.movieId.values)
movie2idx = {}
count = 0
for movie_id in unique_movie_ids:
  movie2idx[movie_id] = count
  count += 1

# add them to the data frame
# takes awhile
df["movie_idx"] = df.movieId.apply(lambda x: movie2idx[x])
df = df.drop(columns=['timestamp'])

df.to_csv('../data/edited_rating.csv')