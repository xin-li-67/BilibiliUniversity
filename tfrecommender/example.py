# simple matrix facotrization model using the MovieLens 100K dataset with TFRS
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs

from typing import Dict, Text

# read the data
ratings = tfds.load('movielens/100k-ratings', split="train")
movies = tfds.load('movielens/100k-movies', split="train")

ratings = ratings.map(lambda x: {
    "movie_title": x["movie_title"],
    "user_id": x["user_id"]
})
movies = movies.map(lambda x: x["movie_title"])

# convert user ids and movie titles into integer indices for embedding layers
user_ids_vocab = tf.keras.layers.experimental.preprocessing.StringLookup(mask_token=None)
user_ids_vocab.adapt(ratings.map(lambda x: x["user_idd"]))
movie_titles_vocab = tf.keras.layers.experimental.preprocessing.StringLookup(mask_token=None)
movie_titles_vocab.adapt(movies)

# define the model
class MovieLensModel(tfrs.Model):
    def __init__(self,
                 user_model: tf.keras.Model,
                 movie_model: tf.keras.Model,
                 task: tfrs.tasks.Retrieval()):
        super().__init__()
    
        self.user_model = user_model
        self.movie_model = movie_model
        self.task = task

    # define the loss func
    def compute_loss(self,
                     features: Dict[Text, tf.Tensor],
                     training=False) -> tf.Tensor:
        user_embeddings = self.user_model(features["user_id"])
        movie_embeddings = self.movie_model(features["movie_title"])

        return self.task(user_embeddings, movie_embeddings)

# define the two models and the retrieval task
user_model = tf.keras.Sequential([user_ids_vocab, tf.keras.layers.Embedding(user_ids_vocab.vocab_size(), 64)])
movie_model = tf.keras.Sequential([movie_titles_vocab, tf.keras.layers.Embedding(movie_titles_vocab.vocab_size(), 64)])
task = tfrs.tasks.Retrieval(metrics=tfrs.metrics.FactorizedTopK(movies.batch(128).map(movie_model)))

# fit and evaluate
model = MovieLensModel(user_model, movie_model, task)
model.compile(optimizer=tf.keras.optimizes.Adagrad(0.5))

model.fit(ratings.batch(4096), epochs=3)

# use brute-force search to set up retrieval using the trained representations.
index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
index.index(movies.batch(100).map(model.movie_model), movies)

_, titiles = index(np.array(["42"]))
print(f"Top 3 recommendations for user 42: {titles[0, :3]}")