'''
Collaborative filtering
Author: Nikhil Vemula

https://towardsdatascience.com/item-based-collaborative-filtering-in-python-91f747200fab
'''
import os
import sys
import json
import pandas as pd
from collections import defaultdict
import argparse
import logging

from sklearn.neighbors import NearestNeighbors

formatter = logging.Formatter('%(asctime)s: %(levelname)s - %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
file_handler = logging.FileHandler('log.out')
file_handler.setFormatter(formatter)

logging.getLogger().setLevel(logging.INFO)
logging.getLogger().addHandler(console_handler)
logging.getLogger().addHandler(file_handler)


def split_train_test():
    train_file = os.path.join(data_set, 'movie2user_train.csv')
    train_file_exists = os.path.exists(train_file)
    test_file = os.path.join(data_set, 'movie2user_test.csv')
    test_file_exists = os.path.exists(test_file)

    if not train_file_exists or not test_file_exists:
        movie2user_df_file = os.path.join(data_set, 'movie2user.csv')
        ratings_file = os.path.join(data_set, 'ratings.csv')
        logging.info('Splitting train and test ' + ratings_file)
        ratings_df = pd.read_csv(ratings_file)

        user_ids = set()
        movie_ids = set()

        for _, row in ratings_df.iterrows():
            user_ids.add(row['userId'])
            movie_ids.add(row['movieId'])
        
        movie2user = pd.DataFrame({userId: {movieId: 0.0 for movieId in movie_ids} for userId in user_ids})

        for idx, row in ratings_df.iterrows():
            if idx % 10000 == 0:
                logging.info(f'Processed {(idx/len(ratings_df) * 100)}')
            movie2user.loc[row['movieId'], row['userId']] = row['rating']

        user2movie = movie2user.T
        logging.info('user2movie: ' + str(user2movie.shape))

        train = user2movie.sample(frac=0.8, random_state=2024)
        test = user2movie[~user2movie.index.isin(train.index)]

        logging.info('Train: ' + str(train.T.shape))
        logging.info('Test: ' + str(test.T.shape))

        train.T.to_csv(train_file)
        test.T.to_csv(test_file)
    
    train = pd.read_csv(train_file, index_col=0, header=0)
    test = pd.read_csv(test_file, index_col=0, header=0)

    return train, test

def train(train_df):
    '''
    train_df is [movies x users]
    Each movie(row) is represented by the ratings given the all the users. Movie embedding.
    We can find similar movies using these embeddings
    '''
    logging.info('Training: ' + str(train_df.shape))
    top_k = 5

    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(train_df.values)
    distances, indices = knn.kneighbors(train_df.values, n_neighbors=top_k)

    movie_2_similar_movies = {}
    for idx, movieId in enumerate(train_df.index.values):
        similar_movies = indices[idx].tolist()
        movie_dist = distances[idx].tolist()
        # Remove current movie from similar movies
        if movieId in similar_movies:
            movie_idx = similar_movies.index(movieId)
            similar_movies.remove(movieId)
            movie_dist.pop(movie_idx)
        movie_similarity = [1 - x for x in movie_dist]
        movie_2_similar_movies[movieId] = list(zip(similar_movies, movie_similarity))

    movie_2_similar_movies_file = os.path.join(data_set, 'movie_2_similar_movies.json')
    with open(movie_2_similar_movies_file, 'w') as f:
        json.dump(movie_2_similar_movies, f)

def get_predicted_rating(movie_2_similar_movies, user_movie_ratings, movieId):
    similar_movies = movie_2_similar_movies[str(movieId)]

    numerator = 0
    denominator = 0

    i = 0
    for sm in similar_movies:
        similar_movie_id = float(sm[0])
        similar_movie_similarity = float(sm[1])

        # user rated similar movie
        if similar_movie_id in user_movie_ratings:
            numerator += similar_movie_similarity * user_movie_ratings[similar_movie_id]
            denominator += similar_movie_similarity
    
    if numerator == denominator == 0:
        return 0

    return numerator / denominator

def get_recommended_movies_for_user(user):
    user = str(user)

    movie_2_similar_movies_file = os.path.join(data_set, 'movie_2_similar_movies.json')
    with open(movie_2_similar_movies_file, 'r') as f:
        movie_2_similar_movies = json.load(f)

    user_movie_ratings = {}
    for movieId, row in test_df.loc[:, [user]].iterrows():
        rating = row[user]
        if rating > 0:
            user_movie_ratings[movieId] = rating

    recommended_movies = []
    for movieId in test_df.index.values:
        if movieId not in user_movie_ratings:
            predicted_rating = get_predicted_rating(movie_2_similar_movies, user_movie_ratings, movieId)
            recommended_movies.append((movieId, predicted_rating))
    
    top_k = 5
    return sorted(recommended_movies, key=lambda x: x[1], reverse=True)[:top_k]
            

def test(test_df):
    '''
    for each user we calculate rmse between actual ratings and predicted rating
    '''
    movie_2_similar_movies_file = os.path.join(data_set, 'movie_2_similar_movies.json')
    with open(movie_2_similar_movies_file, 'r') as f:
        movie_2_similar_movies = json.load(f)

    user2movie = test_df.T

    error = []

    i = 0
    for userId in user2movie.index.values:
        i += 1
        if i % 10 == 0:
            logging.info(f'Processed: {i / len(user2movie)}')
        user_movie_ratings = {}
        for movieId in user2movie.columns:
            rating = user2movie.loc[userId, movieId]
            if rating > 0:
                user_movie_ratings[movieId] = rating
        
        for movieId in user_movie_ratings:
            actual = user_movie_ratings[movieId]
            predicted = get_predicted_rating(movie_2_similar_movies, user_movie_ratings, movieId)
            error.append(actual - predicted)

    se = [x ** 2 for x in error]
    mse = sum(se) / len(se)
    rmse = mse ** 0.5
    logging.info(f'RMSE {rmse}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', help='Specifies train or test', type=str, choices=['train', 'test'])
    args = parser.parse_args()

    data_set = os.path.join('..', 'data', 'ml-latest-small')

    if not os.path.exists(data_set):
        logging.error('Download data set')
        exit(1)
    
    train_df, test_df = split_train_test()

    if args.mode == 'train':
        logging.info('Training the model')
        train(train_df)

    elif args.mode == 'test':
        logging.info('Testing the model')
        test(test_df)