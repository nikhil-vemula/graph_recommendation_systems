'''
Collaborative filtering
Author: Nikhil Vemula
'''
import os
import sys
import pandas as pd
from collections import defaultdict
import argparse
import logging

formatter = logging.Formatter('%(asctime)s: %(levelname)s - %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
file_handler = logging.FileHandler('log.out')
file_handler.setFormatter(formatter)

logging.getLogger().setLevel(logging.INFO)
logging.getLogger().addHandler(console_handler)
logging.getLogger().addHandler(file_handler)


def split_train_test():
    movie2user_df_file = os.path.join(data_set, 'movie2user.csv')
    if not os.path.exists(movie2user_df_file):
        ratings_file = os.path.join(data_set, 'ratings.csv')
        logging.info('Reading ' + ratings_file)
        ratings_df = pd.read_csv(ratings_file)

        user_ids = set()
        movie_ids = set()

        for _, row in ratings_df.iterrows():
            user_ids.add(row['userId'])
            movie_ids.add(row['movieId'])
        
        all_df = pd.DataFrame({userId: {movieId: 0.0 for movieId in movie_ids} for userId in user_ids})

        for idx, row in ratings_df.iterrows():
            if idx % 10000 == 0:
                logging.info(f'Processed {(idx/len(ratings_df) * 100)}')
            all_df.loc[row['movieId'], row['userId']] = row['rating']

        all_df.to_csv(os.path.join(data_set, 'movie2user.csv'))
    
    movie2user = pd.read_csv(movie2user_df_file)
    logging.info('movie2user: ' + str(movie2user.shape))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', help='Specifies train or test', type=str, choices=['train', 'predict'])
    args = parser.parse_args()

    data_set = os.path.join('..', 'data', 'ml-latest-small')

    if not os.path.exists(data_set):
        logging.error('Download data set')
        exit(1)

    if args.mode == 'train':
        logging.info('Training')
        split_train_test()
    elif args.mode == 'predict':
        logging.info('Predict')