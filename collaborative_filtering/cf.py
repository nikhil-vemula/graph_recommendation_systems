'''
Collaborative filtering
Author: Nikhil Vemula
'''
import os
import pandas as pd
from collections import defaultdict
import argparse
import logging

logging.basicConfig(format='%(asctime)s: %(levelname)s - %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logging.getLogger().setLevel(logging.INFO)

def split_train_test():
    ratings_file = os.path.join('..', 'data', 'ml-latest-small', 'ratings.csv')
    logging.info('Reading ' + ratings_file)
    ratings_df = pd.read_csv(ratings_file)

    user_ids = set()
    movie_ids = set()

    for _, row in ratings_df.iterrows():
        user_ids.add(row['userId'])
        movie_ids.add(row['movieId'])
    
    all_df = pd.DataFrame({userId: {movieId: 0 for movieId in movie_ids} for userId in user_ids})

    for idx, row in ratings_df.iterrows():
        if idx % 100 == 0:
            logging.info(f'Processed {(idx/len(ratings_df) * 100)}')
        all_df.loc[row['movieId'], row['userId']] = row['rating']

    all_df.to_csv(os.path.join('..', 'data', 'ml-latest-small', 'movie2user.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', help='Specifies train or test', type=str, choices=['train', 'predict'])
    args = parser.parse_args()

    if args.mode == 'train':
        logging.info('Training')
        split_train_test()
    elif args.mode == 'predict':
        logging.info('Predict')