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

def create_train_and_test():
    logging.info('train.txt, test.txt are not found. creating one.')
    ratings_df = pd.read_csv(os.path.join('..', 'data', 'ml-latest-small', 'ratings.csv'))

    user2item = defaultdict(list)
    items = set()
    for _, row in ratings_df.iterrows():
        user2item[row['userId']].append(row['movieId'])
        items.add(row['movieId'])

    print(len(items))
    
    user2item_df = pd.DataFrame({'user_id': user2item.keys(), 'item_ids': user2item.values()})


    train_data = user2item_df.sample(frac=0.8, random_state=2024)
    test_data = user2item_df[~user2item_df.index.isin(train_data.index)]

    user2item = defaultdict(list)
    for _, row in train_data.iterrows():
        user2item[row['user_id']].append(row['item_ids'])
    with open(os.path.join('..', 'data', 'ml-latest-small', 'train.txt'), 'w') as f:
        for userid, item_ids in user2item.items():
            f.write(' '.join(map(str, [userid] + item_ids)) + '\n')

    user2item = defaultdict(list)
    for _, row in test_data.iterrows():
        user2item[row['user_id']].append(row['item_ids'])
    with open(os.path.join('..', 'data', 'ml-latest-small', 'test.txt'), 'w') as f:
        for userid, item_ids in user2item.items():
            f.write(' '.join(map(str, [userid] + item_ids)) + '\n')

def get_train_data():
    users = set()
    items = set()
    user2item = defaultdict(list)
    item2user = defaultdict(list)

    if not os.path.exists(train_file):
        create_train_and_test()
    
    with open(train_file, 'r') as f:
        for line in f.readlines():
            userid, *itemids = line.split()
            users.add(userid)
            user2item[userid] = itemids
            for itemid in itemids:
                items.add(itemid)
                item2user[itemid].append(userid)
    
    return (users, items, user2item, item2user)

def get_test_data():
    users = set()
    items = set()
    user2item = defaultdict(list)
    item2user = defaultdict(list)

    if not os.path.exists(test_file):
        create_train_and_test()
    
    with open(test_file, 'r') as f:
        for line in f.readlines():
            userid, *itemids = line.split()
            users.add(userid)
            user2item[userid] = itemids
            for itemid in itemids:
                items.add(itemid)
                item2user[itemid].append(userid)
    
    return (users, items, user2item, item2user)

def train():
    train_users, train_items, train_user2item, train_item2user = get_train_data()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', help='Specifies train or test', type=str, choices=['train', 'predict'])
    args = parser.parse_args()

    train_file = os.path.join('..', 'data', 'ml-latest-small', 'train.txt')
    test_file = os.path.join('..', 'data', 'ml-latest-small', 'test.txt')

    if args.mode == 'train':
        logging.info('Training')
        train()
    elif args.mode == 'predict':
        logging.info('Predict')