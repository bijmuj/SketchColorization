import pandas as pd
import os
import requests
import random
import re
import json
import glob
from time import sleep
from multiprocessing.pool import ThreadPool
from paths_safebooru import tags_path, data_path, csv_path, img_path

samples = 15000


def download_url(url):
    file_name_start_pos = url.rfind("/") + 1
    file_name = img_path + url[file_name_start_pos:]
    if not os.path.isfile(file_name):
        while True:
            try:
                r = requests.get('http:' + url, stream=True)
                if r.status_code == requests.codes.ok:
                    with open(file_name, 'wb') as f:
                        for data in r:
                            f.write(data)
                break
            except requests.exceptions.ConnectionError:
                sleep(10)
    return file_name


def download_images(df):
    pd.options.mode.chained_assignment = None

    result = ThreadPool(8).imap_unordered(download_url, df.sample_url)
    total = 0
    pct = 0
    for r in result:
        total += 1
        if total % int(0.05*df_raw.shape[0]) == 0:
            pct += 5
            print(pct, "% downloaded")
    return


def delete_missing(df, paths):
    df = df.set_index('img_path')
    print(df.head(2))
    for path in paths:
        if not os.path.isfile(path):
            df = df.drop(path, axis=0)
    return df


def split_tokens(token):
    tokens = re.split("[ ]", token)
    list = []
    for t in tokens:
        list.append(t)
    return list


def get_data():
    if os.path.isfile(data_path):
        data = pd.read_json(data_path)
    else:
        data = create_data_files()
    return data


def get_tags(df):
    with open(tags_path, 'r') as infile:
        data = json.load(infile)
    return data['tags_list']


def get_img_paths(df):
    img_paths = []
    for i in df.sample_url:
        idx = i.rfind("/") + 1
        img_paths.append(img_path + i[idx:])
    return img_paths


def tags_to_labels(df):
    tags_list = get_tags(df)
    labels = [[0 for _ in tags_list] for _ in range(samples)]
    for i, tags in enumerate(df.tags):
        tokens = split_tokens(tags)
        for j, t in enumerate(tags_list):
            for token in tokens:
                if token in t:
                    labels[i][j] = 1
    df['labels'] = labels
    df.drop(['tags'], axis=1, inplace=True)
    return df


def create_data_files():
    df_raw = pd.read_csv(csv_path, nrows=samples)
    df_raw['img_path'] = get_img_paths(df_raw)
    features = ['img_path', 'tags']
    df_x = df_raw[features]
    df_x = tags_to_labels(df_x)
    df_x = delete_missing(df_x, df_x['img_path'])
    df_x.sort_values(by=['img_path'], inplace=True)
    df_x.to_json(data_path)
    return df_x


# create_data_files()
