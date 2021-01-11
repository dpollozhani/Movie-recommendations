import pandas as pd
import numpy as np
from datetime import datetime
import sys
from sklearn.metrics.pairwise import cosine_similarity, pairwise_kernels
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer
import pickle

#### Helper functions for cleaning ####
def only_types(df, title_typess):
    return df[df.titleType.isin(title_typess)]

def drop_columns(df, cols):
    cols_to_keep = list(set(df.columns.values) - set(cols))
    return df[cols_to_keep]

def to_num(df, cols):
    for col in cols:
        df[col] = pd.to_numeric(df[col])
    return df

def to_categorical(df, cols):
    for col in cols:
        df[col] = df[col].astype('category')
    return df

def add_categories(df, cols, categories):
    for col in cols:
        if df[col].dtype == 'category':
            df[col] = df[col].cat.add_categories(categories)
    return df

def remove_missing(df, cols):
    for col in cols:
        df = df[df[col].notna()]
    return df

def sort(df, by, ascending=True):
    return df.sort_values(by=by, ascending=ascending)

#### Main cleaning functions ####
def load_files(title_types=['movie','tvSeries']):
    ''' Reading and filtering all available files as dataframes. Stores each frame to disk for later handling.
        Returns dictionary of all dataframes.
    '''

    title_basics_cols = ['tconst', 'titleType', 'primaryTitle', 'startYear', 'genres', 'runtimeMinutes', 'isAdult']
    
    print('title_basics << title basics.tsv.gz')
    title_basics = pd.read_csv('data/raw/title basics.tsv.gz', compression='gzip', sep='\t', usecols=title_basics_cols, dtype = {'startYear':'string', 'runtimeMinutes':'str'}, na_values=r'\N', engine='c')
    title_basics = title_basics[title_basics['isAdult'] == 0]
    title_basics = only_types(title_basics, title_types)
    title_basics = remove_missing(title_basics, ['runtimeMinutes', 'startYear', 'genres'])
    title_basics = drop_columns(title_basics, ['isAdult'])
    title_basics = to_num(title_basics, ['startYear', 'runtimeMinutes'])
    title_basics = to_categorical(title_basics, ['titleType', 'genres'])
    title_basics = title_basics.set_index('tconst')
    
    title_basics.to_csv('data/cleaned/title basics.csv')
                    
    print('title_ratings << title ratings.tsv.gz')
    title_ratings = pd.read_csv('data/raw/title ratings.tsv.gz', compression='gzip', sep='\t', na_values=r'\N', engine='c')
    title_ratings = title_ratings.set_index('tconst')

    title_ratings.to_csv('data/cleaned/title ratings.csv')
    
    print('title_principals << title principals.tsv.gz')
    title_principals = pd.read_csv('data/raw/title principals.tsv.gz', compression='gzip', sep='\t', usecols=['tconst', 'nconst'], na_values=r'\N', engine='c')
    existing_titles = title_basics.index.unique()
    title_principals = title_principals[title_principals['tconst'].isin(existing_titles)]
    title_principals = title_principals.set_index('nconst')

    title_principals.to_csv('data/cleaned/title principals.csv')
    
    print('name_basics << name basics.tsv.gz')
    name_basics = pd.read_csv('data/raw/name basics.tsv.gz', sep='\t', compression='gzip', usecols=['nconst', 'primaryName'], na_values=r'\N', engine='c')
    existing_names = title_principals.index.unique()
    name_basics = name_basics[name_basics['nconst'].isin(existing_names)]
    name_basics = name_basics.set_index('nconst')
    
    name_basics.to_csv('data/cleaned/name basics.csv')

    print('own_ratings << ratings.csv')
    own_ratings = pd.read_csv('data/raw/ratings.csv', encoding='Latin-1', usecols=['Const', 'Your Rating', 'Date Rated'])
    own_ratings = own_ratings.rename(columns={'Const': 'tconst', 'Your Rating': 'myRating', 'Date Rated': 'dateRated'})
    own_ratings = own_ratings.set_index('tconst')

    own_ratings.to_csv('data/cleaned/own ratings.csv')

    return {'title_basics': title_basics,
            'title_ratings': title_ratings,
            'title_principals': title_principals,
            'name_basics': name_basics,
            'own_ratings': own_ratings}

def merge_files(df_dict):
    ''' Merges dataframes returned by load_files. Stores resulting dataframe to disk. '''

    #Merging title_principals and name_basics to get crew of each title
    print('crew << title_principals + name_basics (merge)')
    crew = pd.merge(df_dict['title_principals'], df_dict['name_basics'], how='left', left_index=True, right_index=True)
    print('crew << crew (groupby)')
    crew = crew.fillna('Unknown') #If not made to string, aggregation below will not work
    crew = crew.groupby('tconst')['primaryName'].agg(lambda column: ','.join(column)).reset_index()
    crew = crew.rename(columns={'primaryName': 'crewList'})
    crew = crew.set_index('tconst') #Setting tconst to index since merging on index is faster
    
    #Merging title_basics with title_ratings and own_ratings to get basic data, overall ratings and own ratings
    print('titles << title_basics + title_ratings (merge)')
    titles = pd.merge(df_dict['title_basics'], df_dict['title_ratings'], how='left', left_index=True, right_index=True)
    
    print('titles << titles + own_ratings (merge)')
    own_ratings = df_dict['own_ratings']
    titles = pd.merge(titles, own_ratings, how='left', left_index=True, right_index=True)
    
    #Merging titles with crew data
    print('titles << titles + crew (merge)')
    titles = pd.merge(titles, crew, how='left', left_index=True, right_index=True)
    
    titles = sort(titles, by='startYear')
    #titles = add_categories(titles, ['genres'], 'Unknown')
    #titles = titles.fillna('Unknown')
    titles.reset_index(inplace=True)

    titles.to_csv('data/cleaned/imdb_metadata.csv', index=False)
    
    return titles

### Helper functions for model preparation ####
def decade_bucket(x):
    if x < 0:
        decade = 'Unreleased'
    elif x < 10:
        decade = 'Zero'
    elif x < 20:
        decade = 'One'
    elif x < 30:
        decade = 'Two'
    elif x < 40:
        decade = 'Three'
    elif x < 50:
        decade = 'Four'
    elif x < 60:
        decade = 'Five'
    else:
        decade = 'Six or more'
    return decade

def add_title_age(df, col='startYear'):
    df['ageInYears'] = df[col].apply(lambda x: datetime.today().year - x)
    df['decadesSinceRelease'] = df['ageInYears'].apply(decade_bucket)
    df.drop('ageInYears', axis=1, inplace=True)
    return df

def quantile_comparison(x, category_map):
    for q, c in category_map.items():
        if x > q[0] and x <= q[1]:
            return c

def add_length_category(df, col='runtimeMinutes'):
    quantile_values = df[col].quantile([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
    category_map = {(0,quantile_values.loc[0.01]): 'Extremely short',
                 (quantile_values.loc[0.01], quantile_values[0.1]): 'Very short',
                 (quantile_values.loc[0.1], quantile_values.loc[0.25]): 'Short',
                 (quantile_values.loc[0.25], quantile_values.loc[0.75]): 'Medium',
                 (quantile_values.loc[0.75], quantile_values.loc[0.9]): 'Long',
                 (quantile_values.loc[0.9], quantile_values.loc[0.99]): 'Very long',
                 (quantile_values.loc[0.99], 10**6): 'Extremely long'                
                }
    df['titleLength'] = df[col].apply(lambda x: quantile_comparison(x, category_map))
    return df

def string_preparation(x):
    if isinstance(x, str):
        return x.lower().replace(' ', '').replace(',', ' ')
    else:
        return ''

def sanitize_strings(df, cols):
    for col in cols:
        df[col] = df[col].apply(string_preparation)
    return df

def filter_rated_titles(df, filter_column, max_rows=500):
    if filter_column in df.columns:
        last_row = min(max_rows, df.shape[0])
        indices = df[filter_column].sort_values(ascending=False).iloc[:last_row].index
        return df[df.index.isin(indices)]
    else:
        return df

def soup(df, cols):
    df['soup'] = df[cols].apply(lambda x: ' '.join(x), axis=1)
    return df

def similarity(x, y, vectorizer='count', metric='cosine'):
        #y must be a superset of x (vocabulary vector from y will be used)
    try:
        assert set(y).issuperset(x)
    except AssertionError as e:
        print(e, "y must be a superset of x")

    vectorizers = {'count': CountVectorizer,
                   'tfidf': TfidfVectorizer}
    if vectorizer in vectorizers.keys():
        vectorizer = vectorizers[vectorizer]
    else:
        vectorizer = vectorizers['count']

    count = vectorizer(stop_words='english')
    model = count.fit(x)
    count_x = model.transform(x)
    count_y = model.transform(y)
    sim = pairwise_kernels(count_x, count_y, metric=metric) 

    return sim

#### Main preparation function ####
def prepare_model(df, cols_for_soup = ['genres', 'crewList', 'decadesSinceRelease', 'titleLength']):
    #Preparation and creation of metadata soup
    all_titles = df.copy()
    all_titles = add_title_age(all_titles)
    all_titles = add_length_category(all_titles)
    all_titles = sanitize_strings(all_titles, cols_for_soup)
    all_titles = soup(all_titles, cols_for_soup)
    all_titles.to_csv(f'data/prepared/all_titles.gz', index=False, compression='gzip')

    own_titles = all_titles[all_titles['myRating'].notna()].copy()
    own_titles = own_titles.reset_index()
    own_titles = own_titles.rename(columns={'index': 'originalIndex'}) 
    own_titles = filter_rated_titles(own_titles, 'dateRated')
    own_titles.to_csv('data/prepared/own_titles.gz', index=False, compression='gzip')

    #Calculate cosine similarity between own titles and all titles
    #Only comparing rated movies to all movies, to avoid overloading memory
    similarities = similarity(own_titles['soup'], all_titles['soup'])
    
    #Storing similarity matrix to a pickle file to enable quicker usage later on
    with open(f'similarity_models/similarities.npz', 'wb') as f:
        np.savez_compressed(f, similarities)
        f.close()
    
    return similarities, all_titles, own_titles

if __name__ == '__main__':
    #df = pd.read_csv('data/cleaned/movie_metadata.csv', na_values='Unknown')
    
    start_time = datetime.now()
    print('Script started at', start_time)
    
    print('Loading files...')
    dfs = load_files()
    
    elapsed_time = datetime.now()-start_time
    print(f'Elapsed time: {elapsed_time}')
 
    df = merge_files(dfs)
    
    print('Preparing model...')
    prepare_model(df)

    end_time = datetime.now()
    elapsed_time = end_time-start_time
    print('Script finished at', end_time)
    print(f'Elapsed time: {elapsed_time}')

