import pandas as pd
import numpy as np
from datetime import datetime
import requests
import requests_cache
from bs4 import BeautifulSoup
from ast import literal_eval
from pprint import pprint
import dataframe_image as dfi
import pickle
import os
import sys

#### Helper functions ####       
def prettify(recommendations, no_of_cols=3):
    if not isinstance(recommendations ,str):
        first_cols = recommendations.columns[:min(recommendations.shape[1], no_of_cols)]
        return recommendations[first_cols].to_markdown(index=False, tablefmt='simple')
    return recommendations

def table_to_image(recommendations, no_of_cols=3):
    first_cols = recommendations.columns[:no_of_cols]
    recommendations = recommendations[first_cols]
    styled = recommendations.style.background_gradient(subset=recommendations.columns[-1])
    dfi.export(styled, 'data/recommendations/recommendations.png')

def load_model():
    model_path = f'similarity_models/similarities.npz'
    try:
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                archive = np.load(f)
                model = archive['arr_0']
                f.close()
            return model
    except OSError as e:
        print('OS error:', e)

def load_data():
    all_titles_path = 'data/prepared/all_titles.gz'
    own_titles_path = 'data/prepared/own_titles.gz'
    try:
        if os.path.exists(all_titles_path) and os.path.exists(own_titles_path):
            with open(all_titles_path, 'r') as f:
                all_titles = pd.read_csv(all_titles_path, na_values='Unknown', compression='gzip')
                f.close()
            with open(own_titles_path, 'r') as f:
                own_titles = pd.read_csv(own_titles_path, na_values='Unknown', compression='gzip')
                f.close()
            return all_titles, own_titles
    except OSError as e:
        print('OS error:', e)
                  
#### Main function ####
def generate_recommendations(title_type='movie', from_file=True, data=None, model=None, seed=datetime.today().isocalendar()[1], explanation=False):
    ''' Returns final recommendations 

    Draws random weighted sample from own titles and extracts most similar movies from all titles,
    (filtering on average rating and votes, and ignoring movies already watched/rated)
    For each sample the three most similar titles are selected, after which 10 titles are randomly drawn.
    
    '''
    if from_file:
        data = load_data()    
        similarities = load_model()
    elif len(data) == 2 and len(model) > 0:
        similarities = model
    else:
        print('Data and similarity model is missing')
        sys.exit()
    
    all_titles, own_titles = data[0], data[1]

    weights = own_titles['myRating']
    base_titles = own_titles[own_titles['titleType'] == title_type]
    base_sample = base_titles.sample(n=10, weights=weights, random_state=seed)
    sim_scores, score_map = [], {}
    for base_title in base_sample.index.values:
        tmp_scores1 = similarities[base_title]
        tmp_scores1 = list(enumerate(tmp_scores1))
        tmp_scores1 = sorted(tmp_scores1, key=lambda x: x[1], reverse=True)
        tmp_scores2 = []
        for i, sc in tmp_scores1:
            if (i, sc) not in score_map.keys() and i not in own_titles['originalIndex'].values:
                score_map[(i,sc)] = 1 
                rating = all_titles['averageRating'].iloc[i]
                votes = all_titles['numVotes'].iloc[i]
                ttype = all_titles['titleType'].iloc[i]
                min_rating = 6.9 if ttype == 'movie' else 8.0       
                if (rating > min_rating and votes > 5000 and ttype == title_type) or np.isnan(rating):      
                    tmp_scores2.append((i, sc))
            if len(tmp_scores2) == 3:
                break
        sim_scores.append(tmp_scores2)
    
    sim_scores = [score for scores in sim_scores for score in scores]
    
    #Getting the indices of 10 equi-weighted randomly chosen titles from the most (dis)similar titles selected above
    np.random.shuffle(sim_scores)
    movie_indices = [i[0] for i in sim_scores[:10]]
    
    #Present final recommendations in a tidy data frame
    recommendations = pd.DataFrame(all_titles[['primaryTitle', 'startYear', 'averageRating', 'numVotes']].iloc[movie_indices])
    recommendations = recommendations.rename(columns={'primaryTitle': 'Title','startYear': 'Release year', 'averageRating': 'IMDB rating', 'numVotes': 'Votes'})
    recommendations = recommendations.sort_values(by='IMDB rating', ascending=False)
    recommendations['IMDB rating'] = recommendations['IMDB rating'].apply(lambda x: round(x,2))
    recommendations['IMDB page'] = all_titles['tconst'].apply(lambda x: f'https://www.imdb.com/title/{x}/')
    recommendations.reset_index(drop=True, inplace=True)
    
    if explanation:
        text = '\n***\nRecommendations based on movies similar to below:\n***\n'
        based_on = own_titles.iloc[base_sample.index.values][['primaryTitle', 'myRating']].rename(columns={'primaryTitle':'Title', 'myRating': 'My rating'})
        return recommendations, text + prettify(based_on)
    
    return recommendations
    
if __name__ == '__main__':
    #pd.set_option('max_colwidth', None)
    #pd.set_option('max_columns', None)
    r = generate_recommendations(title_type='tvSeries', explanation=False)
    # print(r[0])
    # print(r[1])
    #if isinstance(r, pd.DataFrame):
    #    r.to_csv('data/recommendations/recommendation.csv')
    print(prettify(r, 4))
    #print(explained)