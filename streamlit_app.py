import streamlit as st
from datatransformation import load_files, merge_files, prepare_model
from recommendation import generate_recommendations
import random 

@st.cache(show_spinner=False)
def pre_engine():
    files = load_files()
    data = merge_files(files)
    similarities, all_titles, own_titles = prepare_model(data)
    
    return similarities, all_titles, own_titles

@st.cache
def engine(similarities, all_titles, own_titles, title_type='movie'):
    
    df = generate_recommendations(title_type, from_file=False, data=[all_titles,own_titles], model=similarities)
    
    df['Title'] = df.apply(make_clickable_title, axis=1)
    df.drop(columns=['IMDB page'], inplace=True)
    df['Where to watch'] = df['Where to watch'].apply(make_clickable_link)
    
    return df

def make_clickable_title(df):
    text, link = df['Title'], df['IMDB page']
    return  f'<a target="_blank" href="{link}">{text}</a>'

def make_clickable_link(link):
    text = link.split('//')[1]
    return  f'<a target="_blank" href="{link}">{text}</a>'
    
st.title('Pana$onic 2001 recommendations')
#st.subheader("It's cached and super fast!")
selected_type = st.radio(
    'Select title type',
    ('Movies', 'Series')
)

selection_map = {'Movies': 'movie',
                'Series': 'tvSeries'}
title_type = selection_map[selected_type]

similarities, all_titles, own_titles = pre_engine()
df = engine(similarities, all_titles, own_titles, title_type)
df = df.to_html(escape=False, index=False)

st.write(df, unsafe_allow_html=True)


