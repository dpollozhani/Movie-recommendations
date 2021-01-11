import streamlit as st
from recommendation import generate_recommendations
import random 

def make_clickable_title(df):
    text, link = df['Title'], df['IMDB page']
    return  f'<a target="_blank" href="{link}">{text}</a>'

def make_clickable_link(link):
    text = link.split('//')[1]
    return  f'<a target="_blank" href="{link}">{text}</a>'

@st.cache
def engine(title_type='movie'):
    
    df = generate_recommendations(title_type)
    
    df['Title'] = df.apply(make_clickable_title, axis=1)
    df.drop(columns=['IMDB page'], inplace=True)
    df['Where to watch'] = df['Where to watch'].apply(make_clickable_link)
    
    return df
    
st.title('Pana$onic 2001 recommendations')
#st.subheader("It's cached and super fast!")
selected_type = st.radio(
    'Select title type',
    ('Movies', 'Series')
)

selection_map = {'Movies': 'movie',
                'Series': 'tvSeries'}
title_type = selection_map[selected_type]

df = engine(title_type)
df = df.to_html(escape=False, index=False)

st.write(df, unsafe_allow_html=True)


