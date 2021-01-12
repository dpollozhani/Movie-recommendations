import streamlit as st
from recommendation import generate_recommendations
import random
#from pyngrok import ngrok

#public_url = ngrok.connect('8501')
#print(public_url)

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

title = '''
    <div style ="background-color:white;padding:10px">
    <h1 style ="color:black;text-align:center;">Pana$onic 2001 recommendations</h1>
    <p style ="background-color:#b991cf;color:orange;font-style:oblique;text-align:center;">Originalni Ohridski Biseri</p>
    </div> 
'''

st.markdown(title, unsafe_allow_html=True)

selected_type = st.radio(
    'Select title type',
    ('Movies', 'Series')
)

selection_map = {'Movies': 'movie',
                'Series': 'tvSeries'}
title_type = selection_map[selected_type]

run = st.button('Get recommendations')

if run:
    with st.spinner('Take a sip of your macchiato while this remarkable beast of an engine does its work.'):
        #seed = random.randint(1,50)
        df = engine(title_type)
        df = df.to_html(escape=False, index=False)

        st.write(df, unsafe_allow_html=True)

