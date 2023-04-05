pip import pandas
pip import numpy
pip import matplotlib
pip import seaborn
pip import sklearn

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from wordcloud import WordCloud
import seaborn as sns

from sklearn.cluster import KMeans,DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans


prime_df = pd.read_csv('prime_titles.csv')
hulu_df=pd.read_csv('hulu_titles.csv')
netflix_df=pd.read_csv('netflix_titles.csv')
prime_creds=pd.read_csv('credits_prime.csv')
hulu_creds=pd.read_csv('credits_hulu.csv')
netflix_creds=pd.read_csv('credits_netflix.csv')

prime_df = (prime_df.assign(service="amazon"))
hulu_df = (hulu_df.assign(service="hulu"))
netflix_df=(netflix_df.assign(service="netflix"))
prime_df=(pd.merge(prime_df, prime_creds, on='id'))
hulu_df=(pd.merge(hulu_df, hulu_creds, on='id'))
netflix_df=(pd.merge(netflix_df, hulu_creds, on='id'))
df=pd.concat([prime_df,hulu_df,netflix_df],axis=0,join='inner')


df['actor'] = df.apply(lambda row: row['name'] if row['role'] == 'ACTOR' else None, axis=1)
df['director'] = df.apply(lambda row: row['name'] if row['role'] != 'ACTOR' else None, axis=1)

df = df.groupby('id', as_index=False).agg({
    'title': 'first',
    'type': 'first',
    'description': 'first',
    'release_year': 'first',
    'age_certification': 'first',
    'runtime': 'first',
    'genres': 'first',
    'production_countries': 'first',
    'seasons': 'first',
    'imdb_id': 'first',
    'imdb_score': 'first',
    'imdb_votes': 'first',
    'tmdb_popularity': 'first',
    'tmdb_score': 'first',
    'service': 'first',
    'actor': lambda x: list(x),
    'person_id': lambda x: list(x),
    'name': lambda x: list(x),
    'character': lambda x: list(x),
    'role': lambda x: list(x),
    'director': lambda x: list(x)
})



def extract_director(directors_list):
    for director in directors_list:
        if director is not None:
            if isinstance(director, dict):
                return director['name']
            else:
                return director
    return None

df['director'] = df['director'].apply(extract_director)



df['genres'] = df['genres'].str.strip("[]'")
genres_df = df['genres'].str.split(', ', expand=True)
genres_df.columns = [f'genre_{i}' for i in range(genres_df.shape[1])]
df = pd.concat([df, genres_df], axis=1)
df['genre_0'] = df['genre_0'].str.strip("'")
df['genre_1'] = df['genre_1'].str.strip("'")
df['genre_2'] = df['genre_2'].str.strip("'")
df['genre_3'] = df['genre_3'].str.strip("'")
df['genre_4'] = df['genre_4'].str.strip("'")
df['genre_5'] = df['genre_5'].str.strip("'")
df['genre_6'] = df['genre_6'].str.strip("'")
df['genre_7'] = df['genre_7'].str.strip("'")
df['genre_8'] = df['genre_8'].str.strip("'")


df.drop(['genres'], axis=1, inplace=True)

df.loc[df['type'] == 'MOVIE', ['seasons']] = 0
df['production_countries'] = df['production_countries'].str.replace(r"[", '').str.replace(r"'", '').str.replace(r"]", '')
df['lead_prod_country'] = df['production_countries'].str.split(',').str[0]

# Let's also add a number of countries, envolved in movie making, so that we save a little more data
df['prod_countries_cnt'] = df['production_countries'].str.split(',').str.len()
df.lead_prod_country = df.lead_prod_country.replace('', np.nan)


df['actor'] = df['actor'].apply(lambda x: ' '.join([a if a is not None else '' for a in x]))
v=df['imdb_votes']
R=df['imdb_score']
c=df['imdb_score'].mean()
m=df['imdb_votes'].quantile(0.9)
df['weighted_avg']=((R*v)+ (c*m))/(v+m)
df.drop(df.loc[((df['tmdb_score'].isna()) & (df['tmdb_popularity'].isna()))].index, 
                axis = 0, inplace = True)
((df.isnull().sum()/df.shape[0])*100).sort_values(ascending = False)

df[(df.type == 'SHOW') & (df.imdb_score >= 9) & (df.tmdb_score>=9)][['actor', 'imdb_score', 'tmdb_score', 'title']].reset_index().drop('index', axis = 1)

df.drop(['id', 'imdb_id','person_id','genre_3','genre_4','genre_5','genre_6','genre_7','genre_8','character','role'], axis=1, inplace=True)

df['index_column'] = df.index
features = ['title', 'type', 'genre_0', 'description','actor','director', 'imdb_score', 'tmdb_popularity', 'tmdb_score']
for feature in features:
    df[feature] = df[feature].fillna('')

def combined_features(row):
    return row['title']+" "+row['type']+" "+row['genre_0']+" "+row['description']+" "+row['actor']+" "+row['director']+" "+str(row['imdb_score'])+" "+str(row['tmdb_popularity'])+" "+str(row['tmdb_score'])
                                                                                                                    
df["combined_features"] = df.apply(combined_features, axis =1)

vectorizer = CountVectorizer()
matrix_transform = vectorizer.fit_transform(df["combined_features"])
cosine_similarity_rm = cosine_similarity(matrix_transform)


from flask import Flask, render_template, request
import matplotlib.pyplot as plt
from io import BytesIO
import base64


if __name__ == '__main__':
    app.run()

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the selected movie from the form
        movie_name = request.form['movie_name']
        # Call the get_recommendations function
        recommendations, plot = get_recommendations(movie_name, 10)
        # Encode the plot as a base64 string
        plot_str = base64.b64encode(plot.getvalue()).decode('utf-8')
        # Render the results template with the recommended movies and plot
        return render_template('result.html', movie=movie_name, recommendations=recommendations, plot_str=plot_str)
    else:
        # Render the home template with the form
        return render_template('home.html')

def get_recommendations(movie_name, num_recommendations):
    # Get the index of the movie in the dataframe
    idx = df[df['title'] == movie_name].index[0]
    # Get the cosine similarity scores for all movies
    similarity_scores = list(enumerate(cosine_similarity_rm [idx]))
    # Sort the scores in descending order
    sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    # Get the indices and scores of the top n recommendations
    top_scores = sorted_scores[1:num_recommendations+1]
    movie_indices = [i[0] for i in top_scores]
    scores = [i[1] for i in top_scores]
    # Get the titles of the recommended movies
    rec_movies = df['title'].iloc[movie_indices].tolist()[::-1]
    # Reverse the order of the scores list to match the order of the bars
    scores = scores[::-1]
    # Create a bar chart showing the similarity scores
    fig, ax = plt.subplots()
    ax.barh(rec_movies, scores)
    ax.set_xlabel('Similarity Score')
    ax.set_ylabel('Recommended Movies')
    ax.set_title('Top {} Recommendations for {}'.format(num_recommendations, movie_name))
    # Add the score value to the end of the bar
    for i, v in enumerate(scores):
        ax.text(v + 0.01, i, str(round(v, 2)), color='blue', fontweight='bold')
    # Save the plot as a BytesIO object
    plot = BytesIO()
    fig.savefig(plot, format='png')
    # Close the figure
    plt.close(fig)
    # Return the titles of the recommended movies and the plot
    return rec_movies, plot
