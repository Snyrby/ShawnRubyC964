import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from string import ascii_letters


warnings.simplefilter('ignore')

pd.options.display.max_columns = None

''' This algorithm was retrieved from Kaggle.com'''

# Opens CSV files and stores them in a Pandas Dataframe
meta = pd.read_csv('./Data/movies_metadata.csv')
credits = pd.read_csv('./Data/credits.csv')
keywords = pd.read_csv('./Data/keywords.csv')
links_small = pd.read_csv('./Data/links_small.csv')

# drops all duplicate titles entries from the data frame
meta = meta.drop_duplicates(subset='title', keep="first")

# Cleaning the Dataset of null values
meta['genres'] = meta['genres'].fillna('[]').apply(literal_eval).apply(
    lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
meta['production_companies'] = meta['production_companies'].fillna('[]').apply(literal_eval).apply(
    lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
meta['production_countries'] = meta['production_countries'].fillna('[]').apply(literal_eval).apply(
    lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
meta['spoken_languages'] = meta['spoken_languages'].fillna('[]').apply(literal_eval).apply(
    lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
meta['year'] = pd.to_datetime(meta['release_date'], errors='coerce').apply(
    lambda x: str(x).split('-')[0] if x != np.nan else np.nan)

# creates a variable for all vote counts that are not null and assigns them int values
vote_counts = meta[meta['vote_count'].notnull()]['vote_count'].astype('int')

# creates a variable for all vote averages  that are not null and assigns them int values
vote_averages = meta[meta['vote_average'].notnull()]['vote_average'].astype('int')

# calculates the average of all the averages for the weighted rating calculation
vote_mean = vote_averages.mean()

# confirms that only the top .96 of the vote counts will be used
vote_filter = vote_counts.quantile(0.96)


# this function will calculate the weighted rating for the movies
def weighted_rating(x):
    vote_count = x['vote_count']
    vote_average = x['vote_average']
    return (vote_count / (vote_count + vote_filter) * vote_average) + (vote_filter / (vote_filter + vote_count) *
                                                                       vote_mean)


# creates a variable for all tmdbID that are not null and assigns them int values
links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')

# assigns each movie id an int value
meta['id'] = meta['id'].astype('int')

# creates data frame for the movie information we will use. It will fill the data frame with all movies that are in the
# links small CSV
movie_frame = meta[meta['id'].isin(links_small)]

# replaces all null values in tagline column, so they don't mess up algorithm
movie_frame['tagline'] = movie_frame['tagline'].fillna('')

# creates a description column and fills it with the tagline column and overview column
movie_frame['description'] = movie_frame['overview'] + movie_frame['tagline']

# replaces all null values in the description column
movie_frame['description'] = movie_frame['description'].fillna('')

# assigns the id of keywords, credits integer values and will shape them to form an array
keywords['id'] = keywords['id'].astype('int')
credits['id'] = credits['id'].astype('int')
meta.shape

# links the 3 data sets together on the id value of the movie
meta = meta.merge(credits, on='id')
meta = meta.merge(keywords, on='id')

# creates data frame for the movie information we will use. It will fill the data frame with all movies that are in the
# links small CSV
movie_frame = meta[meta['id'].isin(links_small)]

# takes the cast, crew, and keywords and splits it into dictionary objects for better referencing
movie_frame['cast'] = movie_frame['cast'].apply(literal_eval)
movie_frame['crew'] = movie_frame['crew'].apply(literal_eval)
movie_frame['keywords'] = movie_frame['keywords'].apply(literal_eval)

# creates a cast size and crew size column that will have the total number of cast and crew for each movie
movie_frame['cast_size'] = movie_frame['cast'].apply(lambda x: len(x))
movie_frame['crew_size'] = movie_frame['crew'].apply(lambda x: len(x))


# function that will take in each crew dictionary entry and if the job is a director, it will add it to a director
# column
def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan


# creates a column only for directors for each movie
movie_frame['director'] = movie_frame['crew'].apply(get_director)

# converts the cast column to only have the names of each cast member and only shows the top 4 cast members
movie_frame['cast'] = movie_frame['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
movie_frame['cast'] = movie_frame['cast'].apply(lambda x: x[:4] if len(x) >= 4 else x)

# converts the keywords column to only apply what is associated with the names of each keyword
movie_frame['keywords'] = movie_frame['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

# will remove all spaces for each cast and director columns
movie_frame['cast'] = movie_frame['cast'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])
movie_frame['director'] = movie_frame['director'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))

# applys the weight of the director to 3x for the recommendation calculation
movie_frame['director'] = movie_frame['director'].apply(lambda x: [x, x, x])

# this will split the keywords
s = movie_frame.apply(lambda x: pd.Series(x['keywords']), axis=1).stack().reset_index(level=1, drop=True)
s.name = 'keyword'
s = s.value_counts()
s = s[s > 1]

# this will create a stemmer which for example lovely and loving will convert to love
stemmer = SnowballStemmer('english')


# function that filters each keyword and checks to see if it's individual list and returns the list
def filter_keywords(x):
    words = []
    for i in x:
        if i in s:
            words.append(i)
    return words


# converts each keyword to filtered out, stems it, then removes all spaces
movie_frame['keywords'] = movie_frame['keywords'].apply(filter_keywords)
movie_frame['keywords'] = movie_frame['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
movie_frame['keywords'] = movie_frame['keywords'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])

# creates empty preferences list incase no selection is toggled
preferences = []

# creates a new column based on keywords, cast, director and genres for the case of recommendations
def get_preferences():
    # performs a series of calculations of adding each preference to the 'soup' column
    if len(preferences) == 1:
        movie_frame['soup'] = movie_frame[preferences[0]]
        movie_frame['soup'] = movie_frame['soup'].apply(lambda x: ' '.join(x))
    elif len(preferences) == 2:
        movie_frame['soup'] = movie_frame[preferences[0]] + movie_frame[preferences[1]]
        movie_frame['soup'] = movie_frame['soup'].apply(lambda x: ' '.join(x))
    elif len(preferences) == 3:
        movie_frame['soup'] = movie_frame[preferences[0]] + movie_frame[preferences[1]] + movie_frame[preferences[2]]
        movie_frame['soup'] = movie_frame['soup'].apply(lambda x: ' '.join(x))
    elif len(preferences) == 4:
        movie_frame['soup'] = movie_frame[preferences[0]] + movie_frame[preferences[1]] + movie_frame[preferences[2]] \
                              + movie_frame[preferences[3]]
        movie_frame['soup'] = movie_frame['soup'].apply(lambda x: ' '.join(x))
    elif not preferences:
        movie_frame['soup'] = movie_frame['title']


# retrieves the preference list from checkboxes in GUI
def get_preferences_list(x):
    preferences.clear()
    for i in x:
        preferences.append(i)


# function that will calculate the recommendations based on a movie title
def improved_recommendations(title, movie_frame):
    # runs the get preferences function
    get_preferences()

    # runs vectorization and counts how many similarities based on a movie title
    count = CountVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
    count_matrix = count.fit_transform(movie_frame['soup'])
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    movie_frame = movie_frame.reset_index()
    indices = pd.Series(movie_frame.index, index=movie_frame['title'])

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    movie_indices = [i[0] for i in sim_scores]

    movies = movie_frame.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year']]
    vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')
    vote_filter = vote_counts.quantile(0.50)
    qualified_list = movies[
        (movies['vote_count'] >= vote_filter) & (movies['vote_count'].notnull()) & (movies['vote_average'].notnull())]
    qualified_list['vote_count'] = qualified_list['vote_count'].astype('int')
    qualified_list['vote_average'] = qualified_list['vote_average'].astype('int')
    qualified_list['wr'] = qualified_list.apply(weighted_rating, axis=1)
    qualified_list = qualified_list.sort_values('wr', ascending=False).head(10)
    return qualified_list


# function that will get the movie recommendations and create a plot graph
def get_data_and_plot(name):
    try:
        # will loop through every movie in the database
        for i in meta['title']:
            # this function will compare the strings it will remove all spaces, all punctuation and converts it to lower
            if ''.join([letter for letter in name if letter in ascii_letters]).lower() == ''.join([letter for letter in
                                                                                                   i if letter in
                                                                                                        ascii_letters
                                                                                                   ]).lower():
                # if it exists, it will change the format of the inputted movie title to what's stored for the algorithm
                name = i
                # prints out the recommendations to the console
                print(improved_recommendations(name, movie_frame))
                # collects the list of titles from the algorithm
                titles = improved_recommendations(name, movie_frame)['title']
                # collects the list of ratings from the algorithm
                rating = improved_recommendations(name, movie_frame)['wr']
                # plots the results and will show the plot chart
                plt.plot(titles, rating, marker='o')
                plt.title('Movies related to ' + name + ': ')
                plt.show()
    except:
        print('Movie does not exist')


# function that will get the movie recommendations and create a bar chart
def get_data_and_bar(name):
    try:
        # will loop through every movie in the database
        for i in meta['title']:
            # this function will compare the strings it will remove all spaces, all punctuation and converts it to lower
            if ''.join([letter for letter in name if letter in ascii_letters]).lower() == ''.join([letter for letter in
                                                                                                   i if letter in
                                                                                                        ascii_letters
                                                                                                   ]).lower():
                # if it exists, it will change the format of the inputted movie title to what's stored for the algorithm
                name = i
                # prints out the recommendations to the console
                print(improved_recommendations(name, movie_frame))
                # collects the list of titles from the algorithm
                titles = improved_recommendations(name, movie_frame)['title']
                # collects the list of ratings from the algorithm
                rating = improved_recommendations(name, movie_frame)['wr']

                # Figure Size
                fig, ax = plt.subplots(figsize=(16, 9))

                # Horizontal Bar Plot
                ax.barh(titles, rating)

                # Remove axes splines
                for s in ['top', 'bottom', 'left', 'right']:
                    ax.spines[s].set_visible(False)

                # Remove x, y Ticks
                ax.xaxis.set_ticks_position('none')
                ax.yaxis.set_ticks_position('none')

                # Add padding between axes and labels
                ax.xaxis.set_tick_params(pad=5)
                ax.yaxis.set_tick_params(pad=10)

                # Add x, y gridlines
                ax.grid(b=True, color='grey',
                        linestyle='-.', linewidth=0.5,
                        alpha=0.2)

                # Show top values
                ax.invert_yaxis()

                # Add annotation to bars
                for i in ax.patches:
                    plt.text(i.get_width() + 0.2, i.get_y() + 0.5,
                             str(round((i.get_width()), 2)),
                             fontsize=10, fontweight='bold',
                             color='grey')

                # Add Plot Title
                ax.set_title('Movies related to ' + name + ': ',
                             loc='left', )

                # Show bar chart
                plt.show()
    except:
        print('Movie does not exist')


# function that will get the movie recommendations and create a pie graph
def get_data_and_pie(name):
    try:
        # will loop through every movie in the database
        for i in meta['title']:
            # this function will compare the strings it will remove all spaces, all punctuation and converts it to lower
            if ''.join([letter for letter in name if letter in ascii_letters]).lower() == ''.join([letter for letter in
                                                                                                   i if letter in
                                                                                                        ascii_letters
                                                                                                   ]).lower():
                # if it exists, it will change the format of the inputted movie title to what's stored for the algorithm
                name = i
                # prints out the recommendations to the console
                print(improved_recommendations(name, movie_frame))
                # collects the list of titles from the algorithm
                titles = improved_recommendations(name, movie_frame)['title']
                # collects the list of ratings from the algorithm
                rating = improved_recommendations(name, movie_frame)['wr']
                # creates the pie chart and shows it
                fig = px.pie(rating, values=rating, names=titles)
                fig.show()
    except:
        print('Movie does not exist')
