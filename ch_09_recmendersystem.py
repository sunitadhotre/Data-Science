import pandas as pd

# Get the data
column_names = ['user_id', 'item_id', 'rating', 'timestamp']

path = 'D:\SSD_Python01012023\\file.tsv'

df = pd.read_csv(path, sep='\t', names=column_names)

print(df.head())

# Check out all the movies and their respective IDs
movie_titles = pd.read_csv('D:/SSD_Python01012023/Movie_Id_Titles.csv')
print(movie_titles.head())

data = pd.merge(df, movie_titles, on='item_id')
print(data.head())

# Calculate mean rating of all movies
data.groupby('title')['rating'].mean().sort_values(ascending=False).head()

# Calculate count rating of all movies
data.groupby('title')['rating'].count().sort_values(ascending=False).head()

# creating dataframe with 'rating' count values
ratings = pd.DataFrame(data.groupby('title')['rating'].mean())

ratings['num of ratings'] = pd.DataFrame(data.groupby('title')['rating'].count())

print(ratings.head())

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('white')

# plot graph of 'num of ratings column'
plt.figure(figsize =(10, 4))

# plot graph of 'ratings' column
plt.figure(figsize =(10, 4))

# Sorting values according to
# the 'num of rating column'
moviemat = data.pivot_table(index ='user_id',
			columns ='title', values ='rating')

print(moviemat.head())

ratings.sort_values('num of ratings', ascending = False).head(10)


ratings['rating'].hist(bins = 70)


ratings['num of ratings'].hist(bins = 70)

# analysing correlation with similar movies
starwars_user_ratings = moviemat['Star Wars (1977)']
liarliar_user_ratings = moviemat['Liar Liar (1997)']

print(starwars_user_ratings.head())

# analysing correlation with similar movies
similar_to_starwars = moviemat.corrwith(starwars_user_ratings)
similar_to_liarliar = moviemat.corrwith(liarliar_user_ratings)

corr_starwars = pd.DataFrame(similar_to_starwars, columns =['Correlation'])
corr_starwars.dropna(inplace = True)

print(corr_starwars.head())

# Similar movies as of liarliar
corr_liarliar = pd.DataFrame(similar_to_liarliar, columns =['Correlation'])
corr_liarliar.dropna(inplace = True)

corr_liarliar = corr_liarliar.join(ratings['num of ratings'])
corr_liarliar[corr_liarliar['num of ratings']>100].sort_values('Correlation', ascending = False).head()

print(corr_liarliar.head())





