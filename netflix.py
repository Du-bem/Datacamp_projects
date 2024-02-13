import pandas as pd
import matplotlib.pyplot as plt

netflix_df = pd.read_csv(r"C:\Users\Administrator\OneDrive\Datacamp_csv\Python CSV Datasets\netflix_data.csv")
netflix_subset = netflix_df[netflix_df['type'] == 'Movie']
# print(netflix_subset.head(3))

netflix_movies = netflix_subset[['title', 'country', 'genre', 'release_year', 'duration']]
short_movies = netflix_movies[netflix_movies['duration'] < 60]
# print(short_movies.head(3))

colours = []
for lab, row in netflix_movies.iterrows():
    if row['genre'] == 'Children':
        colours.append('Red')
    elif row['genre'] == 'Documentaries':
        colours.append('Green')
    elif row['genre'] == 'Stand-Up':
        colours.append('Blue')
    else:
        colours.append('Black')
# print(colours.head(3))

fig = plt.figure(figsize=(12, 8))

plt.scatter(netflix_movies['release_year'], netflix_movies['duration'], c=colours)
plt.xlabel('Release year')
plt.ylabel('Duration (min)')
plt.title('Movie Duration by Year of Release')

plt.show()

answer = "Maybe. Further analyses required."
