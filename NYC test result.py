import pandas as pd

schools_df = pd.read_csv(r"C:\Users\Administrator\OneDrive\Datacamp_csv\Python CSV Datasets\schools.csv")
# print(schools_df.head(3))
# print(schools_df.columns)

# What NYC schools have the best math results?
best_math_schools = schools_df[schools_df['average_math'] >= 0.8*800]
best_math_schools = best_math_schools[['school_name', 'average_math']].sort_values(by='average_math', ascending=False)
# print(best_math_schools.head(3))

# What are the top 10 performing schools based on the combined SAT scores?
schools_df['total_SAT'] = schools_df['average_math'] + schools_df['average_reading'] + schools_df['average_writing']
top_10_schools = schools_df[['school_name', 'total_SAT']].sort_values(by='total_SAT', ascending=False).head(10)
# print(top_10_schools)

# Which single borough has the largest standard deviation in the combined SAT score?
boroughs = schools_df.groupby('borough')['total_SAT'].agg(['count', 'mean', 'std']).round(2)
largest_std_dev = boroughs[boroughs['std'] == boroughs['std'].max()]

largest_std_dev = largest_std_dev.rename(columns={'count':'num_schools', 'mean':'average_SAT', 'std':'std_SAT'})
print(largest_std_dev.head())