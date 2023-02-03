int_data=data.select_dtypes(include="int64")
# selects columns data that have type int64

# getting all the columns
my_cols = set(int_data.columns)
# removing the desired column
my_cols.remove('MSSubClass')
my_cols = list(my_cols)
int_data2 = int_data[my_cols]
# printing the modified dataframe
pd.DataFrame(int_data2)
# reorders the columns, dont use this

# Using DataFrame.drop
df.drop(df.columns[[1, 2]], axis=1, inplace=True)
# drop by Name
df1 = df1.drop(['B', 'C'], axis=1)
# Select the ones you want
df1 = df[['a','d']]
# use this to drop columns instead
