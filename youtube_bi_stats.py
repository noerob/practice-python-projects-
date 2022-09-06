# -*- coding: utf-8 -*-
"""youtube bi stats'

# Commented out IPython magic to ensure Python compatibility.
#initial imports and formatting
import pandas as pd
import numpy as np
import json
import seaborn as sns
from os import path
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
# %matplotlib inline
from nltk.corpus import stopwords
#import dataframe
ogvideos_df = pd.read_csv('US_youtube_trending_data.csv', index_col='video_id')

#convert dates to readable formats
#videos_df['trending_date'] = pd.to_datetime(videos_df['trending_date'], format='%Y-%m-%dT%H:%M:%SZ')
#videos_df['publishedAt'] = pd.to_datetime(videos_df['publishedAt'], format='%Y-%m-%dT%H:%M:%SZ')
#videos_df.insert(2,'publish_date',videos_df['publishedAt'].dt.date)
#videos_df.insert(3,'publish_time',videos_df['publishedAt'].dt.time)

#format categoryIds into readable categories using json file
ogvideos_df['categoryId'] = ogvideos_df['categoryId'].astype(str)
categories_df = pd.read_json('US_category_id.json')
categorydict = {}
for category in categories_df['items']:
    categorydict[category['id']] = category['snippet']['title']
ogvideos_df.insert(4, 'category', ogvideos_df['categoryId'].map(categorydict))

ogvideos_df.head()

#data cleanup 

print("Original Shape:\t\t\t", ogvideos_df.shape,"\n")

#remove unneeded columns
unneededColumns = ['channelId','thumbnail_link','comments_disabled','ratings_disabled', 'description']
videos_df = ogvideos_df.drop(unneededColumns, axis=1)
print("Shape with columns removed:\t", videos_df.shape,"\n")

#remove duplicates (this doesnt remove the same videos that trended on different days)
videos_df = videos_df.drop_duplicates(keep = "first")
videos_df = videos_df.dropna()
print("Shape with duplicates removed:\t", videos_df.shape,"\n")

# display first five sample data pieces
videos_df.head()

#this section of code groups the category column 
#and sums the values of other columns (view_count, likes, dislikes, comment_count)
df2 = videos_df.groupby('category').sum()
df2.reset_index(inplace=True)
df2

#summary of mathematical values (count/mean/std/min/etc)
videos_df.describe()

#summary of count, number of unique videos, top videos, freq
videos_df.describe(include=['O'])

#some interesting/useful  video stats

#number of videos with more comments than likes (controversial, encouraging discussion, often politics)
print(videos_df.query('comment_count > likes').shape)

#number of videos with more dislikes than likes (controversial, moreso silly or reality tv)
print(videos_df.query('dislikes > (likes)').shape)

#top trending videos by views, likes, and comments
dataframeShow = input("Sort trending videos by views (1), likes (2), or comments (3)?")
if dataframeShow =='1':
  df1=videos_df.sort_values('view_count',ascending=False)
  print(df1.head())
elif dataframeShow =='2':
  df2=videos_df.sort_values('likes',ascending=False)
  print(df2.head())
elif dataframeShow =='3':
  df3=videos_df.sort_values('comment_count',ascending=False)
  print(df3.head())
else:
  print("incorrect input")

#correlation between likes, dislikes, and viewcount
#strong correlations in general, but strongest between likes and views
cols = ['view_count', 'likes', 'dislikes']
cor = videos_df[cols].corr()
print(cor)

# Commented out IPython magic to ensure Python compatibility.
#Visualization 1
import pandas as pd
import numpy as np
import warnings
import regex as re
warnings.filterwarnings('ignore')#to filter all the warnings
import seaborn as sns
from os import path
pd.set_option('float_format', '{:.4f}'.format)# to keep the float values short
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
# %matplotlib inline
from nltk.corpus import stopwords

stopwords = set(STOPWORDS)
def generate_wordcloud(text, words):
    wordcloud = WordCloud(stopwords=words, background_color="white").generate(text)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()
    
videos_df['trending_date'] = pd.to_datetime(videos_df['trending_date'], format='%Y-%m-%dT%H:%M:%SZ')
videos_df['publishedAt'] = pd.to_datetime(videos_df['publishedAt'], format='%Y-%m-%dT%H:%M:%SZ')
#print(list(categorydict.values()))
#for i in list(categorydict.values()):
#  print (i)
print(videos_df['category'].unique())
category = input("\nWhich category? Choose from the above: ")
timeyear = input("Starting from which year? Enter a number: ")
time = input("Starting from which month? Enter a number: ")
dateMask = (videos_df.trending_date > pd.Timestamp(int(timeyear),int(time),1)) & (videos_df.trending_date < pd.Timestamp(2022,7,1))
tag_text = " ".join(text for text in videos_df.tags[(videos_df.category == category ) & (dateMask)])
title = " ".join(text for text in videos_df.title[(videos_df.category == category ) & (dateMask) ])
tagtitle = tag_text + ' ' + title
generate_wordcloud(tagtitle, stopwords)

#*unfinished
#STACKED BAR GRAPH

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-small',
         'ytick.labelsize':'x-small'}
plt.rcParams.update(params)

cat = df2['category']
lik = df2['likes']
dis = df2['dislikes']

# plot bars in stack manner
plt.bar(cat, lik, color='c')
plt.bar(cat, dis, bottom=lik, color='m')
plt.title("Comparing the number of likes/dislikes for each category")
plt.xlabel("Category")
plt.ylabel("Comparison of likes and dislikes")
plt.show()
