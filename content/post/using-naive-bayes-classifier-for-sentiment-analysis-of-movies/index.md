---
title: Using Naive Bayes Classifier For Sentiment Analysis of movies
subtitle: Sentiment Analysis of movies on IMDb data using Naive Bayes Classifier
date: 2021-11-30T18:01:41.629Z
draft: false
featured: false
image:
  filename: featured
  focal_point: Smart
  preview_only: false
---
#### Introduction and Objective

**Sentiment analysis** is a technique through which we can analyze a piece of text to determine the sentiment behind it, i.e. whether the given text conveys a positive, negative or a neutral sentiment. It is widely used to gauge the feedback and can be used in industries ranging from social media, shopping and movie reviews to election results and healthcare.

We are going to use sentiment analysis on the movie data. The link for the dataset on kaggle can be found in the links section below. We are going to use imdb dataset, but you can choose amazon or yelp if you want. To do the analysis, we are going to use **Naive Bayes Classifier**. The detailed problem statement can be found on the [Google Doc](https://docs.google.com/document/d/1bmCm9TXwqp5tX7lpg14NkaB3dBSg15cCC7ICxeB-vB4/edit) made by my professor [Dr. Deokgun Park](https://crystal.uta.edu/~park/), for the class CSE 5331 Data Mining.

**Naive Bayes Classifier** use Bayes' theorem with strong (naïve) independence assumptions between the features. Because of this they can be used with fairly high accuracy even when some of the data is missing or has unknown values. This makes them a good candidate for text classification.

#### Process

###### Import the Data

After importing the required libraries, our first step is going to import the data. We have three dataset options to choose from in the kaggle link - IMDb, Amazon or Yelp. We are going to select IMDb.

```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data= pd.read_csv('imdb_labelled.txt', names=['Reviews','Sentiment'], delimiter = '\t')
data.info()
```

We get the following output

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 748 entries, 0 to 747
Data columns (total 2 columns):
 #   Column     Non-Null Count  Dtype 
---  ------     --------------  ----- 
 0   Reviews    748 non-null    object
 1   Sentiment  748 non-null    int64 
dtypes: int64(1), object(1)
memory usage: 11.8+ KB
```

This shows that we have 748 rows in our dataset.

To view the data we can call the data object, which would print rows

```
data
```

<table>
<tr>
<th> </th>
<th>Reviews</th>
<th>Sentiment</th>
</tr>
<tr>
<td>0</td>
<td>A very, very, very slow-moving, aimless movie ...</td>
<td>0</td>
</tr>
<tr>
<td>1</td>
<td>Not sure who was more lost - the flat characte...</td>
<td>0</td>
</tr>
<tr>
<td>2</td>
<td>Attempting artiness with black & white and cle...</td>
<td>0</td>
</tr>
<tr>
<td>3</td>
<td>Very little music or anything to speak of.</td>
<td>0</td>
</tr>
<tr>
<td>4</td>
<td>The best scene in the movie was when Gerardo i...</td>
<td>1</td>
</tr>
<tr>
<td>...</td>
<td>...</td>
<td>...</td>
</tr>
<tr>
<td>743</td>
<td>I just got bored watching Jessice Lange take h...</td>
<td>0</td>
</tr>
<tr>
<td>744</td>
<td>Unfortunately, any virtue in this film's produ...</td>
<td>0</td>
</tr>
<tr>
<td>745</td>
<td>In a word, it is embarrassing.</td>
<td>0</td>
</tr>
<tr>
<td>746</td>
<td>Exceptionally bad!</td>
<td>0</td>
</tr>
<tr>
<td>747</td>
<td>All in all its an insult to one's intelligence...</td>
<td>0</td>
</tr>
</table>

```
748 rows × 2 columns
```

###### Split the data

Now that we have the data, we are going to split it into train, test, and development datasets.

We are going to divide the dataset into *train:development: test = 85:10:5* ratio.

<table>
<tr>
<th>DataSet</th>
<th>Percent</th>
<th>Number</th>
</tr>
<tr>
<td>Train</td>
<td>85</td>
<td>639</td>
</tr>
<tr>
<td>Development</td>
<td>10</td>
<td>75</td>
</tr>
<tr>
<td>Test</td>
<td>5</td>
<td>37</td>
</tr>
<tr>
<td><b>Total</b></td>
<td><b>100</b></td>
<td><b>748</b></td>
</tr>
</table>



To make it unbiased, we would shuffle the data before splitting.

```
# make a copy of the data
shuffle_data = data.copy(deep=True)
# shuffle data with sample().
# frac = 1 is entire dataset, random_state=1 for reproducible data
# and reset_index() to reset the index
shuffle_data = shuffle_data.sample(frac=1,
                  random_state=1).reset_index()
shuffle_data.head(10)
```



#### Links

1. [Movie Dataset on kaggle for sentiment analysis](https://www.kaggle.com/marklvl/sentiment-labelled-sentences-data-set)
2. [Our Jupyter Notebook with Code](https://github.com/skbt/Sentiment-analysis-using-Naive-Bayes/blob/main/Sentiment-Analysis-using-NBC.ipynb)

#### References

1. https://en.wikipedia.org/wiki/Sentiment_analysis
2. https://www.lexalytics.com/technology/sentiment-analysis
3. https://en.wikipedia.org/wiki/Naive_Bayes_classifier
4. https://scikit-learn.org/stable/modules/naive_bayes.html