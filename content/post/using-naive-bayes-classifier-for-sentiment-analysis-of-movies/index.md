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



#### Links

1. [Movie Dataset on kaggle for sentiment analysis](https://www.kaggle.com/marklvl/sentiment-labelled-sentences-data-set)

#### References

1. https://en.wikipedia.org/wiki/Sentiment_analysis
2. https://www.lexalytics.com/technology/sentiment-analysis
3. https://en.wikipedia.org/wiki/Naive_Bayes_classifier
4. https://scikit-learn.org/stable/modules/naive_bayes.html