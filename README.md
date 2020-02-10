# SmartRev

## Introduction
SmartRev is a web app created as part of an Insight Data Science project that is designed to help pet owners navigate critical reviews for pet food on Chewy's website smartly. After scraping reviews with 3 stars and less, the model classifies each reviews as either health, quality or service. 

## How SmartRev Works
After prodiving the url of the pet food on Chewy's website, the user can see a visual breakdown of reviews across the 3 categories, as well as the review voted most helpful for each class.

## Datasets
Amazon Customer Reviews Dataset for Pet products available on AWS, as well as critical reviews for pet foods scraped from Chewy's website have been used as input datasets.

## Models 
Two approaches have been taken with respect to the modeling aspect of the problem:

1. Unsupervised Learning: LDA + TF-IDF
2. Supervised Learning

  The models trained are:
  
  a. 1 Dimensional Convolutional Neural Network (CNN)
  
  b. Multilayer LSTM (Long Short-Term Memory)
  
  c. Multilayer Bidirectional LSTM
  
  Among these models, the LSTM model outperformed the other two with respect to the sparse categorical accuracy (95%) used as the evaluation metric, and that was the model that was ultimately deployed to Google AI Platform.



The codebase for the model can be found in **petfood_review** notebook, and the web app files using Streamlit can be found in **SmartRev** folder.
