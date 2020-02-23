# SmartRev

## Introduction
SmartRev is a web app created as part of Insight Data Science project that is designed to help pet owners navigate critical reviews for pet food on Chewy's website smartly. After scraping reviews with 3 stars and less, the model classifies each reviews as either health, quality or service. Moreover, the reviews in each category are ordered according to the number of helpful votes given to them by others, and the review with the highest votes is shown.

## How SmartRev Works
After prodiving the url of the pet food on Chewy's website, the user can see a visual breakdown of reviews across the 3 categories, as well as the review voted most helpful for each class.

Behind the curtain, critical reviews from the url are scraped and fed to the model after some pre-processing on the body of the reviews. The predicted class for each review is then obtained using a Rest API call to the model, which is a 2-layer LSTM (Long Short-Term Memory) neural network model and is deployed on Google AI Platform.

<img src="/images/App.png" width="600" height="350" />


SmartRev can be visited at: https://petfoodrecommend.appspot.com/


## Datasets
Amazon Customer Reviews Dataset for Pet products available on AWS, as well as critical reviews for pet foods scraped from Chewy's website have been used as input datasets to train the model.

## Models 
Two approaches have been taken with respect to the modeling aspect of the problem:

**1. Unsupervised Learning: LDA + TF-IDF**

<img src="/images/LDA.gif" width="800" height="600" />


**2. Supervised Learning**

  The trained models are:
  
  a. 1 Dimensional Convolutional Neural Network (CNN)
  
  b. Multilayer LSTM (Long Short-Term Memory)
  
  c. Multilayer Bidirectional LSTM
  
  Among these models, the LSTM model outperformed the other two with respect to sparse categorical accuracy on the validation dataset (95%) used as the evaluation metric, and that was the model that was ultimately deployed to Google AI Platform.



The codebase for the model can be found in **petfood_review** notebook, and the web app files using Streamlit can be found in **SmartRev** folder.
