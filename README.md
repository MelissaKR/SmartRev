# SmartRev

## Introduction
SmartRev is a web app created as part of Insight Data Science project that is designed to help pet owners navigate critical reviews for pet food on Chewy's website smartly. After scraping reviews with 3 stars and less, the model classifies each reviews as either health, quality or service. Moreover, the reviews in each category are ordered according to the number of helpful votes given to them by others, and the review with the highest votes is shown.

## How SmartRev Works
After prodiving the url of the pet food on Chewy's website, the user can see a visual breakdown of reviews across the 3 categories, as well as the review voted most helpful for each class.

Behind the curtain, critical reviews from the url are scraped and fed to the model after some pre-processing on the body of the reviews. The predicted class for each review is then obtained using a Rest API call to the model, which is a 2-layer LSTM (Long Short-Term Memory) neural network model and is deployed on Google AI Platform.

<img src="/images/App.png" width="600" height="350" />


SmartRev can be visited at: https://petfoodrecommend.appspot.com/


## Datasets
Amazon Customer Reviews Dataset for Pet products available on AWS, as well as critical reviews for pet foods scraped from Chewy's website have been used as input datasets to train the model. The scraped Chewy dataset can be found in the "data" folder.

## Models 
Two approaches have been taken with respect to the modeling aspect of the problem:

**1. Unsupervised Learning: LDA + TF-IDF**

Latent Dirichlet Allocation (LDA) is one of the most popular topic modelling techniques. It's a probabilistic model for discovering the bastract topics in a word document by estimating probability distributions for topics in the document and words in topics. It is a powerful exploratory tool for clustering different topics in a text document.

The different steps involved in building a LDA model involves tokenizing, lemmatizing words in the text document, and finding the optimum number of topics using coherence score between words in each topic. 

The results from deploying LDA with 10 topics (that has been found the optimum number of topics) on the corpus of reviews are shown in the visualization below, which is created using **pyLDAvis** package. It shows the words most associated with each topic, as well as the percentage of tokens for each topic.

<p align="center">
    <img src="/images/LDA.gif" width="800" height="500" />
</p>

A close look at the top 10 words in each topic reveals that these topics can be distilled into a number of general topics; shipping and handling problems (Topic1, Topic4, and Topic8), and food-related issues (Topic9, Topic7, and Topic10). However, there are a few problems with this approach. The first is that in some cases it is difficult to understand what problem a topic is really about based on the associated words. For example, Topic3, Topic5, and Topic6, while Topic3 and Topic5 comprise the majority of the tokens in the corpus. The other issue is that some of the words are not helpful at all; food, cat, dog, like are not really the most helpful words here. The last problem is that even though there is very little overlap between the different topics in the 2D space, some of these topics may potentially be pointing to pretty much the same issues. 

Perhaps a supervised learning may be more helpful in terms of a more accurate implementation of the classification process in this case.

**2. Supervised Learning**

In order to train any classification model, we'd need labels for our data, which is the case here. In order to do that, a dictionary of key words for each of the three topics of "Health", "Quality" and "Service" has been created. Then, each review has been classified under a topic is a word in that topic had appeared in the review, with "Health" having priority over "Quality", and "Quality" over "Service".                                                                                                                                           
Next, three neural network models have been tried. Neural Networks are a powerful tool for uncovering the intrinsic features in the data, especially for unstructured datasets.

The trained models are:
  
### a. 1 Dimensional Convolutional Neural Network (CNN)

CNNs are easier to tune and faster to run. A 1D CNN model works by discovering local, short-term anomalies in the text. Therefore, it is a good starting place. The sparse categorical accuracy on validation data using this model was 91 %.

<p align="center">
  <img width="450" height="350" src="/images/CNN.png">
</p>


### b. Multilayer LSTM (Long Short-Term Memory)

An LSTM model is capable of considering long-term dependencies in a text document, which may be more useful for out problem. Normally, critical reviews are longer than a few sentences, and may hit on different subjects with varying ranges of sentiments. Therefore, a model that does not account for longer-term dependencies may miss on the subtle defining structure of reviews for each category. The LSTM model achieved an accuracy of 95% in this case.

<p align="center">
  <img width="200" height="350" src="/images/LSTM.png">
</p>

### c. Multilayer Bidirectional LSTM

Finally, a Bidirectional LSTM model was trained to see whether including information on what comes next in a sentences as well as what has come before can make any significant improvements in our results. What was discovered was that it takes almost twice as long to train than an LSTM model, and reaches an accuracy of 94% on validation dataset. 

<p align="center">
  <img width="230" height="350" src="/images/BiLSTM.png">
</p>

Since the LSTM model outperformed the other two, that was the model that was ultimately deployed to Google AI Platform.



The codebase for the model can be found in **petfood_review** notebook, and the web app files using Streamlit can be found in **SmartRev** folder. For easier navigation between different sections in the notebook, as well as being able to view some of the plots, please use the following link:

https://nbviewer.jupyter.org/github/MelissaKR/SmartRev/blob/master/perfood_review.ipynb
