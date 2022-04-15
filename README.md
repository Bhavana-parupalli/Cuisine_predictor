# cs5293sp22-project2
## Author: Bhavana Parupalli
## Email: parupallibhavana123@ou.edu
## Packages installed
```bash 
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
import json
import scipy
import numpy
import argparse
import pytest
```
## project2.py
The project2.py contains five different methods which reads the yummly.json file and the ingredients column present inside the dataframe is converted into corpus along with input_ingredients. Followed by using TfidfVectorier the ingredients_corpus is converted into a matrix, and using RandomForestClassifier predicted the cuisine as well as probability score. Finally, using cosine_similarity got the top 'n' cuisines along with their similarity score and added all this values into a dictionary and displayed the output in json format.
### read_json(path)
The read_json function will take the yummly.json file as argument and read the json file and returns the dataframe.
### ingredients_corpus(d_f, input_ingredients)
The ingredients_corpus function will take the dataframe returned from the read_json function along with list of input_ingredients as arguments. The ingredients_corpus function will first convert the ingredients column present in the dataframe into lower case and using join function all the ingredients are joined and appended into a ingredients_list. similarly, input_ingredients list is also appended into the ingredients_list. Finally, the function returns the ingredients_list.
### tfidf_vector(ingredients_corpus)
The tfidf_vector function will take the ingredients_list returned from the ingredients_corpus as argument. I implemented the model using TfidfVectorizer because TfidfVectorizer not only focuses on frequency of words present inside the corpus but also provides the importance of the words. Firstly, initializing the TfidfVectorizer and then using fit_transform to convert ingredients_corpus to a matrix. Finally, the function returns the matrix.
### cuisine_predictor(frequency, d_f)
The cuisine_predictor function will take the matrix returned from the tfidf_vector function along with the dataframe returned from the read_json as arguments.  
