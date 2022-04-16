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
The ingredients_corpus function will take the dataframe returned from the read_json function along with list of input_ingredients as arguments. The ingredients_corpus function will first convert the ingredients column present in the dataframe into lower case and using join function all the ingredients are joined and appended into a ingredients_list. similarly, input_ingredients list is also appended into the ingredients_list. Finally, the function returns the ingredients_list. Assumptions made in this step are there are no null values in the dataframe.
### tfidf_vector(ingredients_corpus)
The tfidf_vector function will take the ingredients_list returned from the ingredients_corpus as argument. I implemented the model using TfidfVectorizer because TfidfVectorizer not only focuses on frequency of words present inside the corpus but also provides the importance of the words. Firstly, initializing the TfidfVectorizer and then using fit_transform to convert ingredients_corpus to a matrix. Finally, the function returns the matrix.
### cuisine_predictor(frequency, d_f)
The cuisine_predictor function will take the matrix returned from the tfidf_vector function along with the dataframe returned from the read_json as arguments. The first vector in the matrix is input_ingredients vector. So, excluding first vector from the matrix and training the model for the remaining vectors which is given to the X_train and the cuisines column from the dataframe is given to y_train. Both the X_train and y_train are given to a RandomForestClassifier to train, the first vector which is input_ingredients is then given to the model.predict to predict the cuisine type. Followed by, predict_proba gives the predicted probabilities for input_ingredients vector. Finally, the function returns the predicted cusine type along with maximum probability score. Assumptions made at this step are i have implemented different classifiers to test the model but found RandomForestClassifier more accurate when compared to other classifiers.
### similar_cuisines(d_f, frequency, cuisine, cuisine_score, n)
The similar_cuisines function will take the dataframe, matrix, cuisine and probability score returned from the cuisine_predictor function as arguments along with 'n' where 'n' represents the top 'n' cuisines that we are supposed to display. In order to implement the similarity score i used cosine_similarity function. The input_ingredients vector and all the remaining vectors in the matrix are passed into cosine_similarity function which returns similarity score. The similarity scores along with their index values are then appended into a list. The list is then sorted in descending order of their scores. Followed by, i converted the 'id' column in the dataframe to string type, then based on 'n' value the top 'n' cuisine ID's and the respective similarity score is appended into a dictionary. Finally, using json.dumps function dictionary is converted into json format and returns the json_output.
### main
The main function contains calls to all the functions present in project2.py. 
### project2.py execution
After connecting to the instance using SSH.

Clone the repository: https://github.com/Bhavana-parupalli/cs5293sp22-project2

Give the following command in command line.
```bash
pipenv run python project2.py --N 5 --ingredient paprika --ingredient banana --ingredient "rice krispies" 
```
## tests
## test_1.py
The test_1.py contains different test cases to test the functions present inside the project2.py. The test_1.py returns the passed and failed test cases.
### test_read_json()
The test_read_json function passes the yummly.json file to read_json function which is present inside project2.py and gets the dataframe. If the extracted output is of type 'pandas.core.frame.DataFrame' then the test case will pass else fail.
### test_ingredients_corpus()
The test_ingredients_corpus function passes the yummly.json file to the read_json() in project2.py and gets the dataframe, the dataframe along with input_ingredients list is passed to ingredients_corpus() function which is present inside project2.py and get the ingredients_corpus. The ingredients_corpus() returns the list type. If the ingredients_corpus is of type list then the test case will pass else fail.
### test_tfidf_vector()
The test_tfidf_vector function passes the yummly.json file to read_json() in project2.py and gets the dataframe, the dataframe along with input_ingredients are passed to ingredients_corpus() in project2.py and get the ingredients_corpus. The ingredients_corpus is then passed to tfidf_vector() in project2.py and gets the output. The tfidf_vector() returns the matrix output. Therefore, if the output is of type scipy.sparse.csr_matrix then test case will pass else fail. 
### test_cuisine_predictor()
The test_cuisine_predictor function passes the yummly.json file to read_json which is present inside project2.py and gets the dataframe, the dataframe along with input_ingredients are passed to ingredients_corpus() in project2.py and gets the ingredients_corpus. Followed by, the ingredients_corpus is passed to tfidf_vector() in project2.py and gets the matrix. The matrix and dataframe are finally passed into cuisine_predictor in poject2.py and gets the cuisine type and it's score. The cuisine_predictor() returns the cuisine which is present inside an array along with score which is of type float. If the returned cuisine is of type array and score is of type float, then the test case will pass else fail. 
### test_similar_cuisines()
The test_similar_cuisines function passes the yummly.json file to read_json which is present inside project2.py and gets the dataframe, the dataframe along with input_ingredients are passed to ingredients_corpus() in project2.py and gets the ingredients_corpus, then the ingredients_corpus is passed to tfidf_vector() in project2.py and gets the matrix. The matrix and dataframe are then passed into a cuisine_predictor in project2.py and gets the cuisine type along with the score. The dataframe, matrix, cuisine type, score and 'n' are passed to similar_cuisines() in project2.py and gets the json output. If the returned output is of type str then test case will pass else fail.
### Test cases execution
After connecting to the instance using SSH.

Clone the repository: https://github.com/Bhavana-parupalli/cs5293sp22-project2

Give the following command in command line.
```bash
pipenv run python -m pytest
```

