import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
import json
import scipy
import numpy
import argparse

def read_json(path):
    d_f=pd.read_json(path)
    return d_f

def ingredients_corpus(d_f, input_ingredients):
    end = d_f.index.size
    for i in range(0, end):
        for j in range(len(d_f['ingredients'][i])):
            d_f['ingredients'][i][j] = "".join(d_f['ingredients'][i][j].split(" ")).lower()
    ingredients_list=[]
    for i in range(len(input_ingredients)):
        input_ingredients[i]="".join(input_ingredients[i].split(" ")).lower()
    ingredients_list.append(" ".join(input_ingredients))
    for i in d_f['ingredients']:
        st=" ".join(i)
        ingredients_list.append(st)
    return ingredients_list

def tfidf_vector(ingredients_corpus):
    vectorizer=TfidfVectorizer()
    frequency=vectorizer.fit_transform(ingredients_corpus)
    return frequency

def cuisine_predictor(frequency, d_f):
    X_train = frequency[1:]
    y_train = d_f['cuisine']
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    cuisine_prediction = model.predict(frequency[0])
    cuisine = cuisine_prediction
    probability = model.predict_proba(frequency[0])
    score = max(probability[0])
    return cuisine, score

def similar_cuisines(d_f, frequency, cuisine, cuisine_score, n):
    l2 = []
    l3 = []
    l4 = []
    closest_cuisines = []
    dict = {}
    l = cosine_similarity(frequency[0], frequency[1:])
    for i in range(0, len(l[0])):
        l1 = []
        l1.append(l[0][i])
        l1.append(i)
        l2.append(l1)
    for i in range(0, len(l2)):
        l3.append(l2[i][0])
    l3.sort(reverse=True)
    for i in range(0, n):
        for j in range(0, len(l2)):
            if l3[i] == l2[j][0]:
                l4.append(l2[j])
    d_f['id'] = pd.Series(d_f['id'], dtype="string")
    for score, index in l4:
        dict = {"id": d_f['id'][index], "score": round(score, 2)}
        closest_cuisines.append(dict)
    cuisines_dict = {'cuisine': cuisine[0], 'score': cuisine_score}
    cuisines_dict['closest'] = closest_cuisines
    json_format = json.dumps(cuisines_dict, indent=4)
    print(json_format)
    return json_format

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument("--N", required=True, type=int)
    parser.add_argument("--ingredient", action="append")
    args=parser.parse_args()

    d_f=read_json("yummly.json")

    ingredients_corpus=ingredients_corpus(d_f, args.ingredient)

    frequency=tfidf_vector(ingredients_corpus)

    (cuisine, score)=cuisine_predictor(frequency, d_f)

    json_output=similar_cuisines(d_f, frequency, cuisine, score, args.N)











