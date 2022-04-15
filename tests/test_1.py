import pytest
import project2 as p2
import pandas
import scipy
import numpy
import json

def test_read_json():
    d_f=pandas.read_json("yummly.json")
    assert type(d_f)==pandas.core.frame.DataFrame

def test_ingredients_corpus():
    d_f=pandas.read_json("yummly.json")
    input_ingredients=['paprika', 'banana', 'rice krispies']
    ingredients_corpus=p2.ingredients_corpus(d_f, input_ingredients)
    assert type(ingredients_corpus)==list

def test_tfidf_vector():
    d_f=pandas.read_json("yummly.json")
    input_ingredients=['paprika', 'banana', 'rice krispies']
    ingredients_corpus=p2.ingredients_corpus(d_f, input_ingredients)
    frequency=p2.tfidf_vector(ingredients_corpus)
    assert type(frequency)==scipy.sparse.csr_matrix

def test_cuisine_predictor():
    d_f = pandas.read_json("yummly.json")
    input_ingredients = ['paprika', 'banana', 'rice krispies']
    ingredients_corpus = p2.ingredients_corpus(d_f, input_ingredients)
    frequency = p2.tfidf_vector(ingredients_corpus)
    (cuisine, score)=p2.cuisine_predictor(frequency, d_f)
    assert type(cuisine)==numpy.ndarray and type(score)==numpy.float64

def test_similar_cuisines():
    d_f = pandas.read_json("yummly.json")
    input_ingredients = ['paprika', 'banana', 'rice krispies']
    ingredients_corpus = p2.ingredients_corpus(d_f, input_ingredients)
    frequency = p2.tfidf_vector(ingredients_corpus)
    (cuisine, score)=p2.cuisine_predictor(frequency, d_f)
    top_n_cuisines=5
    json_format=p2.similar_cuisines(d_f, frequency, cuisine, score, top_n_cuisines)
    assert type(json_format)==str

