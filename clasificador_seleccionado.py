# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 18:08:31 2022

@author: pablo
"""

from nltk.probability import FreqDist
from nltk.lm.vocabulary import Vocabulary
import nltk
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import naive_bayes
import re
import unicodedata
import pathlib

################################## Funciones ###########################################################

def es_mensaje_no_deseado(ruta):
    
    # Lectura de los corpus
    file_legitimo = open("./corpusLegitimo.txt", "r")
    list_corpus_legitimo = eval(file_legitimo.read())
    file_legitimo.close()
    
    file_spam = open("./corpusSpam.txt", "r")
    list_corpus_spam = eval(file_spam.read())
    file_spam.close()
    
    file_dataset = open("./dataset.txt", "r", encoding="utf-8")
    dataset = eval(file_dataset.read())
    file_dataset.close()
    
    # Obtención del vocabulario
    vocab = crear_vocab(list_corpus_legitimo, list_corpus_spam)
    
    # Obtenemos y codificamos las clases de los correos de entrenamiento
    codificador_clases = preprocessing.LabelEncoder()
    
    clases_codificadas = codificar_objetivos(codificador_clases)
    
    # Entrenamiento de los modelos
    countvectorizer = CountVectorizer(vocabulary=vocab)
    count_entrenamiento = countvectorizer.fit_transform(dataset)
    k = 3 # Hiperarámetro de suavizado
    multinomial_NB = naive_bayes.MultinomialNB(alpha=k)
    multinomial_NB.fit(count_entrenamiento.toarray(), clases_codificadas)
    
    # Leer el correo y normalizarlo
    file_correo_prueba = open(ruta, "r")
    correo_prueba = file_correo_prueba.read()
    file_correo_prueba.close()
    
    list_correo_prueba_tokenized = nltk.word_tokenize(correo_prueba)
    list_correo_prueba_normalized = normalize(list_correo_prueba_tokenized)
    
    correo_prueba_normalized = [" ".join(list_correo_prueba_normalized)]
    
    # Predicción de la clase asignada al correo
    count_prueba = countvectorizer.transform(correo_prueba_normalized)
    
    prediccion_NB = multinomial_NB.predict(count_prueba.toarray())
    prediccion_NB = codificador_clases.inverse_transform(prediccion_NB)
    
    return prediccion_NB[0] == 'Spam'
    
    

def crear_vocab(list_corpus_legitimo, list_corpus_spam):
    # Calculamos las 500 palabras más comunes de cada tipo de correo

    fdist_legitimo = FreqDist(list_corpus_legitimo)
    most_common_legitimo = fdist_legitimo.most_common(500)

    fdist_spam = FreqDist(list_corpus_spam)
    most_common_spam = fdist_spam.most_common(500)
    
    # Creamos el vocabulario
    
    vocabulary_terms_repeated = [word[0] for word in most_common_spam] + [word[0] for word in most_common_legitimo]
    vocabulary_terms = [word[0] for word in most_common_spam] + [word[0] for word in most_common_legitimo]
    
    # Eliminamos aquellas palabras que aparezcan tanto en los no deseados como
    # en los legitimos, ya que no nos serviran para diferenciar los correos
    
    for word in vocabulary_terms_repeated:
        if vocabulary_terms.count(word) > 1:
            vocabulary_terms.remove(word)
            
    vocab = Vocabulary(vocabulary_terms)
    
    return vocab

def codificar_objetivos(codificador_clases):
    clases_entrenamiento = []
    
    for path in pathlib.Path("./Enron-Spam/subconjuntos/train/legítimo").iterdir():
        clases_entrenamiento.append('Legitimo')
        
    for path in pathlib.Path("./Enron-Spam/subconjuntos/train/no_deseado").iterdir():
        clases_entrenamiento.append('Spam')
    
    clases_codificadas = codificador_clases.fit_transform(clases_entrenamiento)
    return clases_codificadas

def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def normalize(words):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    return words


################################## Filtro ###########################################################


ruta = "./Enron-Spam/subconjuntos/val/no_deseado/297" # Modificar este valor a la ruta del correo a clasificar
clasificado_como_spam = es_mensaje_no_deseado(ruta)
print("¿Ha sido el correo marcado como spam?")
print(clasificado_como_spam)
    