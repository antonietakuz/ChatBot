import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

# Cargar el modelo
model = load_model('chatbot_model.h5')

# Cargar datos
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
intents = json.loads(open('intents.json').read())

# Inicializar lematizador
lemmatizer = WordNetLemmatizer()

# Función para limpiar la oración
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Función para crear la bolsa de palabras
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for w in sentence_words:
        if w in words:
            bag[words.index(w)] = 1
    # Asegurar que el vector de características tenga la misma longitud que el número de palabras únicas
    bag = bag[:len(words)]
    # Ajustar la longitud del vector de características para que coincida con la dimensión esperada por el modelo
    bag += [0] * (16 - len(bag))
    return np.array(bag)

# Función para predecir la clase
def predict_class(sentence):
    bow = bag_of_words(sentence)
    # Hacer la predicción
    result = model.predict(np.array([bow]))[0]
    # Obtener el índice de la clase con la mayor probabilidad
    predicted_class_index = np.argmax(result)
    # Devolver la etiqueta de clase correspondiente
    predicted_class = classes[predicted_class_index]
    return predicted_class

# Función para obtener una respuesta aleatoria de acuerdo a la etiqueta de clase
def get_response(tag, intents_json):
    list_of_intents = intents_json['intents']
    for intent in list_of_intents:
        if intent['tag'] == tag:
            # Seleccionar una respuesta aleatoria del conjunto de respuestas asociadas a la etiqueta de clase
            response = random.choice(intent['responses'])
            return response
    # Si no se encuentra una respuesta para la etiqueta de clase, devolver una respuesta genérica
    return "Lo siento, no entiendo tu pregunta."

while True:
    user_input = input("You: ")
    # Permitir al usuario salir del bucle si ingresa 'adios'
    if user_input.lower() == 'adios':
        print("Hasta luego!")
        break
    predicted_tag = predict_class(user_input)
    response = get_response(predicted_tag, intents)
    print("Bot:", response)
