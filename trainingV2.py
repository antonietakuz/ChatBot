import random
import json
import pickle

import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

#---------------------------------------------------------------------------------------------------
# Preparar los datos necesarios para entrenar un modelo de clasificación de texto
# tokenizando las palabras, lematizando y almacenando las palabras únicas y las etiquetas de clase
# para su uso posterior en el entrenamiento del modelo.
#----------------------------------------------------------------------------------------------------
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '¿', '.', ',']

# Separar las palabras de los tags
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent["tag"]))
        if intent['tag'] not in classes:
            classes.append(intent["tag"])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

#-----------------------------------------------------------------------------------------------------
# Entrenar el modelo
#-----------------------------------------------------------------------------------------------------

training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = [0] * len(words)
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)

# Separar características de entrada y etiquetas de salida
training_x = np.array([data[0] for data in training])  # Características de entrada
training_y = np.array([label for _, label in training])  # Etiquetas de salida

# Crear modelo secuencial
model = Sequential()
model.add(Dense(128, input_shape=(len(words),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(classes), activation='softmax'))  # Usar softmax para clasificación multi-clase

# Compilar el modelo
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)  # Ajustado para ignorar la advertencia de 'decay'
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Entrenar el modelo
train_process = model.fit(training_x, training_y, epochs=100, batch_size=5, verbose=1)

# Guardar el modelo
model.save('chatbot_model.h5')

