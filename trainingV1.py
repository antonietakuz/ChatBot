#-----------------------------------------------------------------------------------------------------
#Importaciones de librerias
#-----------------------------------------------------------------------------------------------------
import random
import json
import pickle

import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer


from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
#from keras.optimizers import sgd_experimental 

from keras.optimizers import SGD



#---------------------------------------------------------------------------------------------------
#Prepara los datos necesarios para entrenar un modelo de clasificación de texto, 
#tokenizando las palabras, lematizando y almacenando las palabras únicas y las etiquetas de clase
#y las etiquetas de clase para su uso posterior en el entrenamiento del modelo.
#----------------------------------------------------------------------------------------------------
lemmatizer=WordNetLemmatizer()
intents=json.loads(open('intents.json').read())

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')


words=[]
classes=[]
documents=[]
ignore_letters=['?', '!', '¿', '.', ',']

#separamos las palabras de los tags
for intent in intents['intents']:
    for pattern in intent['patterns']:
        world_list=nltk.word_tokenize(pattern)
        words.extend(world_list)
        documents.append((world_list, intent["tag"]))
        if intent['tag'] not in classes:
            classes.append(intent["tag"])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))


pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

#-----------------------------------------------------------------------------------------------------
#Entrenamos el modelo
#-----------------------------------------------------------------------------------------------------

training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = [0] 
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    
    output_row=list(output_empty)
    output_row[classes.index(document[1])]=1
    training.append([bag, output_row])


random.shuffle(training)

# Crear arreglos numpy separados para características de entrada y etiquetas de salida
data_array = np.array([data for data, _ in training])
label_array = np.array([label for _, label in training])


# Combinar los arreglos en un solo arreglo numpy
training = np.column_stack((data_array, label_array))

print (training)

# Utilizar numpy para crear arreglos de características de entrada y etiquetas de salida
train_x = training[:, :-1]  # Todas las columnas excepto la última son características de entrada
train_y = training[:, -1]    # Última columna es la etiqueta de salida

# Crear modelo secuencial
model = Sequential()
model.add(Dense(128, input_shape=(train_x.shape[1],), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))  # Solo necesitamos una neurona de salida para la clasificación binaria

# Compilar el modelo
sgd = SGD(learning_rate=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Entrenar el modelo
train_process = model.fit(train_x, train_y, epochs=100, batch_size=5, verbose=1)

# Guardar el modelo
model.save('chatbot_model.h5')




