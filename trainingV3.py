import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

# Cargar los datos del archivo JSON
intents = json.loads(open('intents.json').read())

# Inicializar el lematizador
lemmatizer = WordNetLemmatizer()

# Preprocesamiento de texto y generación de datos
words = []
classes = []
documents = []
ignore_letters = ['?', '!', '¿', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenizar palabras y agregarlas a la lista de palabras
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        # Agregar patrón y clase a los documentos
        documents.append((word_list, intent['tag']))
        # Agregar la clase a la lista de clases si no está presente
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lematizar palabras y eliminar caracteres ignorados
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(set(words))

# Guardar las palabras y clases en archivos pickle
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Crear el conjunto de entrenamiento
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    word_patterns = doc[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# Mezclar los datos de entrenamiento
random.shuffle(training)

# Convertir a numpy arrays
training = np.array(training, dtype=object)  # Usar dtype=object para manejar listas de diferentes longitudes

# Separar características y etiquetas
train_x = list(training[:, 0])
train_y = list(training[:, 1])

# Crear el modelo
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compilar el modelo
# Compilar el modelo
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001, decay=1e-6), metrics=['accuracy'])


# Entrenar el modelo
history = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

# Guardar el modelo
model.save('chatbot_model.h5')
print("Modelo guardado exitosamente.")
