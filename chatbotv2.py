import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit, QLineEdit, QPushButton, QLabel
from PyQt5.QtGui import QPixmap
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
from PyQt5.QtCore import Qt

class ChatbotWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Chatbot")
        self.setGeometry(100, 100, 600, 400)
        
        # Crear la imagen del chatbot
        self.chatbot_image_label = QLabel(self)
        self.chatbot_image_label.setGeometry(10, 10, 200, 200)
        pixmap = QPixmap('bot.jpg')  # Cambia 'chatbot_image.png' por la ruta de tu imagen
        self.chatbot_image_label.setPixmap(pixmap)

        self.chatbot_image_label = QLabel(self)
        self.chatbot_image_label.setGeometry(10, 10, 200, 200)
        pixmap = QPixmap('bot.jpg').scaled(200, 200, Qt.KeepAspectRatio)  # Cambia 'bot.jpg' por la ruta de tu imagen
        self.chatbot_image_label.setPixmap(pixmap)
        
        # Crear la caja de texto para mostrar el diálogo
        self.text_edit = QTextEdit(self)
        self.text_edit.setGeometry(220, 10, 370, 320)
        self.text_edit.setReadOnly(True)
        
        # Crear el cuadro de entrada para el usuario
        self.input_line_edit = QLineEdit(self)
        self.input_line_edit.setGeometry(220, 340, 370, 30)
        
        # Crear el botón de enviar
        self.send_button = QPushButton("Enviar", self)
        self.send_button.setGeometry(500, 340, 90, 30)
        self.send_button.clicked.connect(self.on_send_clicked)
        
        # Cargar el modelo y los datos
        self.model = load_model('chatbot_model.h5')
        self.words = pickle.load(open('words.pkl', 'rb'))
        self.classes = pickle.load(open('classes.pkl', 'rb'))
        
        # Cargar el archivo JSON con la codificación UTF-8
        with open('intents.json', 'r', encoding='utf-8') as file:
            self.intents = json.load(file)
        
        self.lemmatizer = WordNetLemmatizer()
    
    def clean_up_sentence(self, sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [self.lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words
    
    def bag_of_words(self, sentence):
        sentence_words = self.clean_up_sentence(sentence)
        bag = [0]*len(self.words)
        for w in sentence_words:
            if w in self.words:
                bag[self.words.index(w)] = 1
        bag = bag[:len(self.words)]
        bag += [0] * (16 - len(bag))
        return np.array(bag)
    
    def predict_class(self, sentence):
        bow = self.bag_of_words(sentence)
        result = self.model.predict(np.array([bow]))[0]
        predicted_class_index = np.argmax(result)
        predicted_class = self.classes[predicted_class_index]
        return predicted_class
    
    def get_response(self, tag, intents_json):
        list_of_intents = intents_json['intents']
        for intent in list_of_intents:
            if intent['tag'] == tag:
                response = random.choice(intent['responses'])
                return response
        return "Lo siento, no entiendo tu pregunta."
    
    def on_send_clicked(self):
        user_input = self.input_line_edit.text()
        self.input_line_edit.clear()
        self.text_edit.append("You: " + user_input)
        predicted_tag = self.predict_class(user_input)
        response = self.get_response(predicted_tag, self.intents)
        self.text_edit.append("Bot: " + response + "\n")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ChatbotWindow()
    window.show()
    sys.exit(app.exec_())
