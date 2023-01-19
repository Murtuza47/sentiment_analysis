import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense, Dropout, SpatialDropout1D
from tensorflow.keras.layers import Embedding

df = pd.read_csv("./dataset.csv")

df.head()
df.columns

dataset_df = df[['text','airline_sentiment']]
# print("Shape ---->", dataset_df.shape,"\n") #(14640, 2) 

# it return 5 columns form the table
# print("Dataset ----->",1,"\n")

# we are filtering the data removing netural value from the data
dataset_df = dataset_df[dataset_df['airline_sentiment'] != 'neutral']
# print("Filtered datset shape --->",dataset_df.shape,"\n")
dataset_df.head(5)

dataset_df["airline_sentiment"].value_counts()

# print("Value counts ---->\n",dataset_df["airline_sentiment"].value_counts(),"\n")

# it labels the data in numeric like 0,1,2 according values since we have two 
# values positive and negative there it is labalized as 0 and 1
sentiment_label = dataset_df.airline_sentiment.factorize()
# print("sentiment_label ----->",sentiment_label,"\n")

dataset_text_value = dataset_df.text.values # getting the values of text
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(dataset_text_value) # to get unique integer value for word (and it return NONE)
vocab_size = len(tokenizer.word_index) + 1 #word indexed 
encoded_docs = tokenizer.texts_to_sequences(dataset_text_value) #Transforms each text in texts to a sequence of integers
padded_sequence = pad_sequences(encoded_docs, maxlen=200)

print(tokenizer.word_index)
print(dataset_text_value[0])
print(encoded_docs[0])
print(padded_sequence[0]) #it makes the length of sequence of word equal by adding 0 in the list

# For sentiment analysis project, we use LSTM layers in the machine learning model. The architecture of our model
# consists of an embedding layer, an LSTM layer, and a Dense layer at the end. To avoid overfitting, 
#  we introduced the Dropout mechanism in-between the LSTM layers.

embedding_vector_length = 32
model = Sequential() 
model.add(Embedding(vocab_size, embedding_vector_length, input_length=200) )
model.add(SpatialDropout1D(0.25))
model.add(LSTM(50, dropout=0.5, recurrent_dropout=0.5))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid')) 
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])  
print(model.summary())

history = model.fit(padded_sequence,sentiment_label[0],validation_split=0.2, epochs=5, batch_size=32)

plt.plot(history.history['accuracy'], label='acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()
plt.savefig("Accuracy plot.jpg")

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()
plt.savefig("Loss plot.jpg")

def predict_sentiment(text):
    tw = tokenizer.texts_to_sequences([text])
    tw = pad_sequences(tw,maxlen=200)
    prediction = int(model.predict(tw).round().item())
    print("Predicted label: ", sentiment_label[1][prediction])

test_sentence1 = "I enjoyed my journey on this flight."
predict_sentiment(test_sentence1)

test_sentence2 = "This is the worst flight experience of my life!"
predict_sentiment(test_sentence2)
