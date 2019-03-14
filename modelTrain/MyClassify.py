import json
import re
import numpy as np
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import AlphaDropout, Dropout
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Concatenate
from keras.layers import LSTM, Bidirectional, GRU
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import plot_model

labels = ['disgust', 'joy', 'anger', 'noemo', 'surprise', 'sadness', 'fear']

# Embedding
max_features = 20000
maxlen = 100
embedding_size = 128

# Training
batch_size = 32
epochs = 4

def get_emotion(emovals, labels, emotions):
    truthy = len(list(filter(bool, emovals.values())))
    if truthy == 1:
        emotion = [v for v in emovals if emovals[v]][0]
    elif truthy == 0:
        emotion = "noemo"
    else:
        raise ValueError("Dataset marked as 'single' contains multiple emotions")
    return emotions.get(emotion, emotions.get("noemo"))

def get_train_test(jsonfile, labels, train_size):
    training, testing = [], []

    with open(jsonfile) as f:
        for line in f:
            data = json.loads(line)
            training.append(data)
    training, testing = train_test_split(training, train_size=train_size)


    train_x, train_y, test_x, test_y = [], [], [], []
    emotions = {label: x for x, label in enumerate(labels)}

    for data in training:
        train_x.append(data["text"])
        train_y.append(get_emotion(data["emotions"], labels, emotions))

    for data in testing:
        test_x.append(data["text"])
        test_y.append(get_emotion(data["emotions"], labels, emotions))

    with open('word_index.json') as f:
        tokenizer = json.load(f)
    start_char = 1
    oov_char = 2
    index_from = 3

    for i in range(len(train_x)):
        train_x[i] = train_x[i].lower()
        train_x[i] = re.sub(r"\\u\S+", ' ', train_x[i])
        train_x[i] = re.sub(r"@\S+", ' ', train_x[i])
        train_x[i] = re.sub(r"http\S+", ' ', train_x[i])
        train_x[i] = re.sub(r'[^\w\s]', ' ', train_x[i])
        # train_x[i] = re.sub(r'\s', '', train_x[i])
        train_x[i] = train_x[i].split(' ')
        while '' in train_x[i]:
            train_x[i].remove('')
        for j in range(len(train_x[i])):
            if train_x[i][j] in tokenizer.keys():
                train_x[i][j] = tokenizer[train_x[i][j]] + index_from
                if train_x[i][j] >= max_features:
                    train_x[i][j] = oov_char
            else:
                train_x[i][j] = oov_char
        train_x[i] = [start_char] + train_x[i]

    for i in range(len(test_x)):
        test_x[i] = test_x[i].lower()
        test_x[i] = re.sub(r"\\u\S+", ' ', test_x[i])
        test_x[i] = re.sub(r"@\S+", ' ', test_x[i])
        test_x[i] = re.sub(r"http\S+", ' ', test_x[i])
        test_x[i] = re.sub(r'[^\w\s]', ' ', test_x[i])
        # train_x[i] = re.sub(r'\s', '', train_x[i])
        test_x[i] = test_x[i].split(' ')
        while '' in test_x[i]:
            test_x[i].remove('')
        for j in range(len(test_x[i])):
            if test_x[i][j] in tokenizer.keys():
                test_x[i][j] = tokenizer[test_x[i][j]] + index_from
                if test_x[i][j] >= max_features:
                    test_x[i][j] = oov_char
            else:
                test_x[i][j] = oov_char
        test_x[i] = [start_char] + test_x[i]

    train_x_s = sequence.pad_sequences(train_x, maxlen=maxlen)
    test_x_s = sequence.pad_sequences(test_x, maxlen=maxlen)

    train_y_c = keras.utils.to_categorical(train_y, labels.__len__())
    test_y_c = keras.utils.to_categorical(test_y, labels.__len__())

    return np.array(train_x_s), np.array(train_y_c), np.array(test_x_s), np.array(test_y_c)


train_x, train_y, test_x, test_y = get_train_test("balanced-dataset.jsonl", labels, 0.95)

model = Sequential()
model.add(Embedding(max_features, embedding_size, input_length=maxlen))
model.add(Bidirectional(LSTM(100)))
model.add(Dropout(0.5))
model.add(Dense(labels.__len__(), activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.summary()
# plot_model(model,to_file='model.png')
model.fit(train_x, train_y,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_split=0.1)

# （命名）Sentiment_Analyze_Model_模型版本.h5
model.save('Sentiment_Analyze_Model_1_8.h5')
# 1.1
# batch_size = 20, LSTM128, acc=0.6005
# 1.2
# batch_size = 50, acc=0.6106
# 1.3
# Dropout(0.8), batch_size = 64, epochs = 5, acc=0.5917
# 1.4
# Dropout(0.5), del Bidirectional, embedding_size = 512, acc=0.5901
# 1.5
# add Dense, embedding_size = 256, acc=0.5946
# 1.6
# origin, acc=0.6207
# 1.7
# Dropout(0.8), batch_size = 32, acc=0.5888
# 1.8
# Dropout(0.5), acc=0.6136 GOOD?
# 1.9
# Dropout(0.3), acc=0.6111

# 2.1
# batch_size = 50, embedding_size = 128, LSTM64, add Dense, acc=0.5937
# 2.2
# optimizer = rmsprop, acc=0.5745
# 2.3
# embedding_size = 64, acc=0.5305

# 3.1
# GRU, batch_size = 50, acc=0.6093 testgood?
# 3.2
# sample = single-balanced-dataset, acc=0.4067
# 3.3
# Dropout(0.5), acc=0.4111

# 4.1
# add cnn, Dropout(0.5), LSTM(100), acc=0.5550

# 5.1
# add cnn2, del Bidirectional, acc=
score, acc = model.evaluate(test_x, test_y, batch_size=batch_size, verbose=1)
print('Test score:', score)
print('Test accuracy:', acc)