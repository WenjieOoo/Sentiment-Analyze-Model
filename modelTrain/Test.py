from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import text
from keras.preprocessing import sequence
import pickle
import json
import re
import numpy as np
import tensorflowjs as tfjs

# 要测试的句子
test_string = ['I got A in the exam today.',]

model = load_model('Sentiment_Analyze_Model_1_8.h5')
with open('word_index.json') as f:
    tokenizer = json.load(f)
start_char = 1
oov_char = 2
index_from = 3
test_string[0]= test_string[0].lower()
test_string[0] = re.sub(r'[^\w\s]', ' ', test_string[0])
test_string[0] = test_string[0].split(' ')
for j in range(len(test_string[0])):
    if test_string[0][j] in tokenizer.keys():
        test_string[0][j] = tokenizer[test_string[0][j]] + index_from
        if test_string[0][j] >= 20000:
            test_string[0][j] = oov_char
    else:
        test_string[0][j] = oov_char
test_string[0] = [start_char] + test_string[0]
text_to_pred = sequence.pad_sequences(test_string, maxlen=100)

pred = model.predict(x=text_to_pred,verbose=1)
labels = ['disgust', 'joy', 'anger', 'noemo', 'surprise', 'sadness', 'fear']

# 打印推断结果
print(labels[pred.argmax(1)[0]])