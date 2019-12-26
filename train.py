import numpy as np
import nltk
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, LSTM, CuDNNLSTM, Embedding, Dropout, SpatialDropout1D, Bidirectional, TimeDistributed, Flatten, GlobalMaxPooling1D, GlobalAveragePooling1D, concatenate
from keras.models import Sequential,load_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras_self_attention import SeqSelfAttention
from keras.utils import to_categorical
from sklearn import preprocessing
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

def read_data(path):
    lst = []
    with open(path,'r',encoding='UTF-8') as p:
        for line in p.readlines():
            if line.rstrip('\n').isdigit():
                lst.append(int(line))
            else:
                lst.append(line.rstrip('\n'))
    return lst

train_set = read_data('./data/train_set.txt')
train_label = read_data('./data/train_label.txt')
val_set = read_data('./data/val_set.txt')
val_label = read_data('./data/val_label.txt')
test_set = read_data('./data/test_set.txt')
extra_set = read_data('./data/extra_set.txt')

texts = val_set + test_set + train_set + extra_set
max_features = 50000
tok = Tokenizer(num_words=max_features)
tok.fit_on_texts(texts)
list_tok = tok.texts_to_sequences(texts)#文本序列化
word_index = tok.word_index
embed_size = 128
maxlen = 120
seq_tok = pad_sequences(list_tok, maxlen=maxlen)#填充为0
print(len(word_index))

lbl_enc = preprocessing.OneHotEncoder()#onehot编码
x_train = seq_tok[15000:45000]
y_train = lbl_enc.fit_transform(np.array(train_label).reshape(-1,1))
x_test = seq_tok[:7500]
y_test = lbl_enc.fit_transform(np.array(val_label).reshape(-1,1))
final_test = seq_tok[7500:15000]
x_extra = seq_tok[45000:]

def get_lstm_model():
    model = Sequential()
    model.add(Embedding(len(word_index)+1, embed_size, input_length =maxlen))
    model.add(SpatialDropout1D(0.2))
    model.add(Bidirectional(LSTM(64, recurrent_dropout=0.2, return_sequences=True)))
    model.add(Bidirectional(LSTM(64, recurrent_dropout=0.2, return_sequences=True)))
    model.add(SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
                       kernel_regularizer=regularizers.l2(1e-3),
                       bias_regularizer=regularizers.l1(1e-3),
                       attention_regularizer_weight=1e-3,
                       name='Attention'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Flatten())
    model.add(Dense(3, activation='softmax',kernel_regularizer=regularizers.l2(1e-3),activity_regularizer=regularizers.l1(1e-4)))
    model.summary()
    model.compile(optimizer='Nadam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    return model

model = get_lstm_model()
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
lrate = ReduceLROnPlateau(min_lr=0.00001)
model_train = model.fit(x_train, y_train, batch_size=500, epochs=20, validation_data=(x_test, y_test),callbacks=[early_stopping, lrate])

def plot():#画图
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.plot(model_train.history['categorical_accuracy'], c='g', label='train')
    plt.plot(model_train.history['val_categorical_accuracy'], c='b', label='validation')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Model accuracy')
    plt.show()
plot()

#model = load_model('lstm.h5', custom_objects=SeqSelfAttention.get_custom_objects())

y_test = np.argmax(y_test, axis=1)
y_pred = model.predict_classes(x_test)

def prf1(y_test,y_pred,kind):#计算precision、recall、f1指标
    f1 = f1_score(y_test, y_pred, average=kind)
    p = precision_score(y_test, y_pred, average=kind)
    r = recall_score(y_test, y_pred, average=kind)
    print(p, r, f1)

prf1(y_test,y_pred,'micro')
prf1(y_test,y_pred,'macro')

model.save('lstm.h5')

res = model.predict_classes(final_test)#对测试集进行预测
with open('results.txt', 'w', encoding='UTF-8') as f:
    for i in range(len(res)-1):
        f.write(str(res[i])+'\n')
    f.write(str(res[-1]))

# res = model.predict_classes(x_extra)
# with open('results_extra.txt', 'w', encoding='UTF-8') as f:
#     for i in range(len(res)-1):
#         f.write(str(res[i])+'\n')
#     f.write(str(res[-1]))


