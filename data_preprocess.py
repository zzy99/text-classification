import nltk
import re
import wordninja
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import spacy
import en_core_web_sm

#sw = stopwords.words("english") #不采用自带的过大的停用词表,而是一个小得多的
sw = ['a', 'about', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'have', 'how', 'in', 'is', 'it', \
      'll', 'of', 'o', 'on', 'or', 's', 't', 'that', 'the', 'this', 'to', 'was', 'what', 'when', 'where', 'who', 'will', 'with', 'the']

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
nlp = en_core_web_sm.load()

def replace_abbreviations(text):#去缩写
    texts = []
    for item in text:
        item = item.lower().replace("it's", "it is").replace("i'm", "i am").replace("he's", "he is").replace("she's",
                                                                                                             "she is") \
            .replace("we're", "we are").replace("they're", "they are").replace("you're", "you are").replace("that's",
                                                                                                            "that is") \
            .replace("this's", "this is").replace("can't", "can not").replace("don't", "do not").replace("doesn't",
                                                                                                         "does not") \
            .replace("we've", "we have").replace("i've", " i have").replace("isn't", "is not").replace("won't",
                                                                                                       "will not") \
            .replace("hasn't", "has not").replace("wasn't", "was not").replace("weren't", "were not").replace("let's",
                                                                                                              "let us") \
            .replace("didn't", "did not").replace("hadn't", "had not").replace("waht's", "what is").replace("couldn't",
                                                                                                            "could not") \
            .replace("you'll", "you will").replace("you've", "you have")
        item = item.replace("'s", "")
        texts.append(item)
    return texts

def clean_text(text):
    text_list = wordninja.split(text)#分词
    text_list = replace_abbreviations(text_list)
    texts = []
    for word in text_list:#去符号
        word = word.replace("<br /><br />", "")
        word = re.sub("[^a-zA-Z]", " ", word.lower())
        texts.append(" ".join(word.split()))
    text_list = texts
    text_list = [word for word in text_list if word not in sw]#去停用词
    text_list = [lemmatizer.lemmatize(word, pos='v') for word in text_list]#词性还原
    text_list = [word for word in text_list if word not in sw]
    return ' '.join(text_list)

def constructDataset(path, flag=True):
    _set = []
    _label = []
    print('processing ' + path)
    with open(path,'r',encoding='UTF-8') as p:
        for line in p.readlines():
            s = line.split("<sep>")
            text = clean_text(s[0].rstrip('\n'))
            _set.append(text)
            if(flag):
                _label.append(int(s[1]))
    if(not flag):
        return _set
    else:
        return _set, _label

train_set, train_label = constructDataset('./train.txt')
test_set = constructDataset('./test.txt',False)
val_set, val_label = constructDataset('./self_test.txt')
extra_set = constructDataset('./extra.txt',False)

def saveDataset(data, path):
    print('saving ' + path)
    with open(path,'w',encoding='UTF-8') as f:
        for line in data:
            f.write(str(line)+'\n')

saveDataset(train_set, './data/train_set.txt')
saveDataset(train_label, './data/train_label.txt')
saveDataset(test_set, './data/test_set.txt')
saveDataset(extra_set, './data/extra_set.txt')
saveDataset(val_set, './data/val_set.txt')
saveDataset(val_label, './data/val_label.txt')