import random
import csv
import re
import numpy as np
from nltk.stem.snowball import SnowballStemmer
import xml.etree.ElementTree as ET
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC

from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Model
from keras.utils import np_utils


class LinearClf:
    def __init__(self, D, random_initilizer):
        '''
        D: #features
        '''
        self.dim = D
        self.w = np.array(random_initilizer)
        #self.w = np.random.rand(self.dim)
        #for i in range(self.dim):
        #    self.w[i] = (self.w[i] - 0.5) / 50

    def SVM(self, data, gamma, C, thresh, train = True, max_epoch = 100):
        X_train = np.array(data[0])
        y_train = np.array(data[1])
        if train:
            X_val = data[2]
            y_val = data[3]
        object_change = 100000
        last_obj = 0.0
        current_obj = 0.0
        epoch = 0
        while(object_change > thresh * current_obj and epoch < max_epoch):
            X_train, y_train = shuffle_list(X_train, y_train)
            epoch += 1
            gammat = gamma / (1+epoch)
            for i in range(len(y_train)):
                yi = np.inner(self.w, X_train[i])
                current_obj += np.inner(self.w, self.w)/2
                slack = 1 - y_train[i] * yi
                if slack >= 0:
                    current_obj += C * slack
                    self.w = (1-gammat) * self.w + gammat * C * y_train[i] * X_train[i]
                else:
                    self.w = (1-gammat) * self.w

            object_change = np.abs(current_obj - last_obj)
            last_obj = current_obj
            current_obj = 0.0

        p = 0
        r = 0
        f = 0
        a = 0
        if train:
            y_pred = []
            for i in range(len(y_val)):
                if np.inner(self.w, X_val[i]) > 0:
                    y_pred.append(1)
                else:
                    y_pred.append(-1)
            p,r,f,a = evaluatef1(y_pred, y_val)
        print(p,r,f,a)
        return p,r,f if train else None

    def Logistic(self, data, gamma, var, thresh, train = True):
        X_train = np.array(data[0])
        y_train = np.array(data[1])
        if train:
            X_val = data[2]
            y_val = data[3]
        object_change = 100000
        last_obj = 0.0
        current_obj = 0.0
        epoch = 0
        while(object_change> thresh * current_obj and epoch < 100):
            if epoch == 99:
                print("max_iter reached")
            X_train, y_train = shuffle_list(X_train, y_train)
            epoch += 1
            current_obj = 0
            for i in range(len(y_train)):
                mu = np.dot(self.w, X_train[i])
                current_obj += np.dot(self.w,self.w)/var+ np.log(1+np.exp(-y_train[i]*mu))
                grad = np.multiply(self.w, 2/var) + np.divide(np.multiply(X_train[i],-y_train[i]),(1+np.exp(y_train[i]*mu)))
                self.w = np.subtract(self.w, gamma * grad)

            object_change = np.abs(current_obj - last_obj)
            #print(object_change,last_obj,current_obj)
            last_obj = current_obj
            current_obj = 0.0

        p = 0
        r = 0
        f = 0
        a = 0
        if train:
            y_pred = []
            for i in range(len(y_val)):
                if np.dot(self.w, X_val[i]) > 0:
                    y_pred.append(1)
                else:
                    y_pred.append(-1)
            p,r,f,a  = evaluatef1(y_pred, y_val)
        print(p,r,f,a)
        return p,r,f if train else None

    def predict(self, X_test, y_test):
        y_pred = []
        for i in range(len(X_test)):
            yi = np.dot(self.w, X_test[i])
            if yi <= 0:
                y_pred.append(-1)
            else:
                y_pred.append(1)
        return evaluate(y_pred,y_test)


    def predictf1(self, X_test, y_test):
        y_pred = []
        for i in range(len(X_test)):
            yi = np.inner(self.w, X_test[i])
            if yi <= 0:
                y_pred.append(-1)
            else:
                y_pred.append(1)
        return evaluatef1(y_pred,y_test)


def generate_data(data_path):
    train_path = 'train.csv'
    test_path = 'test.csv'

    data = open(data_path, 'r')
    tree = ET.parse(data)
    root = tree.getroot()

    texts = []
    operations = []
    answers = []
    formulas = []

    for child in root:
        for subchild in child:
            if subchild.tag == "Question":
                texts.append(subchild.text)
            if subchild.tag == "SolutionType":
                operations.append(subchild.text)
            if subchild.tag == "Answer":
                answers.append(subchild.text)
            if subchild.tag == "Formula":
                formulas.append(subchild.text)

    datapairs = list(zip(texts,operations,answers,formulas))
    random.seed(13)
    random.shuffle(datapairs)
    with open(train_path,'w') as f:
        writer = csv.writer(f, delimiter=',')
        for line in datapairs[80:]:
            writer.writerow(line)

    with open(test_path,'w') as f:
        writer = csv.writer(f, delimiter=',')
        for line in datapairs[:80]:
            writer.writerow(line)

    X_train, y_train, z_train, f_train = zip(*datapairs[80:])
    X_test, y_test, z_test, f_test = zip(*datapairs[:80])

    return list(X_train), list(y_train), list(z_train),list(f_train), list(X_test), list(y_test), list(z_test),list(f_test)


def shuffle_list(*ls):
    l =list(zip(*ls))
    random.shuffle(l)
    return zip(*l)

def build_vocab(docs):
    vocab = []
    for doc in docs:
        vocab += doc.split()
    vocab = list(set(vocab))
    w_to_i = {w:i for i,w in enumerate(vocab,1)} #enumerate from 1 to avoid confusion with zero padding
    i_to_w = {i:w for i,w in enumerate(vocab,1)}

    return w_to_i, i_to_w

def one_hot(docs,w_to_i):
    embeddings = []
    for doc in docs:
        embeddings.append([w_to_i[word] for word in doc.split() if word in w_to_i])

    return embeddings


def CNNLOG(X_train, y_train, X_test, y_test):
    one_hot_dict, inv_one_hot_dict = build_vocab(X_train)
    vocabulary_size = len(one_hot_dict)+1
    X_train_embedding = one_hot(X_train,one_hot_dict)
    X_test_embedding = one_hot(X_test,one_hot_dict)
    category_num = len(set(y_train))
    categories = list(set(y_train))
    o_to_i = {o:i for i,o in enumerate(categories)}
    i_to_o = {i:o for i,o in enumerate(categories)}
    y_train_numeric = []
    y_test_numeric = []
    for label in y_train:
        y_train_numeric.append(o_to_i[label])
    for label in y_test:
        y_test_numeric.append(o_to_i[label])

    y_train_distribution = np_utils.to_categorical(y_train_numeric, category_num)
    y_test_distribution = np_utils.to_categorical(y_test_numeric, category_num)

    # pad documents to max length
    max_length = 100
    X_train_embedding_padded = pad_sequences(X_train_embedding, maxlen=max_length, padding='post')
    X_test_embedding_padded = pad_sequences(X_test_embedding, maxlen=max_length, padding='post')


    X_shuffled, y_shuffled = shuffle_list(X_train_embedding_padded, y_train_distribution)
    length = len(X_shuffled)
    X_train_onehot = np.array(X_shuffled[:int(0.8*length)])
    X_dev_onehot = np.array(X_shuffled[int(0.8*length):])
    y_train_distribution = np.array(y_shuffled[:int(0.8*length)])
    y_dev_distribution = np.array(y_shuffled[int(0.8*length):])

    embedding_dim = 256
    filter_sizes = [3, 4, 5]
    num_filters = 64
    drop = 0.2

    epochs = 10
    batch_size = 128

    print("Creating Model...")
    inputs = Input(shape=(max_length,), dtype='int32')
    embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=max_length)(inputs)
    reshape = Reshape((max_length, embedding_dim, 1))(embedding)

    conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid', kernel_initializer='normal',
                activation='relu')(reshape)
    conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid', kernel_initializer='normal',
                activation='relu')(reshape)
    conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid', kernel_initializer='normal',
                activation='relu')(reshape)

    maxpool_0 = MaxPool2D(pool_size=(max_length - filter_sizes[0] + 1, 1), strides=(1, 1), padding='valid')(conv_0)
    maxpool_1 = MaxPool2D(pool_size=(max_length - filter_sizes[1] + 1, 1), strides=(1, 1), padding='valid')(conv_1)
    maxpool_2 = MaxPool2D(pool_size=(max_length - filter_sizes[2] + 1, 1), strides=(1, 1), padding='valid')(conv_2)

    concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
    flatten = Flatten()(concatenated_tensor)
    dropout = Dropout(drop)(flatten)
    output = Dense(units=category_num, activation='softmax')(dropout)

    model = Model(inputs=inputs, outputs=output)

    checkpoint = ModelCheckpoint('CNN.hdf5', monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
    adam = Adam(lr=2e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    #print(model.summary())
    print("Training Model...")
    model.fit(X_train_onehot, y_train_distribution, batch_size=batch_size, epochs=epochs, verbose=0, callbacks=[checkpoint],
         validation_data=(X_dev_onehot, y_dev_distribution))

    model.load_weights("CNN.hdf5")
    y_predict = model.predict(X_test_embedding_padded)
    y_predict = [i_to_o[np.argmax(i)] for i in y_predict]
    #print(y_predict)
    print("Operation acc: {}".format(model.evaluate(X_test_embedding_padded,y_test_distribution,verbose=0)[1]))
    return y_predict

def argument_predict_mul_div(text):
    texts = text.split('.')
    args = []
    for sentence in texts:
        if re.search('([Ee]ach|[Ee]very) (\w+)',sentence):
        # each/every indicate a unit quantity
            try: args.append(re.search('(\d+)',sentence).group(1))
            except: pass
            A0 = re.search('([Ee]ach|[Ee]very) (\w+)',sentence).group(2)
            # find the name of unit
            try:
                X_A0 = re.sub(A0, 'A0', text)
                # find the number of units
                try: args.append(re.search("(\d+) (.*)A0", X_A0).group(1))
                except: pass
            except: pass
    return args

def argument_predict_add_sub(text):
    args = []
    try:
        A0 = re.search("[Hh]ow (many|much)( more | )(\w+)",text).group(3)
        X_A0 = re.sub(A0,'A0',text)
        try:
            arg_m = re.finditer("(\d+)( | more | less )(A0|of them|than)", X_A0)
            for match in arg_m:
                args.append(match.group(1))
        except:pass
    except: pass
    return args

def rule_execute(X_test,y_predict,z_test):
    z_predict = []

    for i in range(len(X_test)):
        if y_predict[i] == 'Addition' or y_predict[i] == 'Subtraction':
            args = argument_predict_add_sub(X_test[i])
            if len(args) == 2:
                if y_predict[i] == 'Addition':
                    z_predict.append(float(args[0])+float(args[1]))
                elif y_predict[i] == 'Subtraction':
                    z_predict.append(abs(float(args[0])-float(args[1])))

            # cant find exactly 2 arguments
            else:
                z_predict.append('NA')

        # Multiplication or division
        else:
            stemmer = SnowballStemmer("english")
            X_test_stem = ' '.join([stemmer.stem(plural) for plural in X_test[i].split()])
            args = argument_predict_mul_div(X_test_stem)

            if len(args) == 2:
                if y_predict[i] == 'Multiplication':
                    z_predict.append(float(args[1]) * float(args[0]))
                elif y_predict[i] == 'CommonDiv':
                    z_predict.append(float(args[1]) / float(args[0]))

            # cant exactly find 2 arguments
            else:
                z_predict.append('NA')

    correct = 0
    attempt = len(z_predict)
    for i in range(len(z_predict)):
        if z_predict[i] == 'NA':
            attempt -= 1
        elif float(z_predict[i]) == float(z_test[i]):
            correct += 1
        #else:
            #print(X_test[i], z_predict[i])

    print("Attempted:{0}, correct:{1}, Precision: {2} Recall: {3}".format(attempt, correct, float(correct/attempt),float(correct/len(z_predict))))

def operation_to_feature(operations):
    if operations == 'Addition':
        return [1,0,0,0]
    elif operations == 'Subtraction':
        return [0,1,0,0]
    elif operations == 'Multiplication':
        return [0,0,1,0]
    elif operations == 'CommonDiv':
        return [0,0,0,1]

def IsNumeric(w):
    try:
        float(w)
        return True
    except ValueError:
        return False

def get_question_obj(text):
    try:
        obj = re.search("[Hh]ow (many|much)( more | )(\w+)",text).group(3)
    except:
        obj = ''
    return obj

def get_question_unit(text):
    try:
        unit = re.search('([Ee]ach|[Ee]very) (\w+)',text).group(2)
    except:
        unit = ''
    return unit

def get_question_subj(text):
    for i in range(len(text)-1,-1, -1):
        if re.search('^[A-Z]',text[i]):
            return text[i]
    return None

def get_number_subj(text,index):
    for i in range(index,-1, -1):
        if text[i] == '.':
            return None
        if re.search('^[A-Z]',text[i]):
            return text[i]
    return None
def get_noun_of_number(text,index):
    return text[index+1]

def get_index_of_each(text):
    for i in range(len(text)):
        if text[i] == 'each' or text[i] == 'every':
            return i
    return -10

def get_index_of_word(text,word):
    for i in range(len(text)):
        if text[i] == word:
            return i
    return -10

def get_arg_label(number, formula):
    if formula == None:
        return None
    else:
        arg_match = re.finditer("([0-9.]+)", formula)
        args = []
        for arg_m in arg_match:
            args.append(arg_m.group(1))
        if float(args[0]) == float(number):
            return 1
        elif float(args[1]) == float(number):
            return 2
        else:
            return 0

def number_to_features(text,index,operator,formula):
    operation_features = operation_to_feature(operator)
    obj = get_question_obj(' '.join(text))
    unit = get_question_unit(' '.join(text))
    qsubj = get_question_subj(text)
    nsubj = get_number_subj(text,index)
    n_of_number = get_noun_of_number(text,index)
    i_each = get_index_of_each(text)
    i_obj = get_index_of_word(text,obj)
    i_unit = get_index_of_word(text,unit)
    dist_to_each = abs(index - i_each)
    dist_to_obj = abs(index - i_obj)
    dist_to_unit = abs(index - i_unit)
    same_subj = 1 if qsubj == nsubj else 0
    same_noun = 1 if n_of_number == obj else 0
    all_features = operation_features + [dist_to_each, dist_to_obj, dist_to_unit, same_subj, same_noun]
    label = get_arg_label(text[index], formula)
    if formula:
        return all_features, label
    else:
        return all_features

def NERpreprocessing(text):
    s = re.sub('How', 'how', text)
    s = re.sub('Each', 'each', s)
    s = re.sub('If', 'if', s)
    s = re.sub('There','there',s)
    s = re.sub('Now', 'now', s)
    return s.split()


def get_max_sequence(distribution):
    prob = {}
    for i in range(len(distribution)):
        prob[i] = {}
        for j in range(len(distribution)):
            if j != i:
                prob[i][j] = distribution[i][1]*distribution[j][2]
                for k in range(len(distribution)):
                    if k != i and k!= j:
                        prob[i][j] = prob[i][j] * distribution[k][0]
    probmax = 0
    max_arg0 = 0
    max_arg1 = 1
    for arg0 in prob.keys():
        arg1 = max(prob[arg0], key=prob[arg0].get)
        if max(prob[arg0].values()) > probmax:
            probmax = max(prob[arg0].values())
            max_arg0 = arg0
            max_arg1 = arg1

    return [max_arg0, max_arg1]

def argSVM_predictor(clf, sample_feature):
    distribution = []
    for k in sample_feature.keys():
        d = clf.predict_proba(np.array(sample_feature[k]['feature']).reshape(1,-1))
        distribution.append(d[0])

    args_index = get_max_sequence(distribution)
    return args_index


def text_to_features(samples,operations,formulas, test = False, clf = None):
    features = []
    args = []
    for k in range(len(samples)):
        sample_feature = {}
        numbers = []
        s = NERpreprocessing(samples[k])
        for i in range(len(s)):
            if IsNumeric(s[i]):
                numbers.append(s[i])
                sample_feature[s[i]] = {}
                if formulas:
                    sample_feature[s[i]]['feature'], sample_feature[s[i]]['label'] = number_to_features(s,i,operations[k],formulas[k])
                else:
                    sample_feature[s[i]]['feature'] = number_to_features(s, i, operations[k], None)
        if test:
            args_index = argSVM_predictor(clf, sample_feature)
            args.append([numbers[args_index[0]],numbers[args_index[1]]])

        features.append(sample_feature)

    if test:
        return args

    return features

def feature_reformat(array_of_dic_of_dic):
    features = []
    labels = []
    for dic_of_dic in array_of_dic_of_dic:
        for k in dic_of_dic.keys():
            features.append(dic_of_dic[k]['feature'])
            try:
                labels.append(dic_of_dic[k]['label'])
            except: pass
    return features, labels

def argSVM(X, y):
    clf = SVC(probability = True)
    clf.fit(X, y)
    print("argSVM train acc {}".format(clf.score(X, y)))
    return clf

def evaluate_answer(truth, predict):
    if len(truth) == len(predict):
        correct = 0
        attempt = 0
        for i in range(len(truth)):
            if float(truth[i]) == float(predict[i]):
                correct += 1
        print("QAacc = {}".format(float(correct/len(truth))))
    else:
        print("Truth - predict length error.")

def execute(operations, args):
    if len(operations) == len(args):
        ans = []
        for i in range(len(operations)):
            if operations[i] == 'Addition':
                ans.append(float(args[i][0]) + float(args[i][1]))
            elif operations[i] == 'Subtraction':
                ans.append(abs(float(args[i][0]) - float(args[i][1])))
            elif operations[i] == 'Multiplication':
                ans.append(float(args[i][0]) * float(args[i][1]))
            elif operations[i] == 'CommonDiv':
                if float(args[i][1]) != 0.0:
                    ans.append(float(args[i][0]) / float(args[i][1]))
                else:
                    ans.append(0)
        return ans
    else:
        print("Operation - args length error.")

def main():
    data_path = './data.xml'
    X_train, y_train, z_train, f_train, X_test, y_test, z_test, f_test = generate_data(data_path)
    y_predict = CNNLOG(X_train, y_train, X_test, y_test)
    rule_execute(X_test,y_predict,z_test)
    num_features_dic = text_to_features(X_train,y_train,f_train)
    num_features, num_labels = feature_reformat(num_features_dic)
    clf = argSVM(num_features, num_labels)
    args = text_to_features(X_test,y_predict, formulas=None, test = True, clf = clf)
    z_predict = execute(y_predict,args)
    evaluate_answer(z_test,z_predict)
main()


