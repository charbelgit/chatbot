import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy

import tensorflow
import tflearn
import random
import json
import pickle
import tensorflow_addons

with open("schemes.json") as file:
    data = json.load(file)
try:
    with open("data.pickle", "rb") as f:
            words,labels,training,output = pickle.load(f)

except:

    words = []
    labels =[]
    docs_x=[]
    docs_y=[]
    for intent in data ['schemes']:
        for pattern in intent['patterns']:
            #let's tokenize
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])
            if intent["tag"] not in labels:
                labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w not in "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []
    out_empty = [0 for _ in range(len(labels))]

    for doc in enumerate(docs_x):
        bag = []
        wrds = [stemmer.stem(w) for w in doc]
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])]=1
        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)
    with open("data.pickle", "wb") as f:
            pickle.dump(words,labels,training,output,f )


tensorflow.reset_default_graph()
#neural networks
net = tflearn.imput_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,len(output[0]), activation ="softmax")
net=tflearn.regression(net)
model= tflearn.DNN(net)

try:
    model.load("model.tflearn")

except:

    model.fit(training, output, n_epoch=1000, batch_size= 8, show_metric=True )
    model.save("model.tflearn")

def bag (s,words):

    bag= [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words= [stemmer.stem(word.lower()) for word in s_words]

    for S in s_words:
        for i, w in enumerate(words):
            if w == S:
                bag[i]=1
        return numpy.array(bag)

def chat():
    print("Say Hello ... End the chat by typing quit")
    while True:
        input = input ("You: ")
        if input.lower() == "quit":
            break
        result = model.predict([bag(input, words)])[0] # returns a list of probabilities
        result_index = numpy.argmax(result) #get the index of the highest value
        tag = labels[result_index]
        if result[result_index]>0.6667:

            for tags in data['schemes']:
                if tags['tag'] == tag:
                    responses = tags ['responses']
            print (random.choice(responses))
        else:
            print ("Sorry, I don't understand what you're saying")
chat()
