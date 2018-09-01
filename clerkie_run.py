
# coding: utf-8

# In[50]:


import tflearn
import tensorflow as tf
import pickle
import nltk
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split
from nltk.corpus import wordnet
import answer_questions
import numpy as np
stemmer = SnowballStemmer("english")


# In[65]:


with open('data_label.pickle','rb') as f:
    training = pickle.load(f)
with open('word_token.pickle','rb') as f:
    words = pickle.load(f)
word_set = set(words)
    
data = list(training[:, 0])
label = list(training[:, 1])

# reset underlying graph data
tf.reset_default_graph()
# Build neural network
net = tflearn.input_data(shape=[None, len(data[0])])
net = tflearn.fully_connected(net, 64, activation='relu')
net = tflearn.fully_connected(net, 64, activation='relu')
net = tflearn.fully_connected(net, len(label[0]), activation='softmax')
net = tflearn.regression(net)
 
# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')


# In[66]:


model.load('tf_nn.model')


# In[67]:


def get_tf_record(sentence):
    global words
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    # bag of words
    bow = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bow[i] = 1

    return(np.array(bow))


# In[68]:


def add_question_unlabel(user_input):
    f = open("unlabeled.csv", "a")
    f.write("{}\n".format(user_input))
    f.close()
    print("sorry, I'm still learning :p, but I've saved your question for future learning")
def add_question_pseudo_label(class_name, user_input):
    f = open("pseudo_label.csv", "a")
    f.write("{},{}\n".format(class_name, user_input))
    f.close()


# In[69]:


# capture entities in questions belong to category 1.
# extract bank names, checking/saving, account number
def cap_entities(user_input):
    token = nltk.word_tokenize(user_input)
    tagged = nltk.pos_tag(token)
    bank_set = set(['boa','bankofamerica','chase','citi'])
    bank_id_type = ['','','']
    potential_bank_name = ''
    for i,x in enumerate(tagged):
        if x[0] in bank_set:
            bank_id_type[0] = x[0]
        elif x[1] == 'CD':
            bank_id_type[1] = x[0]
        elif x[0] == 'checking' or x[0] == 'saving' or x[0] == 'credit':
            bank_id_type[2] = x[0]

    return bank_id_type


# capture entities in questions belong to category 3. 
# determine if the user is asking about buying houses
# return price only if user specified a price
# and the question is about buying house
def house_keyword(user_input):
    token = nltk.word_tokenize(user_input)
    tagged = nltk.pos_tag(token)
    price = '0'
    flag = False
    similarity_threshold = 0.8
    house_set = ['house',
                 'condo',
                 'pad',
                 'crib',
                 'apartment',
                 'residence',
                 'mansion'
                ]
    for x in tagged:
        if x[1] == 'CD':
            price = x[0]
        if 'NN' in x[1]:
            try:
                w1 = wordnet.synset('{}.n.01'.format(x[0]))
            except:
                continue
            for word in house_set:
                w2 = wordnet.synset('{}.n.01'.format(word))
                if w1.wup_similarity(w2) > similarity_threshold:
                    flag = True
                    break
    return price if flag else '-1'


# In[70]:


def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word) for word in sentence_words]
    return sentence_words


# In[71]:


def classify(text):
    
    classes = ['1.check balance','2.check budget','3.house affordability',
                '4.loan', '5.mortgage FAQ', '6.Check spending']
    ERROR_THRESHOLD = 0.65
    result = model.predict([get_tf_record(text)])
    prediction = np.argmax(result)
    prob = result[0][prediction]
    
    if prob < ERROR_THRESHOLD:
        prediction = 6
        print('class:','unidentified','|| prob:','N/A')
    else:
        print('class:',classes[prediction],'|| prob:',prob)
    return prediction, prob


# In[72]:


#Main function
def ask_clerkie():
    print("\nXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    user_input = input("what's your financial question? (type 'quit' to exit.) \n\n").lower()
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    if 'hello' in user_input:
        print('Hi :D, do you have a financial question?')
        return True
    #quitting outer loop
    elif user_input == 'quit':
        print('Thanks for using clerkie :D')
        print("\nXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        return False
    
    #if user input nothing
    elif len(user_input) == 0:
        print('Ask Me a question :D')
        return True
    user_token = clean_up_sentence(user_input)
    related_token = 0
    
    
    #is user question relatable?
    related_threshold = 0.2
    #check how many tokens are related to financial questions
    #also calculate the related percentage
    for token in user_token:
        if token in word_set:
            related_token += 1
    
    #related question, and question length more than 2 words
    if related_token/len(user_token) >= related_threshold and len(user_token) >= 2:
        question_class, prob = classify(user_input)
        
        #this means the question is not recognized by Clerkie
        #ask user's help to label the question
        #and save to file for future learning
        if question_class == 6:
            add_question_unlabel(user_input)
                
        
        else:
            if prob >=0.85:
                add_question_pseudo_label(question_class, user_input)
            
            #question that classified as category 1
            if question_class == 0:
                bank_id_type = cap_entities(user_input)
                alternative_name = {'bankofamerica':'boa'}
                if bank_id_type[0] in alternative_name:
                    bank_id_type[0] = alternative_name[bank_id_type[0]]
                answer_questions.get_balance(bank_id_type)

            #question that classified as category 2
            elif question_class == 1:
                answer_questions.get_budget()

            #question that classified as category 3
            elif question_class == 2:
                price = house_keyword(user_input)

                #question about buying things other than house
                if price == '-1':
                    print("sorry, I can only help with house affordability :p")

                #user didn't specify a price in the question
                elif price == '0':
                    print("sorry, please provide house price so I can help :p")
                else:
                    answer_questions.is_affordable(price)
            elif question_class == 3:
                answer_questions.loan_question()


            elif question_class == 4:
                answer_questions.mortgage_FAQ()


            elif question_class == 5:
                answer_questions.spending()
    else:
        print("Please ask financial related question (more than 2 words) :p")
    return True
    


# In[64]:


flag = True
while flag:
    flag = ask_clerkie()

