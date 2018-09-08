import xml.etree.cElementTree as et
import pandas as pd
import spacy

# https://stackoverflow.com/questions/50774222/python-extracting-xml-to-dataframe-pandas

file_path = 'train.xml'
dfcols = ['category', 'question_id', 'title']
root = et.parse(file_path)
questions = root.findall('.//question')
xml_data = [[question.get(dfcols[0]), question.get(dfcols[1]), question.get(dfcols[2])] for question in questions]
df_xml = pd.DataFrame(xml_data, columns=dfcols)
# print(df_xml)


dfcolsa = ['answer_id', 'group', 'isElectedAnswer', 'text']
answers = root.findall('.//answer')
xml_data_ans = [[answer.get(dfcolsa[0]), answer.get(dfcolsa[1]), answer.get(dfcolsa[2]), answer.get(dfcolsa[3])] for answer in answers]
df_xml_ans = pd.DataFrame(xml_data_ans, columns=dfcolsa)
# print(df_xml_ans)

# textblob
train = [(getattr(row, 'title'), getattr(row, 'question_id')) for row in df_xml.itertuples()]
from textblob.classifiers import NaiveBayesClassifier
#cl = NaiveBayesClassifier(train)
smalltrain = train[0:100] # about 7 min for 10k but more than 10 min when using classifier
cl = NaiveBayesClassifier(smalltrain)



# comparing with questions
nlpl = spacy.load('en_core_web_lg')
test = nlpl('How can i increase usb voltage?') # similar to #28
train1 = [(getattr(row, 'title'), row.Index) for row in df_xml.itertuples()]
smalltrain1 = train1[0:100]
results = []
for sentence in smalltrain1:
    doc0 = nlpl(sentence[0])
    results.append((test.similarity(doc0), sentence[1]))
# sorted(results, key=lambda res: res[0], reverse=True)
sorted(results, key=lambda res: res[0], reverse=True)[0:5] # top 5



# comparing with answers
train2 = [(getattr(row, 'text'), getattr(row, 'answer_id'), getattr(row, 'group')) for row in df_xml_ans.itertuples()]
smalltrain2 = train2[0:1000]
results = []
for sentence in smalltrain2:
    doc0 = nlpl(sentence[0])
    results.append((test.similarity(doc0), sentence[1], sentence[2])) # about 2 min
sorted(results, key=lambda res: res[0], reverse=True)[0:5] # top 5



# using en_vectors_web_lg
nlpv = spacy.load('en_vectors_web_lg')
testv = nlpv('How can i increase usb voltage?')
resultsv = []
for sentence in smalltrain2:
    doc0 = nlpv(sentence[0])
    resultsv.append((testv.similarity(doc0), sentence[1], sentence[2]))
sorted(resultsv, key=lambda res: res[0], reverse=True)[0:5] # results are identical to sunig en_core_web_lg, but ~half time



# https://www.kaggle.com/zackakil/nlp-using-word-vectors-with-spacy-cldspn
from spacy.lang.en.stop_words import STOP_WORDS
def remove_stop_words(text):
    return ' '.join([word for word in text.split(' ') if word.lower() not in STOP_WORDS])



# https://stackoverflow.com/questions/49205736/in-spacy-how-can-i-efficiently-compare-the-similarity-of-one-document-to-all-oth?rq=1
# https://stackoverflow.com/questions/51651934/python-spacy-similarity-without-loop
# https://stackoverflow.com/questions/49179118/spacy-how-to-save-the-text-preprocessed-to-save-time-in-the-future



#HTML tag and newline remover
import re

def remove_tags(text):
    return re.compile(r'<[^>]+>').sub('', text)

def remove_newline(text):
    return re.compile(r'[\n\r\t]').sub('', text)



testans_df = pd.read_csv('testAnswers.csv')
testans = [(getattr(row, 'text'), getattr(row, 'answer_id'), getattr(row, 'group')) for row in testans_df.itertuples()]
nlpv = spacy.load('en_vectors_web_lg')
testv = nlpv(remove_stop_words(remove_newline(remove_tags('Is DRY the enemy of software project management?'))))
resultsv = []
for text in testans:
    doc0 = nlpv(remove_stop_words(remove_newline(remove_tags(text[0]))))
    resultsv.append((testv.similarity(doc0), text[1], text[2]))
sorted(resultsv, key=lambda res: res[0], reverse=True)[0:5]



# about 8 min 1 question for whole test answers data of 76830 answers in 26559 groups



testans = []
i = 0
text = ''
for row in testans_df.itertuples():
    if (i == getattr(row, 'group')):
        text += getattr(row, 'text') # need to find way to append last one
    else:
        testans.append((text, i))
        text = ''
        text += getattr(row, 'text')
        i += 1

# https://stackoverflow.com/questions/17841149/pandas-groupby-how-to-get-a-union-of-strings
testans_gb = testans_df.groupby('group').agg(lambda col: ''.join(col))
resultsgb = []
for row in testans_gb.itertuples():
    doc0 = nlpv(remove_stop_words(remove_newline(remove_tags(getattr(row, 'text')))))
    resultsgb.append((testv.similarity(doc0), row.Index))
sorted(resultsgb, key=lambda res: res[0], reverse=True)[0:5]





testques_df = pd.read_csv('testQuestions.csv')
testques = [(getattr(row, 'title'), getattr(row, 'question_id')) for row in testques_df.itertuples()]
ansv = nlpv(remove_stop_words(remove_newline(remove_tags(testans_df.text[0]))))
resultsr = []
for title in testques:
    doc0 = nlpv(remove_stop_words(remove_newline(remove_tags(title[0]))))
    resultsr.append((ansv.similarity(doc0), title[1]))
sorted(resultsr, key=lambda res: res[0], reverse=True)[0:5]




testanscat_df = pd.read_csv('category_test_answers.csv')
category = testques_df.category[0] # bitcoin
filtered = testanscat_df[testanscat_df['Category'].str.contains(category)]
testv = nlpv(remove_stop_words(remove_newline(remove_tags(testques_df.title[0])))) # 'Generating Public & Private key pairs for cracking Bitcoin'
resultsc = []
for row in filtered.itertuples():
    doc0 = nlpv(remove_stop_words(remove_newline(remove_tags(getattr(row, 'CleanText')))))
    resultsc.append((testv.similarity(doc0), getattr(row, 'ID'), getattr(row, 'GroupID')))
sorted(resultsc, key=lambda res: res[0], reverse=True)

# filtered only has 3 answers and none of them are from the correct group


