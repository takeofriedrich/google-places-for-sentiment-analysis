from utils import Helper
from utils import LexicalClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
import time
from sklearn.neural_network import MLPClassifier
import sys

class LexicalClassifier1(LexicalClassifier):

    def __init__(self) -> None:
        super().__init__()

    def parseLex(self, lex_path:str):
        self.lexicon = {}
        entrys = [x.replace('\n','').split(',') for x in open(lex_path,'r').readlines()]
        for entry in entrys[1:]:
            if entry[1] not in self.lexicon.keys():
                self.lexicon[entry[1]] = int(entry[0])

    def classify(self,texto):
        sent = 0
        countN = 0
        summ = 0
        has_period = False
        sentences = []
        classified_list = []

        doc = Helper().nlp(texto)
        
        for token in doc:
            if token.is_punct:
                has_period = True

        if(has_period == False):
            sentences.append(doc)

        if(has_period):
            inicio = 0
            fim = 0
            for token in doc[1:]:
                if token.is_punct: 
                    fim = token.i
                    sentences.append(doc[inicio:fim])
                    inicio = token.i+1
                elif token.is_sent_end:
                    fim = token.i+1
                    sentences.append(doc[inicio:fim])

        for sentenca in sentences:
            for tk in sentenca:
                if(tk.text == 'não'):
                    countN += 1
                elif tk.text in self.lexicon.keys() and self.lexicon[tk.text] == 1:
                    sent += 1
                elif tk.text in self.lexicon.keys() and self.lexicon[tk.text] == -1:
                    sent -= 1
            if(countN != 0):
                sent = sent * -1
                classified_list.append(sent)
                sent = 0
                countN = 0
            else:
                classified_list.append(sent)
                sent = 0
                countN = 0
        
        for classificada in classified_list:
            summ += classificada

        resposta = 1 if summ > 0 else -1 if summ < 0 else 0
        return resposta 

if len(sys.argv) != 2 or sys.argv[1].lower() not in ['lex','ml','dl']:
    print('Run as python3 main.py <EXECUTION MODE>')
    print('Where execution mode can be:')
    print('\tLEX for lexical approaches')
    print('\tML for machine learning approaches')
    print('\tDL for deep learning approaches')
    sys.exit(-1)

start = time.time()

if sys.argv[1] == 'LEX':

    print('Running test with lexical approaches')

    Helper().trainAndTestLexicalApproaches(LexicalClassifier1(),start)

elif sys.argv[1] == 'ML':

    print('Running test with machine learning approaches')

    classifiers = {
        'Logistic Regression' : LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg', multi_class='multinomial', n_jobs=-1, random_state=40),
        'Naive Bayes' : MultinomialNB(),
        'SVM' : svm.SVC(decision_function_shape='ovo')
    }

    Helper().trainAndTestSeveralMLClassifiers(classifiers,start)

else:

    print('Running test with deep learning approaches')

    classifiers = {
        # 'tanH': MLPClassifier(activation='tanh', solver='adam', alpha=1e-5, hidden_layer_sizes=(101,), random_state=1, max_iter=900),
        # 'ReLu': MLPClassifier(activation='relu', solver='adam', alpha=1e-5, hidden_layer_sizes=(101,), random_state=1, max_iter=900),
        'Sigmoid': MLPClassifier(activation='logistic', solver='adam', alpha=1e-5, hidden_layer_sizes=(101,), random_state=1, max_iter=900)
    }

    Helper().trainAndTestSeveralMLClassifiers(classifiers,start)
