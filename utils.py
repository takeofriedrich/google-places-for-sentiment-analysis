import spacy
import pandas as pd
from abc import ABC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold
import time

class LexicalClassifier(ABC):

    '''
    Must return a value in {-1,0,1}
    '''
    def classify(sent):
        pass


class Helper:

    __instance = None

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
            cls.nlp = spacy.load('pt_core_news_lg')
            cls.nlp.add_pipe('emoji',first=True)
            print('Helper created')
        return cls.__instance

    @staticmethod
    def loadCleanCorpus() -> pd.DataFrame:
        return pd.read_csv('linguistics-resources/corpus.csv',sep='|')

    @staticmethod
    def preprocessLEX(sentence: str, strFormat=False) -> list:
        preprocessed = [tk.text for tk in Helper.nlp(sentence.lower()) if (tk.is_alpha or tk.is_punct or tk.is_quote)]
        if strFormat:
            string = ''
            for tk in preprocessed:
                string += tk + ' '
            return string.strip()
        else:
            return preprocessed

    @staticmethod
    def preprocessL(sentence: str, strFormat = False) -> list:
        preprocessed = [tk.lemma_ for tk in Helper.nlp(sentence.lower()) if (tk.is_alpha or tk.is_punct or tk.is_quote) and (not tk.is_stop or tk.text in ['não','n'])]
        if strFormat:
            string = ''
            for tk in preprocessed:
                string += tk + ' '
            return string.strip()
        else:
            return preprocessed

    @staticmethod
    def preprocessN(sentence: str, strFormat = False) -> list:
        doc = Helper.nlp(sentence.lower())
        fstToken = doc[0]
        string = ''
        if strFormat:
            for tk in doc[1:]:
                if fstToken.lower_ in ['não','nao','n'] and (tk.is_alpha or tk._.is_emoji):
                    string += 'NOT_{} '.format(tk.lemma_)
                elif tk.lower_ in ['não','nao','n']:
                    pass
                else:
                    string += tk.lemma_ + ' '
                fstToken = tk
            return string.strip()
        else:
            pass

    @staticmethod
    def get_metrics(y_test, y_predicted):  
        precision = precision_score(y_test, y_predicted, pos_label=None,
                                        average='weighted')             
        recall = recall_score(y_test, y_predicted, pos_label=None,
                                average='weighted')
        f1 = f1_score(y_test, y_predicted, pos_label=None, average='weighted')
        accuracy = accuracy_score(y_test, y_predicted)
        return accuracy, precision, recall, f1

    @staticmethod
    def lexicalValidation(classifier:LexicalClassifier,df):
        df['predicted'] = df.apply(lambda row:classifier.classify(row['text']),axis=1)
        accuracy, precision, recall, f1 = Helper.get_metrics(df['pol'], df['predicted'])
        return pd.DataFrame({
            'accuracy':[accuracy],
            'precision':[precision],
            'recall':[recall],
            'f1':[f1]
        })

    @staticmethod
    def crossvalidation(classificador,df):

        test = df[0:int(len(df)/10)].copy(deep=True)
        test = test.reset_index()

        train = df[int(len(df)/10)+1:].copy(deep=True)
        train = train.reset_index()

        count_vectorizer = CountVectorizer()
        
        target = 'pol'
        y = train[target]

        kf = KFold(n_splits=10, random_state=1066, shuffle=True)

        accuracys = []
        precisions = []
        recalls = []
        f1s = []

        for train_index, _ in kf.split(train):
            X_train = train.iloc[train_index, :]['text'].to_list()
            y_train = y[train_index].to_list()

            X_train_counts = count_vectorizer.fit_transform(X_train)
            X_test_counts = count_vectorizer.transform(test['text'])

            
            classificador.fit(X_train_counts, y_train)
            y_predicted_counts = classificador.predict(X_test_counts)

            acuracia, precisao, recall, f1 = Helper.get_metrics(test['pol'], y_predicted_counts)

            accuracys.append(acuracia)
            precisions.append(precisao)
            recalls.append(recall)
            f1s.append(f1)

        return pd.DataFrame({
            'accuracy':accuracys,
            'precision':precisions,
            'recall':recalls,
            'f1':f1s
        })

    @staticmethod
    def trainAndCalculateMetrics(classifier, classifierName,corpus,lex=False):
        print('\nTraining classifier with {}...'.format(classifierName))
        if lex:
            metrics = Helper().lexicalValidation(classifier,corpus)
        else:
            metrics = Helper().crossvalidation(classifier,corpus)

        print('Accuracy: ',metrics['accuracy'].mean(), ' - Std: ',metrics['accuracy'].std())
        print('Precision: ',metrics['precision'].mean(), ' - Std: ',metrics['precision'].std())
        print('Recall: ',metrics['recall'].mean(), ' - Std: ',metrics['recall'].std())
        print('F1: ',metrics['f1'].mean(), ' - Std: ',metrics['f1'].std())
        print()

    @staticmethod
    def trainAndTestLexicalApproaches(classifier,start):
        liwc_lexicon_filtred_path = 'linguistics-resources/liwc-filtred.txt'
        dsl_path = 'linguistics-resources/dsl.txt'

        print('Loading cleaned corpus')
        corpus = Helper().loadCleanCorpus()
        corpus['text'] = corpus['text'].apply(lambda row:Helper().preprocessLEX(row,strFormat=True))

        print('Parsing LIWC lexicon')
        classifier.parseLex(liwc_lexicon_filtred_path)
        Helper().trainAndCalculateMetrics(classifier,'LIWC-LEX',corpus,lex=True)
        print('Total time of execution', time.time()-start)

        print('Parsing DSL')
        classifier.parseLex(dsl_path)
        Helper().trainAndCalculateMetrics(classifier,'DSL',corpus,lex=True)
        print('Total time of execution', time.time()-start)

    @staticmethod
    def trainAndTestSeveralMLClassifiers(classifiers,start):
        corpus = Helper().loadCleanCorpus()
        print('Corpus loaded')
        corpus['text'] = corpus['text'].apply(lambda row: Helper().preprocessL(row, strFormat=True))
        print('Corpus preprocessed with LN strategy')
        print(corpus['pol'].value_counts())

        print('\n------------------------------------------------------------')
        for i in classifiers.keys():
            Helper().trainAndCalculateMetrics(classifiers[i], i, corpus)
            print('Total time of execution', time.time()-start)

        corpus = Helper().loadCleanCorpus()
        print('Corpus reloaded')
        corpus['text'] = corpus['text'].apply(lambda row: Helper().preprocessN(row, strFormat=True))
        print('Corpus preprocessed with LN strategy')
        print(corpus['pol'].value_counts())

        print('\n------------------------------------------------------------')
        for i in classifiers.keys():
            Helper().trainAndCalculateMetrics(classifiers[i], i, corpus)
            print('Total time of execution', time.time()-start)