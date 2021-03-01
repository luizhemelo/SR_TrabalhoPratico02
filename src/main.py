"""
Luiz Henrique de Melo Santos - 2017014464
Recommendation Systems, 2020/2 - Rodrygo Santos

Usage:
$ python3 main.py content.csv ratings.csv targets.csv
"""

import sys  # receive input parameters
import json  # process content input
import numpy as np
import pandas as pd
from math import log
from gc import collect  # free unreferenced memory
from dateutil.parser import parse


"""
DATA IMPORT
"""
def read_content(path):
    content = dict()
    with open(path) as f:
        for line in f.readlines()[1:]:
            item = line.split(',{')
            itemId = item[0]
            data = json.loads('{' + item[1])
            content[itemId] = data.get('Year', '').replace(' ', '') + ';;'
            content[itemId] += data.get('Title', '') + ';;'
            content[itemId] += data.get('Rated', '').replace(' ', '') + ';;'
            content[itemId] += data.get('Released', '').replace(' ', '') + ';;'
            content[itemId] += data.get('Director', '').replace(' ', '') + ';;'
            content[itemId] += data.get('Genre', '') + ';;'
            content[itemId] += data.get('Runtime', '').replace(' ', '') + ';;'
            content[itemId] += data.get('Writer', '').replace(' ', '') + ';;'
            content[itemId] += data.get('Actors', '').replace(' ', '') + ';;'
            content[itemId] += data.get('Language', '').replace(' ', '') + ';;'
            content[itemId] += data.get('Awards', '').replace(' ', '') + ';;'
            content[itemId] += data.get('Poster', '').replace(' ', '') + ';;'
            content[itemId] += data.get('Metascore', '').replace(' ', '') + ';;'
            content[itemId] += data.get('Country', '') + ';;'
            content[itemId] += data.get('imdbRating', '') + ';;'
            content[itemId] += data.get('Type', '') + ';;'
            content[itemId] += data.get('Plot', '')
    df = pd.DataFrame(content, index=[0]).transpose()  # create pandas.DataFrame
    df['itemId'] = df.index
    df = pd.concat([df, df[0].str.split(';;', expand=True)], axis=1, ignore_index=True)
    df = df.drop([0,1], axis=1)
    df = df.reset_index()
    df.columns = ['itemId','Year','Title','Rated','Released','Director','Genre','Runtime','Writer','Actors',
                  'Language','Awards','Poster','Metascore','Country','imdbRating','Type','Plot']
    return df

content = read_content(sys.argv[1])
ratings = pd.read_csv(sys.argv[2])

# separate first column in ratings
sep = ratings['UserId:ItemId'].str.split(':', expand=True)
sep.columns = ['UserId', 'ItemId']
ratings = pd.concat([ratings, sep], axis=1)
# selete columns
ratings = ratings.drop(['UserId:ItemId'], axis=1)


"""
FEATURE EXTRACTION
"""
def feature_extraction(content):
    ln = content.shape[0]
    content = content.replace(['N/A', ''], 0)

    # -- Year - transform the elements to numeric categories
    content['Year'] = pd.to_numeric(content['Year'], downcast='integer')

    # -- Rated - One-Hot Encoding
    rated = pd.get_dummies(content['Rated'], prefix='Rated')
    content = pd.concat([content, rated], axis=1)

    # -- Released - creates 'year', 'month' and 'day' features
    content['Released_year'] = 0
    content['Released_month'] = 0
    content['Released_day'] = 0
    for i in range(ln):
        date = str(content['Released'][i])
        if (date!='0'):
            prs = parse(date)
            content.loc[i, 'Released_year'] = prs.year
            content.loc[i, 'Released_month'] = prs.month
            content.loc[i, 'Released_day'] = prs.day

    # -- Runtime - creates 'hour' and 'minutes' features
    content['Runtime_min'] = 0
    content['Runtime_hour'] = 0
    for i in range(ln):
        runtime = str(content['Runtime'][i])
        if (runtime!='0'):
            runtime = runtime.split('h')
            if len(runtime)>1:
                content.loc[i, 'Runtime_min'] = int(runtime[1][:(len(runtime[1])-3)])
                content.loc[i, 'Runtime_hour'] = int(runtime[0])
            else:
                tm = int(runtime[0][:(len(runtime[0])-3)])
                minutes = tm%60
                hour = (tm-minutes)/60
                content.loc[i, 'Runtime_min'] = minutes
                content.loc[i, 'Runtime_hour'] = hour

    # -- Languages - categorizes the elements
    for i in range(ln):
        cont = content['Language'][i]
        if cont=='English':
            content.loc[i, 'Language'] = 1
        elif cont=='French':
            content.loc[i, 'Language'] = 2
        elif cont=='Japanese':
            content.loc[i, 'Language'] = 3
        elif cont=='Spanish':
            content.loc[i, 'Language'] = 4
        else:
            content.loc[i, 'Language'] = 0
    content['Language'] = pd.to_numeric(content['Language'], downcast='integer')

    # -- Metascore - transform the elements to numeric values
    content['Metascore'] = pd.to_numeric(content['Metascore'], downcast='integer')

    # -- Country - categorizes the elements
    for i in range(ln):
        cont = content['Country'][i]
        if cont=='USA':
            content.loc[i, 'Country'] = 1
        elif cont=='UK':
            content.loc[i, 'Country'] = 2
        elif cont=='India':
            content.loc[i, 'Country'] = 3
        elif cont=='Japan':
            content.loc[i, 'Country'] = 4
        else:
            content.loc[i, 'Country'] = 0
    content['Country'] = pd.to_numeric(content['Country'], downcast='integer')

    # -- imdbRating - transform the elements to numeric categories
    content['imdbRating'] = pd.to_numeric(content['imdbRating'], downcast='integer')

    # -- Type - categorizes the elements
    content['Type'] = content['Type'].replace(['movie'], 1)
    content['Type'] = content['Type'].replace(['episode'], 2)
    content['Type'] = content['Type'].replace(['series'], 3)
    
    # --  Combining remaining categorical features
    for i in range(ln):
        content.loc[i, 'combinedFeatures'] = str(content['Genre'][i]) + ' ' +  str(content['Actors'][i])
    
    # -- Dropping columns
    content = content.drop(['Rated', 'Released', 'Runtime', 'Awards', 'Poster', 'Title',
                            'Director', 'Genre', 'Writer', 'Actors', 'Plot'], axis=1)
    
    return content

content = feature_extraction(content)


"""
TF-IDF APPROACH

ATENTION: TfIdf class based on source-code presented in: https://streamsql.io/blog/tf-idf-from-scratch
"""
class TfIdf:
    def __init__(self, content, length):
        data = [[(word.replace(',', '').replace('.', '').replace('(' , '').replace(')', ''))
                     for word in row.lower().split()]
                     for row in content['combinedFeatures']]
        self.data_len = len(data)
        
        # compute the tf values for each combine features in movies dataset
        tfDict = []
        for row in data:
            tfDict.append(self.computeReviewTFDict(row))
            
        # stores the review count dictionary
        countDict = self.computeCountDict(tfDict)
        
        # stores the idf dictionary
        self.idfDict = self.computeIDFDict(countDict)
        
        # compute tfidf for combined features in movies dataset
        tfidfDict = self.computeReviewTFIDFDict( self.computeIDFDict( self.computeCountDict(tfDict) ) )
        
        # stores the TF-IDF dictionaries
        tfidfDict = [self.computeReviewTFIDFDict(review) for review in tfDict]
        
        # create the matrix os tfidf values
        wordDict = sorted(countDict.keys())
        del self.data_len,self.idfDict
        self.tfidfVector = [self.computeTFIDFVector(review, wordDict)[:length] for review in tfidfDict]
    
    def computeReviewTFDict(self, review):
        """
        Returns a tf dictionary for each review whose keys are all
        the unique words in the review and whose values are their
        corresponding tf.
        """
        # Counts the number of times the word appears in review
        reviewTFDict = {}
        for word in review:
            if word in reviewTFDict:
                reviewTFDict[word] += 1
            else:
                reviewTFDict[word] = 1
        # Computes tf for each word
        for word in reviewTFDict:
            reviewTFDict[word] = reviewTFDict[word] / len(review)
        return reviewTFDict

    def computeCountDict(self, tfDict):
        """
        Returns a dictionary whose keys are all the unique words in
        the dataset and whose values count the number of reviews in which
        the word appears.
        """
        countDict = {}
        # Run through each review's tf dictionary and increment countDict's (word, doc) pair
        for review in tfDict:
            for word in review:
                if word in countDict:
                    countDict[word] += 1
                else:
                    countDict[word] = 1
        return countDict

    def computeIDFDict(self, countDict):
        """
        Returns a dictionary whose keys are all the unique words in the
        dataset and whose values are their corresponding idf.
        """
        idfDict = {}
        for word in countDict:
            idfDict[word] = log(self.data_len / countDict[word])
        return idfDict

    def computeReviewTFIDFDict(self, reviewTFDict):
        """
        Returns a dictionary whose keys are all the unique words in the
        review and whose values are their corresponding tfidf.
        """
        reviewTFIDFDict = {}
        #For each word in the review, we multiply its tf and its idf.
        for word in reviewTFDict:
            reviewTFIDFDict[word] = reviewTFDict[word] * self.idfDict[word]
        return reviewTFIDFDict

    def computeTFIDFVector(self, review, wordDict):
        """
        Returns a matrix as numpy.array whose each line corresponds to
        each item present in dataset
        """
        tfidfVector = [0.0] * len(wordDict)

        # For each unique word, if it is in the review, store its TF-IDF value.
        for i, word in enumerate(wordDict):
           if word in review:
                tfidfVector[i] = review[word]
        return tfidfVector
    
    def get_TFIDFVector(self):
        """
        Returns the features matrix created in 'computeTFIDFVector'
        """
        return self.tfidfVector

tfidfVector = TfIdf(content, length=15000).get_TFIDFVector()
collect()  # free unreferenced memeory
content.index = content['itemId']
content = content.drop(['itemId', 'combinedFeatures'], axis=1)
tfidfVector = np.concatenate((content.values,tfidfVector), axis=1)


"""
CREATING USERS/ITEMS VECTORS
"""
# dict of items vectors
itemsDict = dict()
i = 0
for itemID in content.index:
    itemsDict[itemID] = tfidfVector[i]
    i += 1
del tfidfVector

# dict of items classified by each user and its ratings
userItems = dict()
i = 0
for userID in ratings['UserId']:
    if userID in userItems:
        if ratings['ItemId'][i] not in userItems[userID]:
            userItems[userID].append([ratings['ItemId'][i], ratings['Prediction'][i]])
    else:
        userItems[userID] = [[ratings['ItemId'][i], ratings['Prediction'][i]]]
    i += 1
del ratings


"""
MAKING PREDICTIONS
"""
class SimilarityPredict:
    """
    Prediction of item scores from user id and item
    """
    
    def __init__(self, itemsDict):
        # create a cache for norm and for similarities values - improves execution time
        self.similarities = dict()
        self.norms = dict()
        for key in itemsDict.keys():
            self.norms[key] = np.linalg.norm(itemsDict[key])
    
    def predict(self, userID, itemID):
        num = 0
        div = 0
        if userID not in userItems:  # Cold-Start case
            rt = content['imdbRating'][itemID]
            if rt==0: return 6.53
            else: return rt
        for item in userItems[userID]:  # run the prediction formula
            st = itemID + item[0]
            if st not in self.similarities:  # checks if similarity has already been calculated
                self.similarities[st] = np.dot(itemsDict[itemID],itemsDict[item[0]]) / (self.norms[itemID]*self.norms[item[0]])
            num += self.similarities[st] * item[1]
            div += self.similarities[st]
        return num/div  # return the weighted avarage

# reading targets dataset
targets = pd.read_csv(sys.argv[3])
# separate first column in targets
sep = targets['UserId:ItemId'].str.split(':', expand=True)
sep.columns = ['UserId', 'ItemId']
targets = pd.concat([targets, sep], axis=1)
# selete columns
targets = targets.drop(['UserId:ItemId'], axis=1)

# generate output
cs = SimilarityPredict(itemsDict)
sys.stdout.write('UserId:ItemId,Prediction\n')
for i in targets.index:
    userID = targets['UserId'][i]
    itemID = targets['ItemId'][i]
    sys.stdout.write(userID)
    sys.stdout.write(':')
    sys.stdout.write(itemID)
    sys.stdout.write(',')
    sys.stdout.write(str(cs.predict(userID, itemID)))  # make prediction for each input in targets
    sys.stdout.write('\n')
