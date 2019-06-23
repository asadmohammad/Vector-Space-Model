from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import numpy.linalg as LA

corpus = []
for i in range(0,50):
    f = open(str(i+1) + ".txt", "r")
    string = f.read()
    corpus.append(string)
    del string

query = input("Enter Query: ")
test_set = [query] #Query
train_set = [] #Documents
for doc in corpus:
    train_set.append(doc)

stop_words = ["a","is","the","of","all","and","to","can","be","as","once","for","at","am",
              "are","has","have","had","up","his","her","in","on","no","we","do"]

vectorizer = CountVectorizer(stop_words=stop_words)
transformer = TfidfTransformer()

trainVectorizer = vectorizer.fit_transform(train_set).toarray()
testVectorizer = vectorizer.transform(test_set).toarray()

#Cosine Similarity calculation
cx = lambda a, b: round(np.inner(a,b)/(LA.norm(a)*LA.norm(b)),16)

result_set = []
for vector in trainVectorizer:
    for testV in testVectorizer:
        cosine = cx(vector,testV)
        if cosine > 0:
            result_set.append(cosine)
        else:
            result_set.append(0)

transformer.fit(trainVectorizer)
print()
transformer.fit(testVectorizer)
tfidf = transformer.transform(testVectorizer)

if __name__ == '__main__':
    
    posAppear = []
    resSim = []
    docPos = []
    relv = 0
    pos = 1
    tupNo  = 0
    alpha = 0.005
    
    for res in result_set:
        if res > 0:
            #print(str(pos)+".txt      ", res)
            relv = relv + 1
            posAppear.append((pos,res))
        pos = pos + 1
    
    print("Total Documents: " ,relv+1)



posAppear.sort(key = lambda x:x[1], reverse = True)


print("Doc No  CosineSimilarity")
for tup in posAppear:
    print(str(tup[0])+".txt","  ",tup[1])
