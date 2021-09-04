import pandas as pd
import  numpy as np
import matplotlib.pyplot as plt
import  seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

ds = pd.read_csv('train.csv')
#print(ds.head(10))
#print(ds.shape)
#print(ds.columns)
#print(ds.isnull().sum())
#print(ds.info())
#print(ds['price_range'].describe() , ds['price_range'].unique())
#ds = ds.drop(columns= 'blue')
#ds = ds.drop(columns= 'm_dep')  # we have removed m_dep and blue columns because its a redundant feature which dosn't affect of price
#print(ds.columns)
#print( sns.boxplot(x= 'price_range' , y= 'talk_time' , data= ds) )

print('/////////////////////////////////////////////////////////////////////////////////////')
#Using  KNN algorithm
X = ds.drop('price_range' , axis= 1)
Y = ds['price_range']
X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size = 0.25 , random_state = 0)
KNN = KNeighborsClassifier(n_neighbors=  5, metric='minkowski', p=2)
KNN.fit(X_train , Y_train)
print(KNN.score(X_test , Y_test))        # note that accuracy of KNN is 93.33 %

print('/////////////////////////////////////////////////////////////////////////////////////////')
#Using Decision Tree Algorithm
decisionTree = DecisionTreeClassifier(random_state = 40)
decisionTree.fit(X_train,Y_train)
print(decisionTree.score(X_test , Y_test))  # note that accuracy of DTC is 80.33 % So KNN is better that DTC and note that DTC can used in  classifier and regression


#So we will use KNN in  prediction of mobile price
test_dataset = pd.read_csv('test.csv')  #Getting  the test data
test_dataset = test_dataset.drop('id',axis=1) #to match data set to trainig data or to be identical
print(test_dataset)
predicted_price = KNN.predict(test_dataset)
print(predicted_price)






