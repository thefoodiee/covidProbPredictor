import pandas as pd #for reading database
import pickle #to store python values
import numpy as np #to convert data into arrays
from sklearn.linear_model import LogisticRegression #for logistic regression model

#function to split the data, 80% for training and 20% testing
def data_split(data, ratio):
    shuffled = np.random.permutation(len(data)) #outputs random array in specified range
    test_set_size = int(len(data)*ratio) #setting test data size
    test_indices = shuffled[:test_set_size] #making array of test data
    train_indices = shuffled[test_set_size:] #making array of train data
    return data.iloc[train_indices], data.iloc[test_indices] #returns the split training and testing data

if __name__ == "__main__":
    dt = pd.read_csv('data.csv') #reads the data
    train, test = data_split(dt, 0.2) #splits the data using the function
    
    x_train = train[['fever','bodyPain','age','runnyNose','diffBreath']].to_numpy()#puts different train data types into one array
    x_test = test[['fever','bodyPain','age','runnyNose','diffBreath']].to_numpy()#puts different test data types into one array
    
    
    y_train = train[['infectionProb']].to_numpy().reshape(1200, )#puts train output label into array
    y_test = test[['infectionProb']].to_numpy().reshape(299, )#puts test output label into array
    clf = LogisticRegression()#initialises logistic regression model
    clf.fit(x_train,y_train)#gives the value to the model

    file = open('model.pkl','wb')#dumps the python value to a file

    pickle.dump(clf,file)

    file.close#closes the file
    
