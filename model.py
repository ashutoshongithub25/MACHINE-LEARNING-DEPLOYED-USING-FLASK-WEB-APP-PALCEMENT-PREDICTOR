import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv('Placement_Data_Full_Class.csv')


from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder() 
dataset['gender']= label_encoder.fit_transform(dataset['gender']) 
dataset['gender'].unique() 

dataset['ssc_b']= label_encoder.fit_transform(dataset['ssc_b']) 
dataset['ssc_b'].unique() 

dataset['hsc_b']= label_encoder.fit_transform(dataset['hsc_b']) 
dataset['hsc_b'].unique() 

dataset['hsc_s']= label_encoder.fit_transform(dataset['hsc_s']) 
dataset['hsc_s'].unique() 

dataset['degree_t']= label_encoder.fit_transform(dataset['degree_t']) 
dataset['degree_t'].unique() 

dataset['workex']= label_encoder.fit_transform(dataset['workex']) 
dataset['workex'].unique() 

dataset['specialisation']= label_encoder.fit_transform(dataset['specialisation']) 
dataset['specialisation'].unique() 

dataset['status']= label_encoder.fit_transform(dataset['status']) 
dataset['status'].unique()


x = dataset.iloc[:, 1:13].values


y=dataset.iloc[:,13].values
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 1)


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=1000,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=0, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

print(classifier.predict(([[1,68,1,91,1,1,58,2,0,55,1,58]])))

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

pickle.dump(classifier, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))