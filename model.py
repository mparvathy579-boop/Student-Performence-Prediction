import pandas as pd
import numpy as np
import pickle    #using for preserving
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# load dataset
data=pd.read_csv('student-data (1).csv')

x =data.drop(['passed'], axis=1)
y =data['passed']


cate_cols = ['school', 'sex', 'address', 'famsize', 'Pstatus','Mjob','Fjob','reason','guardian','schoolsup', 'famsup', 'paid', 'activities', 'nursery',
       'higher', 'internet', 'romantic']
num_cols=['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel',
       'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences']


x_train,x_test,y_train,y_test=train_test_split(
    x,y,test_size=0.2,random_state=42
)

x_train = pd.get_dummies(
    x_train, columns=cate_cols, dtype=int
)

x_test = pd.get_dummies(
    x_test, columns=cate_cols, dtype=int
)


model=LogisticRegression()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)
print('Accuracy=',accuracy_score(y_test,y_pred))

#save the model
pickle.dump(model,open('model.pkl','wb'))
# feature colunm
pickle.dump(x.columns.tolist(), open('feature_columns.pkl', 'wb'))

print('Model training completed')