import pandas as pd
import numpy as np
import pickle

dataset = pd.read_csv("Forest_fire.csv")
dataset = np.array(dataset)

X = dataset[1:, 1:-1]
y = dataset[1:, -1]
y = y.astype('int')
X = X.astype('int')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

model.fit(X_train, y_train)

inpt=[int(x) for x in "45 32 60".split(' ')]
final=[np.array(inpt)]

pickle.dump(model,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))