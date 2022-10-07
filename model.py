import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv("Crop_recommendation.csv")

df.drop(['temperature'], axis=1)
df.drop(['humidity'], axis=1)
df.drop(['rainfall'], axis=1)

X = np.array(df[['N', 'P', 'K', 'ph']])
Y = np.array(df['label'])


sc = StandardScaler()
X = sc.fit_transform(X)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=20)


model = RandomForestClassifier()
model.fit(X_train, Y_train)


y_pred = model.predict(X_test)
print(y_pred)


pickle.dump(model, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))
print('success')
