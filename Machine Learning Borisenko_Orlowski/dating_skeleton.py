import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler 
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

#Create your df here:
df = pd.read_csv("/Users/admon/Downloads/capstone_starter/profiles.csv") 

#Visualize some of the Data
%matplotlib inline
plt.hist(df.age, bins=20)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.xlim(16, 80)
plt.show()

fig, ax = plt.subplots()
ax.hist(df.height.dropna(), alpha=0.9, color='teal', bins=80)
plt.xlabel("Height")
plt.ylabel("Frequency")
plt.xlim(55.0, 80.0)
plt.show()

plt.pie(df.body_type.value_counts(), autopct='%0.1f%%', explode = [.05,.05,.05,.07, .08, .09, .2, .5, .7, 1.2, 1.5, 2])
plt.legend(['average', 'fit', 'athletic', 'thin', 'curvy', 'a little extra', 'skinny', 'full figured', 'overweight', 'jacked', 'used up', 'rather not say'], bbox_to_anchor=(1,.52), loc="lower right")
plt.axis('equal')
plt.show()

#Transformation
drink_mapping = {"not at all": 0, "rarely": 1, "socially": 2, "often": 3, "very often": 4, "desperately": 5}

df["drinks_code"] = df.drinks.map(drink_mapping)
smokes_mapping = {"no":0, "sometimes":1, "when drinking":2, "yes":4, "trying to quit": 3}
df["smokes_code"] = df.smokes.map(smokes_mapping)
drugs_mapping = {"never":0, "sometimes":2, "often":4}
df["drugs_code"] = df.drugs.map(drugs_mapping)
sex_mapping = {"m": 0, "f": 1}
df["sex_code"] = df.sex.map(sex_mapping)
pet_mapping = {"likes dogs and likes cats":5, "likes dogs and has cats":7, "has dogs and likes cats":7,\
               "has dogs and has cats":10,\
               "has dogs":6, "has cats":6, "likes dogs":4, "likes cats":4, \
               "likes dogs and dislikes cats":2,"has dogs and dislikes cats":3,\
               "dislikes dogs and has cats":3, \
               "dislikes dogs and likes cats":2, "dislikes dogs and dislikes cats":0, "dislikes cats":1, \
               "dislikes dogs":1} 
df["pet_code"] = df.pets.map(pet_mapping)
body_mapping = {"average": 5, "fit":4, "athletic":3, "rather not say":5, "overweight":9, \
                "a little extra":6, "curvy":7, "thin":2,\
                "jacked":3, "full figured":6, "skinny":1, "used up":10}
df["body_code"] = df.body_type.map(body_mapping)
education_mapping = {"graduated from college/university":5,\
                     "graduated from masters program":7,\
                     "working on college/university":4,\
                     "working on masters program":6,\
                     "graduated from two-year college":4,\
                     "graduated from high school":2,\
                     "graduated from ph.d program":8,\
                     "graduated from law school":5,\
                     "working on two-year college":3,\
                     "dropped out of college/university":3,\
                     "working on ph.d program":9,\
                     "college/university":4,\
                     "graduated from space camp":5,\
                     "dropped out of space camp":3,\
                     "graduated from med school":5,\
                     "working on space camp":4,\
                     "working on law school":4,\
                     "two-year college":3,\
                     "working on med school":4,\
                     "dropped out of two-year college":2,\
                     "dropped out of masters program":5,\
                     "dropped out of ph.d program":7,\
                     "dropped out of high school":0,\
                     "high school":2,\
                     "working on high school":1,\
                     "space camp":4,\
                     "ph.d program":7,\
                     "law school":4,\
                     "dropped out of law school":3,\
                     "dropped out of med school":3,\
                     "med school":4}
df["education_code"] = df.education.map(education_mapping)

essay_cols = ["essay0","essay1","essay2","essay3","essay4","essay5","essay6","essay7","essay8","essay9"]
# Removing the NaNs
all_essays = df[essay_cols].replace(np.nan, '', regex=True)
#Combining the essays
all_essays = all_essays[essay_cols].apply(lambda x: ' '.join(x), axis=1)


df["essay_len"] = all_essays.apply(lambda x: len(x))
split_essays = all_essays.apply(lambda x: x.split())
df["average_word_length"] = split_essays.apply(lambda x: float(sum(len(i) for i in x)) / len(x) if len(x) != 0 else 0)
df["essays_word_count"] = split_essays.apply(lambda x: len(x))
df["selfish_rate"] = split_essays.apply(lambda x: x.count("I") + x.count("me") + x.count("i")+ x.count("i'm") + x.count("I'm"))

#Normalization
feature_data = df[['smokes_code','drinks_code','drugs_code','essay_len','average_word_length','selfish_rate','essays_word_count','income','sex_code','pet_code','body_code','education_code','height']].replace(np.nan, 0, regex=True)

x = feature_data.values
min_max_scaler = MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)

feature_data = pd.DataFrame(x_scaled, columns=feature_data.columns)

#Linear Regression
selfish = feature_data.selfish_rate.values
age = np.clip(df.age.values, 0, 71, out=age)
selfish = selfish.reshape(-1,1)
line_fitter = LinearRegression()
line_fitter.fit(selfish, df.age)
age_predict = line_fitter.predict(selfish)
plt.plot(selfish, age, 'o', color="pink", alpha=.2)
plt.plot(selfish, age_predict, color="purple")
plt.xlim(0,1)
plt.ylim(17,70)
plt.xlabel("Frequency of I and me")
plt.ylabel("Age")
plt.show()

education = feature_data.education_code.values
education = education.reshape(-1,1)
line_fitter = LinearRegression()
line_fitter.fit(education, df.age)
age_predict = line_fitter.predict(education)
plt.plot(education, age, 'o', color="pink", alpha=.2)
plt.plot(education, age_predict, color="purple")
plt.xlim(0,1)
plt.ylim(17,70)
plt.xlabel("Education Level")
plt.ylabel("Age")
plt.show()

#Multiple Linear Regression
features = feature_data[['smokes_code', 'drinks_code', 'drugs_code', 'essay_len', 'average_word_length', 'selfish_rate', 'essays_word_count', 'income', 'sex_code', 'pet_code', 'body_code', 'education_code']]
x_train, x_test, y_train, y_test = train_test_split(features, age, train_size = 0.8, test_size = 0.2, random_state=6)
mlr = LinearRegression()
model=mlr.fit(x_train, y_train)
y_predict = mlr.predict(x_test)

print("Age Prediction Accuracy")
print(mlr.score(x_train, y_train))
print(mlr.score(x_test, y_test))


features = feature_data[['smokes_code', 'drinks_code', 'drugs_code', 'essay_len', 'average_word_length',\
                         'selfish_rate', 'essays_word_count', 'income', 'pet_code', 'body_code', 'education_code']]
sex = feature_data.sex_code
x_train, x_test, y_train, y_test = train_test_split(features, sex, train_size = 0.8, test_size = 0.2, random_state=6)

mlr = LinearRegression()

model=mlr.fit(x_train, y_train)

y_predict = mlr.predict(x_test)

print("Sex Prediction Accuracy")
print(mlr.score(x_train, y_train))
print(mlr.score(x_test, y_test))

features = feature_data[['smokes_code', 'drinks_code', 'income', 'essay_len', 'average_word_length', 'selfish_rate', 'essays_word_count', 'pet_code', 'body_code', 'education_code', 'sex_code']]
drugs = feature_data.drugs_code
x_train, x_test, y_train, y_test = train_test_split(features, drugs, train_size = 0.8, test_size = 0.2, random_state=6)

mlr = LinearRegression()

model=mlr.fit(x_train, y_train)

y_predict = mlr.predict(x_test)

print("Drug Addiction Prediction Accuracy")
print(mlr.score(x_train, y_train))
print(mlr.score(x_test, y_test))

features = feature_data[['drugs_code', 'drinks_code', 'income', 'essay_len', 'average_word_length', 'selfish_rate', 'essays_word_count', 'pet_code', 'body_code', 'education_code', 'sex_code']]
smokes = feature_data.smokes_code
x_train, x_test, y_train, y_test = train_test_split(features, smokes, train_size = 0.8, test_size = 0.2, random_state=6)

mlr = LinearRegression()

model=mlr.fit(x_train, y_train)

y_predict = mlr.predict(x_test)

print("Smoking Prediction Accuracy")
print(mlr.score(x_train, y_train))
print(mlr.score(x_test, y_test))

#Classification
glass_ceiling = ["income", "education_code", "pet_code", "body_code", "essay_len", 'height']
x_train, x_test, y_train, y_test = train_test_split(feature_data[glass_ceiling], feature_data.sex_code,\
                                                    train_size = 0.8, test_size = 0.2, random_state=6)
classifier = KNeighborsClassifier(n_neighbors = 90)
classifier.fit(x_train, y_train)
guesses = classifier.predict(x_test)
labels = y_test
print("Accuracy score: " + str(accuracy_score(labels, guesses)))
print("Recall score: " + str(recall_score(labels, guesses)))
print("Precision score: " + str(precision_score(labels, guesses)))
print("F1 score: " + str(f1_score(labels, guesses)))