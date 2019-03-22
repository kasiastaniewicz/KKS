import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from keras.utils import to_categorical
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from scipy import stats
# fix random seed for reproducibility
np.random.seed(7)

df = pd.read_csv('/mnt/storage/home/ey18822/DATA.csv', error_bad_lines = False)

scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(df.iloc[:,0:115])

dataset = df.values

Y = dataset[:,115]


#
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
dummy_y = np_utils.to_categorical(encoded_Y)

X_train, X_test, y_train, y_test =train_test_split(X_scaled,dummy_y, test_size = 0.25)

# create model


#echos is fixed
batch_size = [250,500,1000,1500,2000,2500,3000]
loss = list()
acc = list()
for i in batch_size:
    print("batch : " + str(i))
    model = Sequential()
    model.add(Dense(250, input_dim=115, activation='relu'))
    model.add(Dense(250, activation='relu'))
    model.add(Dense(25, activation='sigmoid'))
    model.add(Dense(6, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, batch_size=i)
    scaledtorscores = model.evaluate(X_train, y_train)
    loss.append(scaledtorscores[0])
    acc.append(scaledtorscores[1]*100)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scaledtorscores[1]*100))
print("the loss list is")
print(loss)
print("the accuracy list is")
print(acc)


# evaluate the model
model1 = Sequential()
model1.add(Dense(250, input_dim=115, activation='relu'))
model1.add(Dense(250, activation='relu'))
model1.add(Dense(25, activation='sigmoid'))
model1.add(Dense(6, activation='softmax'))
model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs=list()
trainacc=list()
testacc=list()
deltaepochs=5
eon=0
for i in range(0,10) :
        print("epoch : " + str(eon))
        model1.fit(X_train, y_train, epochs=deltaepochs, batch_size=1500)
        scores = model1.evaluate(X_train, y_train)
        testscores = model1.evaluate(X_test, y_test)
        eon+=deltaepochs
        epochs.append(eon)
        trainacc.append(scores[1])
        testacc.append(testscores[1])
results=[epochs,
        trainacc,
        testacc]
print(results)

predictions = model1.predict(X_test)
prediction = predictions
for i in range(len(prediction)):
    a = 0
    indexpred = 0
    for j in range(6):
        if prediction[i][j] > a:
            a = prediction[i][j]
            indexpred = j
            
    for k in range(6):
        if k==indexpred:
            prediction[i][k] = 1
        else:
            prediction[i][k] = 0

#prediction
index_pred = np.where(np.in1d(prediction, [1]))[0]
index_pred = index_pred%6
#true
index =  np.where(np.in1d(y_test, [1]))[0]
index = index%6

#confusion matrix
matrix = confusion_matrix(index,index_pred)
print(matrix)

#sensitivity
def sensitivity(confusion_matrix,i):
    true_positives = confusion_matrix[i][i]
    false_negatives = sum(confusion_matrix[i]) - true_positives
    return true_positives/(true_positives+false_negatives)
print("The sensitivity is ")   
for i in range(6):
    print(sensitivity(matrix,i))

#specficity
def specificity(confusion_matrix,i):
    a=np.array(confusion_matrix)
    cols = (np.sum(a, axis=0))
    rows = (np.sum(a, axis=1))
    true_positives = confusion_matrix[i][i]
    true_negatives = sum(sum(confusion_matrix)) - rows[i] - cols[i] + confusion_matrix[i][i]
    false_positives = cols[i] - true_positives
    return true_negatives/(true_negatives+false_positives)
print("The specificity is ")    
for i in range(6):
    print(specificity(matrix,i))

##binary classify udp and tcp
tcp_udp = df[df['Type'].str.contains("udp|tcp")]
#preprocessing
tcp_udp['Type'] = tcp_udp['Type'].map({'tcp':0, 'udp':1})
tcp_udp_dataset = tcp_udp.values
scaler = preprocessing.StandardScaler()
X_tcpudp = scaler.fit_transform(tcp_udp.iloc[:,0:115])
Y_tcpudp = tcp_udp_dataset[:,115]

#split the data
Xtcpudp_train, Xtcpudp_test, ytcpudp_train, ytcpudp_test =train_test_split(X_tcpudp,Y_tcpudp, test_size = 0.25)

# create model
model = Sequential()
model.add(Dense(200, input_dim=115, activation='softmax'))
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(Xtcpudp_train, ytcpudp_train, epochs=50, batch_size=1500)
#predict on test set
predictions = model.predict(Xtcpudp_test)
rounded = [round(x[0]) for x in predictions]

print(pd.crosstab(rounded-ytcpudp_test,columns="Residual"))

print(pd.crosstab(ytcpudp_test,columns="Type"))


#Non-paramatric test: kruskal wallis test

df_udp = df[df['Type'].str.contains("udp")]
df_tcp = df[df['Type'].str.contains("tcp")]

scaler = preprocessing.StandardScaler()
dataset_tcp = scaler.fit_transform(df_tcp.iloc[:,0:115])
dataset_udp = scaler.fit_transform(df_udp.iloc[:,0:115])
#just test on very small set of data (i.e 0.1%)
X1_tcp, X11_tcp =train_test_split(dataset_tcp, test_size = 0.001)
X1_udp, X11_udp =train_test_split(dataset_udp, test_size = 0.001)

pvalue = []

for i in range(len(X11_tcp)):
    for j in range(len(X11_udp)):
        a = (stats.kruskal(dataset_tcp[i], dataset_udp[j]))
        pvalue.append(a.pvalue)
# Multiple test, Bonferroni correction
q = 0
alpha = 0.05
n = len(X11_tcp) * len(X11_udp)

for i in range(len(pvalue)):
    if pvalue[i] < (alpha/n):
        q = q + 1
#print percentage of pvalue that being rejected(not the same distribution)       
print(q/len(pvalue))

# combine udp and tcp as tcp&udp
df1 = df
df1['Type'] = df.Type.replace(["tcp","udp"],"tcp&udp")
dataset1 = df1.values
##preprocessing
scaler = preprocessing.StandardScaler()
X_scaled_1 = scaler.fit_transform(df1.iloc[:,0:115])
Y_1 = dataset1[:,115]
Z_1 = dataset1[:,116]
encoder = LabelEncoder()
encoder.fit(Y_1)
encoded_Y = encoder.transform(Y_1)
dummy_y1 = np_utils.to_categorical(encoded_Y)

#split the data
X1_train, X1_test, y1_train, y1_test,z1_train,z1_test =train_test_split(X_scaled_1,dummy_y1, Z_1, test_size = 0.25)

# create model
model2 = Sequential()
model2.add(Dense(250, input_dim=115, activation='relu'))
model2.add(Dense(250, activation='relu'))
model2.add(Dense(25, activation='sigmoid'))
model2.add(Dense(5, activation='softmax'))

model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs=list()
trainacc=list()
testacc=list()
deltaepochs=5
eon=0
for i in range(0,10) :
        print("epoch : " + str(eon))
        model2.fit(X1_train, y1_train, epochs=deltaepochs, batch_size=1500)
        scores = model2.evaluate(X1_train, y1_train)
        testscores = model2.evaluate(X1_test, y1_test)
        eon+=deltaepochs
        epochs.append(eon)
        trainacc.append(scores[1])
        testacc.append(testscores[1])
results=[epochs,
        trainacc,
        testacc]
print(results)

#predict on test data
predictions = model2.predict(X1_test)
prediction = predictions
for i in range(len(prediction)):
    a = 0
    indexpred = 0
    for j in range(5):
        if prediction[i][j] > a:
            a = prediction[i][j]
            indexpred = j
            
    for k in range(5):
        if k==indexpred:
            prediction[i][k] = 1
        else:
            prediction[i][k] = 0
index_pred = np.where(np.in1d(prediction, [1]))[0]
index_pred = index_pred%5
index =  np.where(np.in1d(y1_test, [1]))[0]
index = index%5

#confusion matrix
matrix1 = confusion_matrix(index,index_pred)
print(matrix1)

print("The sensitivity is ")   
for i in range(5):
    print(sensitivity(matrix1,i))
print("The specificity is ") 
for i in range(5):
    print(specificity(matrix1,i))

#Hide one type of attack from data(e.g we hide "scan" here) to check if the our
#model is able to classify unknown attack.

##extract "scan" from our dataset
scan_index = np.where(y1_train[:,3] == 1)
X2_train = np.delete(X1_train,scan_index,axis=0)
y2_train = np.delete(np.delete(y1_train,scan_index,axis=0),np.s_[3],1)
X2_test = X1_test
y2_test = y1_test
# create model
model3 = Sequential()
model3.add(Dense(250, input_dim=115, activation='relu'))
model3.add(Dense(250, activation='relu'))
model3.add(Dense(25, activation='sigmoid'))
model3.add(Dense(4, activation='softmax'))

model3.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model3.fit(X2_train, y2_train, epochs=50, batch_size=1500)

#test on previous test data
predictions = model3.predict(X2_test)
#softmax-posterior probability value of the output layer for all classes
#and threshold it(say 0.5). If the probability is < 0.5 for all other classes,
#then definitely this test case is an outlier(our unknown class in this case).
prediction1 = predictions
threshold = 0.5
for i in range(len(prediction1)):
    for j in range(4):
        if prediction1[i][j] >= threshold:
            prediction1[i][j] = 1
        else:
            prediction1[i][j] = 0
prediction2 = np.insert(prediction1,3,0,axis=1)
for i in range(len(prediction2)):
    if sum(prediction2[i]) == 0:
        prediction2[i][3] = 1
index_pred = np.where(np.in1d(prediction2, [1]))[0]
index_pred = index_pred%5
index= np.where(np.in1d(y2_test, [1]))[0]
index = index%5
#confusionmatrix
matrix3 = confusion_matrix(index,index_pred)
print(matrix3)

print("The sensitivity is ")   
for i in range(5):
    print(sensitivity(matrix3,i))
print("The specificity is ") 
for i in range(5):
    print(specificity(matrix3,i))

#Try hide another type of attack("combo")
combo_index = np.where(y1_train[:,1] == 1)
X2_train = np.delete(X1_train,combo_index,axis=0)
y2_train = np.delete(np.delete(y1_train,combo_index,axis=0),np.s_[1],1)
X2_test = X1_test
y2_test = y1_test
# create model
model3 = Sequential()
model3.add(Dense(250, input_dim=115, activation='relu'))
model3.add(Dense(250, activation='relu'))
model3.add(Dense(25, activation='sigmoid'))
model3.add(Dense(4, activation='softmax'))

model3.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model3.fit(X2_train, y2_train, epochs=50, batch_size=1500)

#test on previous test data
predictions = model3.predict(X2_test)
#softmax-posterior probability value of the output layer for all classes
#and threshold it(say 0.5). If the probability is < 0.5 for all other classes,
#then definitely this test case is an outlier(our unknown class in this case).
prediction1 = predictions
threshold = 0.5
for i in range(len(prediction1)):
    for j in range(4):
        if prediction1[i][j] >= threshold:
            prediction1[i][j] = 1
        else:
            prediction1[i][j] = 0
prediction2 = np.insert(prediction1,1,0,axis=1)
for i in range(len(prediction2)):
    if sum(prediction2[i]) == 0:
        prediction2[i][3] = 1
index_pred = np.where(np.in1d(prediction2, [1]))[0]
index_pred = index_pred%5
index= np.where(np.in1d(y2_test, [1]))[0]
index = index%5
#confusionmatrix
matrix3 = confusion_matrix(index,index_pred)
print(matrix3)

print("The sensitivity is ")   
for i in range(5):
    print(sensitivity(matrix3,i))
print("The specificity is ") 
for i in range(5):
    print(specificity(matrix3,i))

#Try to classify devices
encoder = LabelEncoder()
encoder.fit(Z_1)
encoded_Z = encoder.transform(Z_1)
dummy_z1 = np_utils.to_categorical(encoded_Z)
X3_train, X3_test,z3_train,z3_test =train_test_split(X_scaled_1,dummy_z1, test_size = 0.25)
# create model
model4 = Sequential()
model4.add(Dense(250, input_dim=115, activation='relu'))
model4.add(Dense(250, activation='relu'))
model4.add(Dense(25, activation='sigmoid'))
model4.add(Dense(4, activation='softmax'))
model4.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model4.fit(X3_train, z3_train, epochs=50, batch_size=1500)
predictions = model4.predict(X3_test)
prediction = predictions
for i in range(len(prediction)):
    a = 0
    indexpred = 0
    for j in range(4):
        if prediction[i][j] > a:
            a = prediction[i][j]
            indexpred = j
            
    for k in range(4):
        if k==indexpred:
            prediction[i][k] = 1
        else:
            prediction[i][k] = 0
index_pred = np.where(np.in1d(prediction, [1]))[0]
index_pred = index_pred%4
index =  np.where(np.in1d(z3_test, [1]))[0]
index = index%4
matrix4 = confusion_matrix(index,index_pred)
print(matrix4)
print("The sensitivity is ")   
for i in range(4):
    print(sensitivity(matrix4,i))
print("The specificity is ") 
for i in range(4):
    print(specificity(matrix4,i))

#hide one type of device("doorbell")
doorbell_index = np.where(z3_train[:,0] == 1)
X4_train = np.delete(X3_train,doorbell_index,axis=0)
z4_train = np.delete(np.delete(z3_train,doorbell_index,axis=0),np.s_[0],1)
X4_test = X3_test
z4_test = z3_test
# create model
model5 = Sequential()
model5.add(Dense(250, input_dim=115, activation='relu'))
model5.add(Dense(250, activation='relu'))
model5.add(Dense(25, activation='sigmoid'))
model5.add(Dense(3, activation='softmax'))

model5.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model5.fit(X4_train, z4_train, epochs=50, batch_size=1500)

#test on previous test data
predictions = model5.predict(X4_test)
#softmax-posterior probability value of the output layer for all classes
#and threshold it(say 0.5). If the probability is < 0.5 for all other classes,
#then definitely this test case is an outlier(our unknown class in this case).
prediction1 = predictions
threshold = 0.5
for i in range(len(prediction1)):
    for j in range(3):
        if prediction1[i][j] >= threshold:
            prediction1[i][j] = 1
        else:
            prediction1[i][j] = 0
prediction2 = np.insert(prediction1,0,0,axis=1)
for i in range(len(prediction2)):
    if sum(prediction2[i]) == 0:
        prediction2[i][0] = 1
index_pred = np.where(np.in1d(prediction2, [1]))[0]
index_pred = index_pred%4
index= np.where(np.in1d(z4_test, [1]))[0]
index = index%4
#confusionmatrix
matrix5 = confusion_matrix(index,index_pred)
print(matrix5)

print("The sensitivity is ")   
for i in range(4):
    print(sensitivity(matrix5,i))
print("The specificity is ") 
for i in range(4):
    print(specificity(matrix5,i))
