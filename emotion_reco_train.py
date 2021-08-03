import sys
import os
import cv2 , time
import pandas as pd
import numpy as np

face_recognizer = cv2.face.FisherFaceRecognizer_create()

df=pd.read_csv("icml_face_data.csv")
# print(df.describe())
i=0
X_train,Y_train,X_test,Y_test=[],[],[],[]
for index,row in df.iterrows():
    # print(row['pixels'])
    val = row['pixels'].split(" ")
    try:
        if 'Training' in row['Usage']:
            X_train.append(np.array(val,'float32'))
            Y_train.append(row['emotion'])
        elif 'PublicTest' in row['Usage']:
            X_test.append(np.array(val,'float32'))
            Y_test.append(row['emotion'])
    except:
        print(f"error occured at row :{row} index:{index}")
# print(f"data of X_train {X_train[0:2]}")
# print(f"data of Y_train {Y_train[0:2]}")
# print(f"data of X_test {X_test[0:2]}")
# print(f"data of Y_test {Y_test[0:2]}")

face_recognizer.train(np.asarray(X_train), np.asarray(Y_train))
face_recognizer.save("trained.yml")
print("done")

