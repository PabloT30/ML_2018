import shutil
import pandas as pd
import numpy as np
import csv

reader = csv.reader(open("train_image_paths.csv","r"),delimiter=',')
x = list(reader)
result = np.array(x).astype("str")

for i in range(36807):
    if "ELBOW" in result[i+1][0]:
        if "positive" in result[i+1][0]:
            print("hi")
            new_name = "data/elbow_test_dataset/positive/"+str(i+1)+".png"
            shutil.copy(result[i+1][0], new_name)
        else:
            continue
    else:
        continue

for a in range(36807): #36807 for training and 3196 for testing
    if "ELBOW" in result[a+1][0]:
        if "negative" in result[a+1][0]:
            new_name = "data/elbow_test_dataset/negative/"+str(a+1)+".png"
            shutil.copy(result[a+1][0], new_name)
        else:
            continue
    else:
        continue

#for i in range(36808):
#    new_name = "./data/"+str(i+1)+".png"
#    shutil.copy(result[i+1][0], new_name)

#if(result[0][0]=="id"):
#    print(result[1][0])
#if "negative" in result[0][0]:
#    print("Siiii")