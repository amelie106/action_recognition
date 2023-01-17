import pandas as pd
import pickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.linear_model import RidgeClassifier

def max_value(inputlist):
    return max([int(sublist[-1]) for sublist in inputlist])
    
def min_value(inputlist):
    return min([int(sublist[-1]) for sublist in inputlist])

# load CLIP features
with open('extracted/features_clip.pkl', 'rb') as f:
    data=pickle.load(f)
with open('extracted/labels_clip.pkl', 'rb') as f:
    label=pickle.load(f)

# load baseline features
# with open('extracted/features_baseline.pkl', 'rb') as f:
#     data=pickle.load(f)
# with open('extracted/labels_baseline.pkl', 'rb') as f:
#     label=pickle.load(f)

svm_train_top_k_accuracy_scores=[]
svm_test_top_k_accuracy_scores=[]

ridge_train_top_k_accuracy_scores=[]
ridge_test_top_k_accuracy_scores=[]

for subject in range(min_value(label), max_value(label)+1):
    print(subject)

    X_train = []
    X_test = []
    y_train = []
    y_test = []

    for idx in range(len(data)):
        if label[idx][2] == str(subject):
            X_test.append(data[idx])
            y_test.append(label[idx][0])
        else:
            X_train.append(data[idx])
            y_train.append(label[idx][0])

    if X_train and X_test and y_train and y_test:
        
        ## Ridge
        clf = RidgeClassifier()
        clf.fit(X_train, y_train)

        Y_train_pred_ridge = clf.predict(X_train)
        Y_test_pred_ridge = clf.predict(X_test)

        ridge_train_top_k_accuracy_scores.append(accuracy_score(y_train,Y_train_pred_ridge))
        ridge_test_top_k_accuracy_scores.append(accuracy_score(y_test,Y_test_pred_ridge))

        ## SVC
        clf = SVC(C=100)
        clf.fit(X_train, y_train)

        Y_train_pred_svc = clf.predict(X_train)
        Y_test_pred_svc = clf.predict(X_test)

        svm_train_top_k_accuracy_scores.append(accuracy_score(y_train,Y_train_pred_svc))
        svm_test_top_k_accuracy_scores.append(accuracy_score(y_test,Y_test_pred_svc))

accuracy = pd.DataFrame({'svm_test': svm_test_top_k_accuracy_scores, 'svm_train': svm_train_top_k_accuracy_scores,
                        'ridge_test': ridge_test_top_k_accuracy_scores, 'ridge_train': ridge_train_top_k_accuracy_scores})

accuracy.to_csv('data/csv/accuracy_clip.csv') # save clip results for single image
# accuracy.to_csv('data/csv/accuracy_clip_all.csv') # save clip results for 10 images
# accuracy.to_csv('data/csv/accuracy_baseline.csv') # save baseline results for single image
# accuracy.to_csv('data/csv/accuracy_baseline_all.csv') # save baseline results for 10 images