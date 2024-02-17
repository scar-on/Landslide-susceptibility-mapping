from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import argparse
import numpy as np

import data_prepare as dp
from utils import drawAUC_TwoClass

def parse_args():
    parser = argparse.ArgumentParser(description="Train ML models Processes on data")
    parser.add_argument( "--feature_path", default='origin_data/feature/', type=str)
    parser.add_argument( "--label_path", default='origin_data/label/label1.tif', type=str)
    parser.add_argument( "--model", default='ModelRF', type=str)
    args = parser.parse_args()
    return args
def main():
    args = parse_args()
    train_df, val_df = dp.get_ML_data(args.feature_path, args.label_path)
    x_train = train_df.iloc[:,0:9]
    y_train = np.array(train_df.iloc[:,9:10]).ravel()
    x_test = val_df.iloc[:,0:9]
    y_test = np.array(val_df.iloc[:,9:10]).ravel()

    ModelRF = RandomForestClassifier()
    ModelET = ExtraTreesClassifier()
    ModelKNN = KNeighborsClassifier(n_neighbors=2)
    ModelSVM = SVC(probability=True)
    ModelDC = DecisionTreeClassifier()

    models = {
        "ModelRF": ModelRF,
        "ModelET": ModelET,
        "ModelKNN": ModelKNN,
        "ModelSVM": ModelSVM,
        "ModelDC": ModelDC
    }

    if args.model in models:
        model = models[args.model]
        model.fit(x_train, y_train)   
        y_pred = model.predict(x_test)
        y_pred_prob = model.predict_proba(x_test)
        print('Model Name: ', model)
        tp, fn, fp, tn = confusion_matrix(y_test,y_pred,labels=[1,0]).reshape(-1)
        accuracy = round((tp+tn)/(tp+fp+tn+fn), 3)
        print('Accuracy :', round(accuracy*100, 2),'%')   
        
        drawAUC_TwoClass(y_test, model.predict_proba(x_test)[:,1], str(model)+'_AUC.png')

if __name__=='__main__':
    main()