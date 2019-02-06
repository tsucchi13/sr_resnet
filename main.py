import os, sys
import csv
import numpy as np
import pandas as pd
import sklearn
import warnings
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from statistics import mean
from preparator import Preparator
from preprocessor import Preprocessor
from sklearn.svm import SVC
from sklearn.decomposition import PCA
# from sklearn.naive_bayes import GaussianNB, ComplementNB, BernoulliNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
import pickle

warnings.filterwarnings('ignore')
def main():
    parser = argparse.ArgumentParser(description='SVM testing tool')
    parser.add_argument('--dataset', '-d', type=str, default=".\\only_dataset",
                    help='Pass of the directory which contains datasets')
    parser.add_argument('--superresolution', '-sr', type=bool, default=False,
                    help='Use SuperResolution to scale up imgae (default:False)')
    parser.add_argument('--resize', '-r', type=int, default=0,
                    help='Resizing image with lanczos interpolation (default:False)')
    parser.add_argument('--grayscale', '-g', type=bool, default=False,
                    help='Grayscale images (default:False)')
    parser.add_argument('--clahe', '-c', type=bool, default=False,
                    help='Apply clahe to images (default:False)')
    parser.add_argument('--yoshida', '-y', type=bool, default=False,
                    help='Scale down images like yoshida\'s implementation (default:False)')
    parser.add_argument('--scaledown', '-sd', type=bool, default=False,
                    help='Scale down images 10x10  (default:False)')
    parser.add_argument('--save', '-s', type=str, default=None,
                    help='Save loaded images to pkl (default:None)')
    parser.add_argument('--load', '-l', type=str, default=None,
                    help='Load images from pkl (default:None)')
    parser.add_argument('--eyes', '-e', type=str, default="both",
                    help='Choose image of eyes to learn (default:both)')
    parser.add_argument('--processStyle', '-ps', type=str, default=None,
                    help='save csv as result_ps.csv')
    args = parser.parse_args()

    if args.load==None:
        dir_list = [os.path.join(args.dataset,n) for n in os.listdir(args.dataset)
                            if os.path.isdir(os.path.join(args.dataset,n))]
    else:
        dir_list = None

    data = Preparator(dir_list=dir_list,
            resize_rate=3,
            load=args.load,
            save=args.save)

    print("-----------------------------")
    if args.superresolution:
        data.srresize()
        print("SuperResolution is applied")
        data.srres_save()
    if args.resize:
        if args.superresolution:
            warnings.warn("Using both SuperResolution and Resize is not recommended")
        data.resize(resize_rate=args.resize)
        print("Resized the image")
    if args.grayscale:
        data.grayscale()
        print("Grayscaled the image")
        # data.grayscale_save()
    if args.clahe:
        if not args.grayscale:
            raise Exception("Images should be grayscaled before Clahe")
        data.apply_clahe()
        print("Applied Clahe")
        # data.clahe_save()
    if args.yoshida:
        if args.superresolution or args.resize or not args.grayscale:
            raise Exception("Yoshida's implementation must be grayscaled and not scaled up")
        data.scale_down((10,10))
        print("yoshida data is Selected")
        # data.yoshida_save()
    if args.scaledown:
        data.scale_down((20,10))
        print("yoshida data is Selected")

    pre_data = Preprocessor(data=data.img_data, label=data.label, eyes = args.eyes)
    print("-----------------------------\n")

    all_avg_pr = []
    all_avg_re = []
    varience_ratio_sum = 0
    savedata = ['params','mean_score','sd']
    dimension = 3
    loop = 0
    # for dimension in range(3,200,20):
    for i in range(10):
        pre_data.split_data(i, stds=True, mms=False)
        print('--')
        print(pre_data.test_data.shape)
        pca = PCA(n_components=dimension)
        pca.fit(X=pre_data.test_data)
        test_data = pca.transform(pre_data.test_data)
        # varience_ratio_sum += sum(pca.explained_variance_ratio_)
        # print('累積寄与率: {0}'.format(sum(pca.explained_variance_ratio_)))#各処理(gray,sr)でのaverage算出のためコメントアウト
        # train_data = pre_data.train_data.reshape(-1,600)#pcaしない場合は，これのコメントアウト外す．ここで10(px)×10(px)×3(ch)×2(両目)次元をSVMに入れられるように1次元フラットに直してる．
        # test_data = pre_data.test_data.reshape(pre_data.test_label.shape[0],-1)
        # clf = SVC()
        # clf = GaussianNB()
        tuned_parameters = [
        # {'C': [1,], 'kernel': ['linear']},
        {'C': [10], 'kernel': ['rbf'], 'gamma': [0.001]}, #best
        # {'C': [10], 'kernel': ['poly'], 'degree': [2], 'gamma': [0.001]},
        # {'C': [100], 'kernel': ['sigmoid'], 'gamma': [0.001]}
        # {'C':[1, 10,100], "verbose":[0], "solver":["lbfgs"], "multi_class":["auto"], "max_iter":[10000, 100000]}
        ]
        # tuned_parameters = [{"var_smoothing":[1e-09]}]
        # tuned_parameters = [{"alpha":[1.0]}]
        score = "f1_macro"
        clf = GridSearchCV(
                SVC(),
                tuned_parameters,
                cv=10,#交差検証でデータを何分割にするか
                scoring=score,
                n_jobs=-1)

        clf.fit(test_data, pre_data.test_label.ravel())

        # print(clf.grid_scores_)#average算出のためコメントアウト
        # print(clf.best_params_)

        # print("Grid scores on development set:")#average算出のためコメントアウト
        # print()#average算出のためコメントアウト

        for params, mean_score, scores in clf.grid_scores_:
            # print("%0.3f (+/-%0.03f) for %r"
            #     % (mean_score, scores.std() * 2, params))#average算出のためコメントアウト
            savedata = np.vstack([savedata,[params,mean_score, scores.std() / 2]])

        savefilename = 'result_' + args.processStyle + '.csv'
        with open(savefilename, 'w') as f:
            writer = csv.writer(f, lineterminator='\n') # 改行コード（\n）を指定しておく
            writer.writerows(savedata) # 2次元配列も書き込める
        # print()#average算出のためコメントアウト

        # print("The scores are computed on the full evaluation set.")#average算出のためコメントアウト
        #10人で交差検証しないときは以下をコメントアウト
        # test_data = pca.transform(pre_data.test_data)
        # # test_data = pre_data.test_data.reshape(-1,600)
        # print(test_data.shape)
        # y_true, y_pred = pre_data.test_label.ravel(), clf.predict(test_data)
        # # print(classification_report(y_true, y_pred))
        # result = classification_report(y_true, y_pred, output_dict=True)
        # # precision = sum([result[i]["precision"] for i in result])/9
        # # recall = sum([result[i]["recall"] for i in result])/9
        # print("Precision is {}, and Recall is {}\n".format(result["macro avg"]["precision"], result["macro avg"]["recall"]))
        # all_avg_pr.append(result["macro avg"]["precision"])
        # all_avg_re.append(result["macro avg"]["recall"])
    # print("All average of Precision is {}, and Recall is {}".format(mean(all_avg_pr), mean(all_avg_re)))
    # print(dimension)

        if i == 0:
            with open('svmmodel_0202_v2.pickle', mode='wb') as fp:
                pickle.dump(clf, fp)
            with open("test_data_0202_v2.pkl", "wb") as f:
                pickle.dump([test_data], f)

    print(test_data.shape)

    df = pd.read_csv('result_' + args.processStyle + '.csv',header=0,encoding='utf-8')
    data_list = df.mean_score.values.tolist()
    average = sum(data_list[loop:])/10
    print('acuracy_average:{}'.format(average))
    print('varience_ratio_average:{}'.format(varience_ratio_sum/10))
    loop = loop +10
    varience_ratio_sum = 0

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # ax.scatter(test_data[:,0], test_data[:,1], test_data[:,2])
    for i in range(test_data.shape[0]):
        if 0 <= pre_data.test_label[i] <= 3 or 8 <= pre_data.test_label[i] <= 11:
            ax.scatter(test_data[i,0], test_data[i,1], test_data[i,2], c='red')
        if 4 <= pre_data.test_label[i] <= 7 or 12 <= pre_data.test_label[i] <= 15:
            ax.scatter(test_data[i,0], test_data[i,1], test_data[i,2], c='blue')
        if 16 <= pre_data.test_label[i] <= 19 or 24 <= pre_data.test_label[i] <= 27:
            ax.scatter(test_data[i,0], test_data[i,1], test_data[i,2], c='yellow')
        if 20 <= pre_data.test_label[i] <= 23 or 28 <= pre_data.test_label[i] <= 31:
            ax.scatter(test_data[i,0], test_data[i,1], test_data[i,2], c='green')

    plt.savefig('result_{}.jpeg'.format(args.processStyle))
    plt.show()

if __name__ == '__main__':
    main()
