import os, sys
import sklearn
import warnings
import argparse
from preparator import Preparator
from preprocessor import Preprocessor
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

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
    parser.add_argument('--save', '-s', type=str, default=None,
                    help='Save loaded images to pkl (default:None)')
    parser.add_argument('--load', '-l', type=str, default=None,
                    help='Load images from pkl (default:None)')
    parser.add_argument('--eyes', '-e', type=str, default="both",
                    help='Choose image of eyes to learn (default:both)')
    args = parser.parse_args()

    dir_list = [os.path.join(args.dataset,n) for n in os.listdir(args.dataset)
                        if os.path.isdir(os.path.join(args.dataset,n))]
    data = Preparator(dir_list=dir_list,
            resize_rate=3,
            load=args.load,
            save=args.save)

    if args.superresolution:
        data.srresize()
        print("SuperResolution is applied")
    if args.resize:
        if args.superresolution:
            warnings.warn("Using both SuperResolution and Resize is not recommended")
        data.resize(resize_rate=args.resize)
        print("Resized the image")
    if args.grayscale:
        data.grayscale()
        print("Grayscaled the image")
    if args.clahe:
        if not args.grayscale:
            raise Exception("Images should be grayscaled before Clahe")
        data.apply_clahe()
        print("Applied Clahe")
    if args.yoshida:
        if args.supperresolution or args.resize or not args.grayscale:
            raise Exception("Yoshida's implementation must be grayscaled and not scaled up")
        data.scale_down()
        print("yoshida data")

    pre_data = Preprocessor(data=data.img_data, label=data.label, eyes = args.eyes)
    pre_data.split_data(1)

    pca = PCA(n_components=3)
    pca.fit(X=pre_data.train_data)
    train_data = pca.transform(pre_data.train_data)

    clf = SVC()
    tuned_parameters = [
    {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
    {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.001, 0.0001]},
    {'C': [1, 10, 100, 1000], 'kernel': ['poly'], 'degree': [2, 3, 4], 'gamma': [0.001, 0.0001]},
    {'C': [1, 10, 100, 1000], 'kernel': ['sigmoid'], 'gamma': [0.001, 0.0001]}
    ]
    score = "f1"
    clf = GridSearchCV(
            SVC(),
            tuned_parameters,
            cv=5,
            scoring='%s_weighted' % score )

    clf.fit(train_data, pre_data.train_label)
    print(clf.grid_scores_)
    print(clf.best_params_)

    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() * 2, params))
    print()

if __name__ == '__main__':
    main()
