import os, sys
import cv2
import dlib
import pickle
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from imageio import imread
from collections import Counter
from itertools import product
from sr_resnet import sr_resnet

class Preparator(object):
    """
    データセットから目の近傍画像を抽出し、同時に正解ラベルを生成する。
    全てのユーザに関するデータセットをNumpyの配列形式で返す。

    Attributes
    ----------
    img_data ; numpy.array
        学習の際に入力となる画像データの配列。

    """

    def __init__(self, dir_list, resize_rate=4, load=None, save=None):
        self.__img_data = []
        self.__label = []
        self.__df = None
        self.__coords = None
        self.__aligned = False
        self.__resized = False
        self.__grayscaled = False
        self.__resize_rate = resize_rate
        self.__detector = dlib.get_frontal_face_detector() # 顔検出器の準備
        self.__predictor = dlib.shape_predictor("landmark.dat") # 顔器官検出器の準備

        if load: # 前回の画像データを読み込み
            with open(load, "rb") as f:
                self.__img_data, self.__label = pickle.load(f)
        else:
            with tqdm(total=len(dir_list)) as pbar_p:
                for dir in dir_list:
                    self.__img_data.append([])
                    self.__label.append([])
                    self.__analysing_df(dir)
                    with tqdm(total=len(self.__df.index)) as pbar_c:
                        for img_file in self.__df.index:
                            self.__clipping_eye(dir, img_file)
                            pbar_c.update()
                    pbar_p.update()
        height, width = self.__img_data[0][0][0].shape[:2]
        print("\nThe size of a image is about {}".format((height, width)))
        if save:
            with open(save, "wb") as f:
                pickle.dump([self.__img_data, self.__label], f)

    def srres_save(self):
        with open("dataset_sr_origin_8x4.pkl", "wb") as f:
            pickle.dump([self.__img_data, self.__label], f)

    def grayscale_save(self):
        with open("dataset_gray.pkl", "wb") as f:
            pickle.dump([self.__img_data, self.__label], f)

    def clahe_save(self):
        with open("dataset_clahe.pkl", "wb") as f:
            pickle.dump([self.__img_data, self.__label], f)

    def yoshida_save(self):
        with open("dataset_yoshida.pkl", "wb") as f:
            pickle.dump([self.__img_data, self.__label], f)

    def __clipping_eye(self, dir, img_file):
        img = imread(os.path.join(dir, img_file))
        height, width = img.shape[:2]
        resize_rate = self.__resize_rate
        temp_frame = cv2.resize(img,
                (int(width/resize_rate),
                int(height/resize_rate))
                )
        dets = self.__detector(temp_frame, 1)
        if len(dets) == 0:
            resize_rate = 1
            temp_frame = cv2.resize(img,
                    (int(width/resize_rate),
                    int(height/resize_rate))
                    )
            dets = self.__detector(temp_frame, 1)
        if len(dets)!=0:
            face_heights = [d.bottom()-d.top() for d in dets]
            face = dets[np.array(face_heights).argmax()]
            shape = np.array([[p.x, p.y] for p
                    in self.__predictor(temp_frame, face).parts()])
            eye_CR = ((shape[39]+shape[36])*resize_rate/2).astype(np.int32)
            eye_CL = ((shape[45]+shape[42])*resize_rate/2).astype(np.int32)
            eye_WR = ((shape[39]-shape[36])*resize_rate/4+3)[0].astype(np.int32)
            eye_WL = ((shape[45]-shape[42])*resize_rate/4+3)[0].astype(np.int32)
            img_R = img[int(eye_CR[1]-eye_WR):int(eye_CR[1]+eye_WR),
                    int(eye_CR[0]-eye_WR*2):int(eye_CR[0]+eye_WR*2)]
            img_L = img[int(eye_CL[1]-eye_WL):int(eye_CL[1]+eye_WL),
                    int(eye_CL[0]-eye_WL*2):int(eye_CL[0]+eye_WL*2)]
            coord = self.__df.loc[img_file].values
            idx = np.where(np.all(self.__coords == coord, axis=1))[0]
            img_L = cv2.resize(img_L,(36,18))
            img_R = cv2.resize(img_R,(36,18))
            self.__img_data[-1].append([img_L, img_R])
            self.__label[-1].append(idx)

    def __analysing_df(self, dir):
        # 画像のファイル名と注視座標の一覧が記されたCSVファイルを読み込み
        df = pd.read_csv(os.path.join(dir,"log.csv"),
                header= None,
                names = ["name", "x", "y"],
                index_col = 0
                )
        coords = self.__getFixedCoords(df)
        print(coords)
        self.__coords = coords
        self.__df = df[df["x"].isin(coords[:,0]) & df["y"].isin(coords[:,1])]

    def __getFixedCoords(self, df):
        # 読み込んだ座標一覧から固定点（例：3x3の点）の一覧を取得
        x = Counter(df["x"].values).most_common()
        x = [n[0] for n in x if n[1] > 200]
        y = Counter(df["y"].values).most_common()
        y = [n[0] for n in y if n[1] > 200]
        x.sort()
        y.sort()
        return np.array([[m, n] for n, m in product(y, x)])

    def resize(self, resize_rate):
        height, width = self.__img_data[0][0][0].shape[:2]
        for person in self.__img_data:
            for i,imgs in enumerate(person):
                person[i][0] = cv2.resize(imgs[0],
                                    (int(width*resize_rate),
                                    int(height*resize_rate))
                                    )
                person[i][1] = cv2.resize(imgs[1],
                                    (int(width*resize_rate),
                                    int(height*resize_rate))
                                    )
        print("The size of image is now {}".format(person[i][0].shape))
        self.__resized =True

    def scale_down(self, size):
        for person in self.__img_data:
            for i,imgs in enumerate(person):
                person[i][0] = cv2.resize(imgs[0], size, interpolation=cv2.INTER_AREA)
                person[i][1] = cv2.resize(imgs[1], size, interpolation=cv2.INTER_AREA)
        print("The size of image is nows {}".format(person[i][0].shape))

    def srresize(self):
        if self.__grayscaled:
            raise Exception("Grayscaled image can not be Super Resized")
        sr_resnet(self.__img_data)
        height, width = self.__img_data[0][0][0].shape[:2]
        print("\nthe size of a image is now {}".format((height, width)))
        # self.scale_down((20,10))
        self.__resized =True

    def apply_clahe(self):
        if not self.__resized:
            raise Exception("Enlarge the images before applying clahe")
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        for person in self.__img_data:
            for i,imgs in enumerate(person):
                if self.__grayscaled:
                    person[i][0] = clahe.apply(imgs[0])
                    person[i][1] = clahe.apply(imgs[1])
                elif not self.__grayscaled:
                    person[i][0] = clahe.apply(
                            cv2.cvtColor(imgs[0], cv2.COLOR_RGB2GRAY)
                            )
                    person[i][1] = clahe.apply(
                            cv2.cvtColor(imgs[1], cv2.COLOR_RGB2GRAY)
                            )
        self.__grayscaled = True

    def grayscale(self):
        if self.__grayscaled == True:
            raise Exception("Already grayscaled")
        for person in self.__img_data:
            for i,imgs in enumerate(person):
                person[i][0] = cv2.cvtColor(imgs[0], cv2.COLOR_RGB2GRAY)
                person[i][1] = cv2.cvtColor(imgs[1], cv2.COLOR_RGB2GRAY)
        self.__grayscaled = True

    def preview(self, target = 0):
        for i,imgs in enumerate(self.__img_data[target]):
            if not self.__grayscaled:
                img_L = cv2.cvtColor(imgs[0].astype(np.uint8), cv2.COLOR_RGB2BGR)
                img_R = cv2.cvtColor(imgs[1].astype(np.uint8), cv2.COLOR_RGB2BGR)
            else:
                img_L = cv2.cvtColor(imgs[0].astype(np.uint8), cv2.COLOR_GRAY2BGR)
                img_R = cv2.cvtColor(imgs[1].astype(np.uint8), cv2.COLOR_GRAY2BGR)
            cv2.imshow("Preview Window", np.hstack((img_L, img_R)))
            if cv2.waitKey(400)==27:
                break
        cv2.destroyAllWindows()

    @property #gettter
    def img_data(self):
        if not self.__aligned:
            self.resize(resize_rate=1)
            self.__aligned=True
        return np.array(self.__img_data)

    @property
    def label(self):
        return np.array(self.__label)

if __name__ == '__main__':
    dataset_path = ".\\only_dataset"
    dir_list = [os.path.join(dataset_path,n) for n in os.listdir(dataset_path)
                        if os.path.isdir(os.path.join(dataset_path,n))]

    data = Preparator(dir_list=dir_list, resize_rate=3, save="dataset.pkl")
