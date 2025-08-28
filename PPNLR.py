import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from DDHIP import *
import random
import time
import psutil
import os
from PIL import Image
from skimage.transform import resize



class PPNLR(object):


        def __init__(self, rate, lanta,x, Y):
            self.rate = rate
            self.size = len(Y)
            self.lanta = lanta
            setup = DDHIP_Setup(l = len(x))
            self.mpk, self.msk = setup.setup()



        def generating_matrix(self, x, y, n):
             half_n = int(n / 2)
             A = x[0:half_n]
             B = x[half_n:]
             A_1 = np.hstack((A.reshape(1, A.shape[0]), np.ones((1, n - half_n))))
             A_2 = np.hstack((A.reshape(1, A.shape[0]), np.ones((1, n - half_n - 1))))
             A_2 = np.insert(A_2, 0, np.ones(1), axis=1)
             B = np.hstack((np.ones((1, half_n)), B.reshape(1, B.shape[0])))
             B_1 = np.hstack((B.reshape(1, B.shape[1]), y.reshape(1, 1)))
             # B_1 = np.array(list(B) + list(y))
             B_2 = np.insert(B, 0, [1], axis=1)
             Ma = np.dot(A_2.reshape(A_2.shape[1], A_2.shape[0]), A_1)
             Mb = np.dot(B_2.reshape(B_2.shape[1], B_2.shape[0]), B_1)
             return Ma, Mb



        def encrypt_matrix(self, x, y, n):
            M = np.zeros((n, n))
            A1 = np.zeros((len(x), n, n))
            B1 = np.zeros((len(x), n, n))
            for k in range(len(x)):
                A2, B2 = self.generating_matrix(x[k], y[k], n)
                for i in range(A2.shape[0]):
                    for j in range(A2.shape[1]):
                         A1[k,i,j] = A2[i][j]
                         B1[k,i,j] = B2[i][j]

            for i in range(A1.shape[1]):
                for j in range(A1.shape[2]):
                    gamma = np.random.randn()
                    gamma_inv =  1/gamma
                    # print('gamma=',gamma*gamma_inv)

                    encrypt = DDHIP_Encrypt(A1[:, i, j]*gamma, self.mpk, self.msk)
                    # print(len(mpk),len(msk))
                    ct = encrypt.encrypt()
                    decrypt = DDHIP_Decrypt(B1[:, i, j]*gamma_inv, self.msk, ct)
                    M[i][j] = decrypt.decrypt()

            print(M.shape)
            return M




        def omega_zero(self, omega_init_r, z, u0):
            omega_new = (1 - 2 * self.lanta * self.rate) * omega_init_r - self.rate / self.size * (0.25 * z - 0.5 * u0)
            return omega_new


        def generate_V(self, Ar):
            V0 = Ar[0][:-1]
            V = [np.insert(V0, 0, self.size)]
            Uj = [float(Ar[0][-1])]
            for i in range(1, Ar.shape[0]):
                vj = Ar[i][:-1]
                V += [np.insert(vj, 0, Ar[0][i - 1])]
                Uj += [float(Ar[i][-1])]
            return V, Uj



        def item_model_while(self, itemmax, omega, A):
            V, Uj = self.generate_V(A)
            omega_old = omega
            for u in range(itemmax):
                omega_new = []
                for i in range(len(Uj)):
                    z = np.dot(omega_old,  V[i].T)
                    omega_new.append(self.omega_zero(omega_old[0][i], z, Uj[i]))
                omega_old = np.array(omega_new).T
            return omega_old





def normalization(data):
    max_ = data.max(axis=0)
    min_ = data.min(axis=0)
    diff = max_ - min_
    zeros = np.zeros(data.shape)
    m = data.shape[0]
    zeros = data - np.tile(min_, (m, 1))
    zeros = zeros / np.tile(diff, (m, 1))
    return zeros



def confusion_matrix(D, X, Y):
    def sigmoid(inX):
        return 1.0/(1+np.exp(-inX))
    omega0 = D[0]
    omega = D[1:]
    predict = sigmoid(np.dot(omega, X.T) + omega0) > 0.5
    Y_predict = Y > 0
    result = [[np.sum(predict), len(Y) - np.sum(predict)], [np.sum(Y_predict), len(Y)-np.sum(Y_predict)]]
    return result



def accuracy(matrix):
    return (sum(matrix[1])-abs(matrix[0][0] - matrix[1][0])) / sum(matrix[1])



def load_csv_diabetes(path):
    X = []
    Y = []
    with open(path, 'r') as f:
        datas = f.readlines()

    for data in datas[1:]:
        split_data = data.split(',')
        X.append([float(i) for i in split_data[:-1]])
        if "positive" in split_data[-1]:  # 阳性为1，阴性为-1
            Y.append(1)
        elif "negative" in split_data[-1]:
            Y.append(-1)
        else:
            Y.append(split_data[-1])
    return np.array(X), np.array(Y)



def load_csv_australian(path):
    X = []
    Y = []
    with open(path, 'r') as f:
        datas = f.readlines()

    for data in datas:
        split_data = data.split(' ')
        d = []
        for i in split_data:
            try:
                d.append(int(float(i)))
            except:
                d.append(np.NAN)
        X.append(d)
        if "0" in split_data[-1]:
            Y.append(-1)
        elif "1" in split_data[-1]:
            Y.append(1)
        else:
            Y.append(split_data[1])
    return np.array(X), np.array(Y)





def load_csv_wisconsin(path):
    X = []
    Y = []
    with open(path, 'r') as f:
        datas = f.readlines()

    for data in datas:
        split_data = data.split(',')
        d = []
        for i in split_data[1:-1]:
            try:
                d.append(float(i))
            except:
                d.append(np.NAN)
        X.append(d)
        if "2" in split_data[-1]:
            Y.append(1)
        elif "4" in split_data[-1]:
            Y.append(-1)
        else:
            Y.append(split_data[-1])
    return np.array(X), np.array(Y)

def load_data1(data_dir):
    images = []
    labels = []
    for label in ['CT_COVID', 'CT_NonCOVID']:
        class_dir = os.path.join(data_dir, label)
        for img_name in os.listdir(class_dir):
            if img_name.endswith('.png'):
                img_path = os.path.join(class_dir, img_name)
                images.append(img_path)
                labels.append(-1 if label == 'CT_COVID' else 1)
    return images, labels


def load_data2(data_dir):
    images = []
    labels = []
    for label in ['normal', 'tuberculosis']:
        class_dir = os.path.join(data_dir, label)
        for img_name in os.listdir(class_dir):
            if img_name.endswith('.png'):
                img_path = os.path.join(class_dir, img_name)
                images.append(img_path)
                labels.append(-1 if label == 'normal' else 1)
    return images, labels

if __name__ == '__main__':

    # path = r"C:\Users\Yuhao Zhang\Desktop\数据集\diabetes_csv.csv"     #DD
    # X, Y = load_csv_diabetes(path)


    # dataset = pd.read_csv(r"C:\Users\ZYH\Desktop\processed.cleveland1.csv") #HDD
    # X = dataset.iloc[:219, :-1].values
    # Y = dataset.iloc[:219, -1].values

    #
    # path = r'C:\Users\Yuhao Zhang\Desktop\数据集\australian.csv' #ACAD
    # X, Y = load_csv_australian(path)

    #
    # path = r'C:\Users\Yuhao Zhang\Desktop\数据集\breast-cancer-wisconsin.csv' #WIBC
    # X, Y = load_csv_wisconsin(path)

    # data_dir = 'C:/Users/ZYH/Desktop/CTCOVID'
    data_dir = 'C:/Users/ZYH/Desktop/montgomery_chest_xray'


    images, labels = load_data2(data_dir)

    def load_image(image_path, target_size=(32, 32)):
        image = Image.open(image_path)
        image = image.convert('L')  # 转换为灰度图像
        image = np.array(image)
        image = resize(image, target_size, anti_aliasing=True)
        return image.flatten()



    X = [load_image(img_path) for img_path in images]
    X = np.array(X)
    Y = np.array(labels)

    X = normalization(X)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    A = PPNLR(0.01, 0.001, X_train, Y_train)
    M1 = A.encrypt_matrix(X_train, Y_train, X.shape[1]+1)
    # print(A1,A0)
    omega0 = np.random.randint(low=np.min(1), high=np.max(10), size=(1, X.shape[1]+1))
    omega1 = random.randint(0, 10) - omega0
    D = A.item_model_while(1000, omega0, M1)
    matrix = confusion_matrix(D[0], X_train, Y_train)
    matrix1 = confusion_matrix(D[0], X_test, Y_test)
    print(accuracy(matrix), matrix)
    print(accuracy(matrix1), matrix1)
