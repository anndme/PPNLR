import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from skimage.transform import resize
from sklearn.preprocessing import StandardScaler
import time
start = time.time()
data_dir = 'C:/Users/ZYH/Desktop/new1112/CTCOVID'

def load_data(data_dir):
    images = []
    labels = []
    # 用于计数的变量
    covid_count = 0
    non_covid_count = 0

    for label in ['CT_COVID', 'CT_NonCOVID']:
        class_dir = os.path.join(data_dir, label)
        for img_name in os.listdir(class_dir):
            if img_name.endswith('.png'):
                img_path = os.path.join(class_dir, img_name)
                images.append(img_path)
                if label == 'CT_COVID':
                    labels.append(0)
                    covid_count += 1
                else:
                    labels.append(1)
                    non_covid_count += 1

    print(f"Number of CT_COVID images: {covid_count}")
    print(f"Number of CT_NonCOVID images: {non_covid_count}")

    return images, labels

images, labels = load_data(data_dir)

def load_image(image_path, target_size=(32, 32)):
    image = Image.open(image_path)
    image = image.convert('L')  # 转换为灰度图像
    image = np.array(image)
    image = resize(image, target_size, anti_aliasing=True)
    return image.flatten()


X = [load_image(img_path) for img_path in images]
X = np.array(X)
y = np.array(labels)
print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = LogisticRegression(max_iter=1000, solver='lbfgs')
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
print('Classification Report:')
print(classification_report(y_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
end = time.time()
print("time：", end - start, "s")