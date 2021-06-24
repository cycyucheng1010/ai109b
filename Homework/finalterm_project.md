# 期末專案 : 微笑檢測
## 背景介紹
* 此專案為去年暑期參加於交通大學舉辦之「深度學習與工業大數據課程」時所寫。
* 參考當時授課老師上課時之課程後與父親共同寫出。
## 前言
* 目的: 透過深度學習的方式判別人臉是否為笑臉。
* 本專案利用CNN框架訓練出模型。
* 同時也探討2種不同Optimizer在此專案上的精準度差異。  

## 程式說明
### 資料集簡介
* 有相當多的數據已在github上開源，可供教學使用。
* 使用資料集之來源: https://github.com/hromi/SMILEsmileD.git
* 包含13165張灰度照片 
* 每張圖片皆為64x64
* 有9475張unsmiling照片
* 有3690張smiling照片

![image](https://user-images.githubusercontent.com/62127656/120694521-e6440a00-c4dc-11eb-9027-3a0dda5258cc.png)
>(圖一) smiling

![image](https://user-images.githubusercontent.com/62127656/120694571-f825ad00-c4dc-11eb-9893-598e3217cafe.png)
>(圖二) unsmiling
### 使用環境
* Colab ----> google的開發環境，可免費使用內建其GPU長達8hr/day
* tensorflow ----> 人工智慧套件，因在impport keras時已有tensorflow的套件，這邊僅做版本選擇。
* matplotlib ----> 圖形繪製
* numpy ----> 數學計算
* cv2 -----> 基本影像處理，用於輸入本人測試照片時使用。
* skimage ----> 用來處理影像
* keras -----> 深度學習函式庫

### [程式碼連結](https://github.com/cycyucheng1010/ai109b/blob/main/Homework/%E6%9C%9F%E6%9C%AB%E4%BD%9C%E6%A5%AD.ipynb)
>以下圖片只針對部分重點做說明

![image](https://user-images.githubusercontent.com/62127656/120694975-623e5200-c4dd-11eb-9469-5ad113a1cc19.png)
>(圖三) 抓取資料集

![image](https://user-images.githubusercontent.com/62127656/120695141-9fa2df80-c4dd-11eb-9436-12c2a9a2df3e.png)
>(圖四) 進行資料預處理，讓數據等量

![image](https://user-images.githubusercontent.com/62127656/120696246-e80ecd00-c4de-11eb-8283-73a1f7cb48f5.png)
>(圖五) 將圖片統一格式轉換成矩陣，每500回報一次

![image](https://user-images.githubusercontent.com/62127656/120696499-36bc6700-c4df-11eb-919a-37f33129afcb.png)
>(圖六) 取80%進行學習20測試，結束後進行捲積及池化

![image](https://user-images.githubusercontent.com/62127656/120697008-c6621580-c4df-11eb-9b51-c85bf51d0fa9.png)
>(圖七) Epoch選擇30次，讓模型去訓練，這邊的optmizer先後測試了rmsprop以及Adam

![image](https://user-images.githubusercontent.com/62127656/120697496-5e5fff00-c4e0-11eb-948d-e491140d7900.png)
>(圖八) 模型儲存以及評估

![image](https://user-images.githubusercontent.com/62127656/120697598-84859f00-c4e0-11eb-9c1a-884e387dea54.png)
![image](https://user-images.githubusercontent.com/62127656/120697651-936c5180-c4e0-11eb-91e2-2c2b6dec188b.png)
![image](https://user-images.githubusercontent.com/62127656/120697684-9bc48c80-c4e0-11eb-9807-8eec19380edb.png)
>(圖九、圖十、圖十一) 使用外部照片進行測試

## 結語
* 利用了4張照片進行測試，而電腦與我們主觀的判斷一致說明測試是成功的。(可參考圖十一之結果)
* 測試後發現RMSPROP的accuracy比ADAM來的好(accuracy 99% V.S.97%)
 
![image](https://user-images.githubusercontent.com/62127656/120698902-438e8a00-c4e2-11eb-8514-73f3bf048c97.png)
>(圖十二) RMSPROP與ADAM之間的比較
* 未來可以與硬體結合並優化程式，成為一個有趣的實驗，Ex: 在樹梅派上裝上相機進行人臉的微笑判斷。

## 完整程式碼
```
from google.colab import drive
drive.mount('/content/drive')

#4621 從github網站下載數據集
!git clone https://github.com/hromi/SMILEsmileD.git

#4622 安裝資料夾 樹狀結構工具
!apt-get install tree


#4623 顯示資料夾 樹狀結構
!tree SMILEsmileD -L 3

#4624 沒有微笑(負向)圖片數量 
%%time
from imutils import paths #路徑檔案管理
neg_images = sorted(list(paths.list_images('SMILEsmileD/SMILEs/negatives/negatives7'))) 
print(len(neg_images))

#4625 微笑(正向)圖片數量 
pos_images = sorted(list(paths.list_images('SMILEsmileD/SMILEs/positives/positives7'))) 
print(len(pos_images))

#4626 兩個分類檔案 取相同數量 
neg_images = neg_images[:3690]
print(len(neg_images))
print(len(pos_images))

#4627 製作數據集 [圖片路徑,label] 微笑:1 沒微笑:0
dataset = [(path, 0) for path in neg_images] + [(path, 1) for path in pos_images]
print(dataset[-2])
len(dataset)

#4628
from IPython.display import Image
Image(dataset[8][0],width=200)

Image(dataset[26][0],width=200)

#4629
%tensorflow_version 1.x
%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from skimage.io import imread #skimage python影像處理模組
from skimage.measure import block_reduce

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras import models
from keras import layers
from sklearn.model_selection import train_test_split

#4630 將所有圖片轉成矩陣
%%time
x_train = []
y_train = []
count=0
for path,label in dataset:
    image = cv2.imread(path) #載入圖片
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #轉灰階
    image = cv2.resize(image,(180,192)) #統一圖片尺寸
    # 3x3局部採樣取平均值 類似池化層 減少數據量 
    image = block_reduce(image, block_size=(3, 3), func=np.mean) 
    x_train.append(image)
    y_train.append(label)
    count=count+1
    if count % 500 == 0:
        print(count,'處理完成..',label)
#imgplot = plt.imshow(image)


#4631 list轉成np矩陣不產生副本
x_train = np.asarray(x_train) 
y_train = np.asarray(y_train)


x_train.shape


np.save('x_train', x_train)
np.save('y_train', y_train)


x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')

x_train.shape


#4632 標準化成0~1浮點數
x_train = x_train.astype(np.float32) / 255. #標準化
y_train = y_train.astype(np.int32)
print (x_train.dtype, x_train.min(), x_train.max(), x_train.shape)
print (y_train.dtype, y_train.min(), y_train.max(), y_train.shape)

#4633 資料預處理

from keras.utils import np_utils
number_of_categories = 2 #分類數
# 標籤轉 one hot
y_train = np_utils.to_categorical(y_train, number_of_categories).astype(np.float32)

#隨機打亂順序
indices = np.arange(len(x_train))
temp_x = x_train
temp_y = y_train
np.random.shuffle(indices)
x_train = x_train[indices]
y_train = y_train[indices]

#4634 顯示訓練數據與標籤
w=10
h=10
fig=plt.figure(figsize=(8, 8))
columns = 4
rows = 5
for i in range(1, columns*rows +1):
    img = x_train[i]
    fig.add_subplot(rows, columns, i)
    plt.title(y_train[i])
    plt.imshow(img)
plt.show()

#4635 分割測試集
x_train = np.expand_dims(x_train, axis=-1)
# 分割20%成為測試集
(trainX, testX, trainY, testY) = train_test_split(x_train, y_train, test_size=0.20, random_state=1)
print(trainX.shape)

#4636 建模

from keras.layers import LeakyReLU
filters = 32
conv_size = 3 #卷積 3*3
pool_size = 2 #池化 2*2

model = Sequential()
model.add(layers.Conv2D(filters,(conv_size,conv_size),LeakyReLU(alpha=0.1),input_shape=trainX.shape[1:]))
#model.add(layers.Conv2D(filters,(conv_size,conv_size),activation="relu",input_shape=trainX.shape[1:]))
model.add(layers.MaxPooling2D((pool_size,pool_size)))
model.add(layers.Conv2D(filters*2,(conv_size,conv_size),activation="relu"))
model.add(layers.MaxPooling2D((pool_size,pool_size)))
model.add(layers.Conv2D(filters,(conv_size,conv_size),activation="relu"))
model.add(layers.MaxPooling2D((pool_size,pool_size)))
model.add(layers.Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(number_of_categories, activation='softmax'))

model.summary()

#4637 模型編譯與訓練
%%time
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(trainX, trainY, epochs=30)


#4638 儲存模型
model.save('smile_epochs30.h5')
print("The model has been saved!")

#4639 模型評估
score = model.evaluate(testX, testY)
print('Test score:', score[0])
print('Test accuracy:', score[1])

history.history

#4640 繪製訓練歷程圖表
history_dict = history.history
print(history_dict.keys())
plt.style.use("ggplot")
plt.figure()
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["accuracy"], label="acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()


#顯示圖片 
from IPython.display import Image
#圖片路徑
PATH1 = "/content/drive/My Drive/my_photo/B24EAC70-B3D4-4A51-ADF1-BA730BC8E7B7.jpg"                       
Image(filename = PATH1 , width=300, height=300)

#顯示圖片 
from IPython.display import Image
#圖片路徑
PATH2 = "/content/drive/My Drive/my_photo/273E6014-046A-4CCB-B2C8-6D2EDA7F235A.jpg"
Image(filename = PATH2 , width=300, height=300)

#顯示圖片 
from IPython.display import Image

#圖片路徑
PATH3 = "/content/drive/My Drive/my_photo/o20200513170148.jpg"                   
Image(filename = PATH3 , width=300, height=300)


#顯示圖片 
from IPython.display import Image

#圖片路徑
PATH4 = "/content/drive/My Drive/my_photo/E00187A0-7283-4CA5-BECF-2DBEEEF3A84B.jpg"                        
Image(filename = PATH4 , width=300, height=300)

#4646 取得Image file路徑list (i.e., ..\\image\\~.jpg)
from imutils import paths #路徑檔案管理
test_img = sorted(list(paths.list_images('/content/drive/My Drive/my_photo'))) 
print(len(test_img))
test_img # Show all images in the "image" folder.

# Smile prediction, developed by Horace for NCTU class @ 2020/08/25
def JDG_smile(pid):
    test_images = []
    t_image = cv2.imread(test_img[pid]) #2000 1000 
    t_image = cv2.cvtColor(t_image, cv2.COLOR_BGR2GRAY)
    t_image = cv2.resize(t_image,(60,64))
    test_images.append(t_image)

    plt.imshow(t_image)
    test_images = np.asarray(test_images)
    test_images = test_images.astype(np.float32) / 255.

    test_images = np.expand_dims(test_images, axis=-1)
    p = model.predict(np.array([test_images[0]]))[0]

    print(p)
    class_names = ["W/o Smiling","Unsmiling"] # why "Neutral"
    bar_width = 50 #刻度寬度
    left_count = int(p[1] * bar_width) #使用Smiling決定 左邊刻度
    right_count = bar_width - left_count 
    left_side = '-' * left_count #顯示左邊長度
    right_side = '-' * right_count #顯示右邊長度
    print (class_names[0], left_side + '<|>' + right_side, class_names[1])
    
    JDG_smile(0)
    
    JDG_smile(1)
    
    JDG_smile(2)
    
    JDG_smile(3)
    
    
```
