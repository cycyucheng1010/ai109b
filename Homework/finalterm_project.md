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
