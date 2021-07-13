
import keras
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from imutils import paths
from keras.models import Model
from keras import applications
from keras import optimizers
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten
from sklearn.model_selection import train_test_split
# import library untuk melakukan one-hot-encoding dengan memanfaatkan method LabelBinarizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
# import library optimizer, sebagai contoh SGD optimizer
from keras.optimizers import SGD

import matplotlib.pyplot as plt


# siapkan variabel kosong untuk menampung gambarnya
data = []

# siapkan variabel kosong untuk menampung label yang akan diprediksi
labels = []

# load dataset gambarnya dari path 'dataset/animals/'
dataset_path = 'Images/original/'
imagePaths = sorted(list(paths.list_images(dataset_path)))

# masukan data gambar hasil load tadi satu persatu ke dalam variabel data yang kosong tadi
for imagePath in imagePaths:
    
    # baca gambarnya
    img = cv2.imread(imagePath)
    
    # ubah ukuran / dimensi gambarnya menjadi 32x32 px, serta convert ke dalam format vektor
    img_flat = cv2.resize(img, (224,224))
    
    # simpan / tumpuk gambar yang sudah dibaca tadi
    data.append(img_flat)    
    
    # baca labelnya dengan melihat path foldernya
    label = imagePath.split(os.path.sep)[-2]
    
    # simpan / tumpuk label yang sudah dibaca tadi 
    labels.append(label)

# ubah data label tadi menjadi format array agar lebih memudahkan dalam komputasi
lbl = np.array(labels)

# ubah var data menjadi array numpy agar lebih memudahkan dalam komputasi
dt = np.array(data, dtype='float32')

# munculkan 
# plt.imshow(gbr)
print('dimensi        : ', dt.shape)
print('jumlah gambar  : ', dt.shape[0])
print('Ukuran gambar  : ', dt.shape[1])


img_width, img_height = 224, 224
input_shape = (img_width, img_height, 3)
# pecah data yang sudah diload tadi menjadi 2 bahagian iaitu
# 1) bahagian train
# X Train
# Y Train
# 2) bahagian Test
# X Test
# Y Test
#X = data input ; Y = label/output


# pecah dataset dengan jumlah data testing sebanyak 25% dari keseluruhan total data
x_train, x_test, y_train, y_test = train_test_split(dt, lbl, 
                                                    test_size=0.25, 
                                                    random_state=42)

# cek apakah sudah benar atau belum pemecaannya
print('x_train : ',x_train.shape)
print('x_test  : ', x_test.shape)
print('y_train : ', y_train.shape)
print('y_test  : ', y_test.shape)

# ubah tipe data gambarnya ke float
x_train = x_train.astype('float')
x_test = x_test.astype('float')

# normalisasi data gambarnya agar setiap nilai pixel memiliki rentang dari 0 sampai 1
x_train = x_train / 255
x_test = x_test / 255



# membuat object construktur
lb = LabelBinarizer()

# memulai melakukan one-hot-encoding untuk y_train dan y_test
y_train_encoded = lb.fit_transform(y_train)
y_test_encoded  = lb.transform(y_test)

# # cek nilainya
# y_test_encoded




# mulai membuat arsitektur NN dengan tipe fully connected layer
input_layer    = Input(shape=(img_width, img_height,3), dtype='float')
# buat hidden layer ke-1 dengan jumlah node sebanyak 50
hidden_1 = Dense(50, activation='relu', name='hidden_1')(input_layer)

#buat hidden layer ke-2 dengan jumlah node sebanyak 100
hidden_2 = Dense(100, activation='relu', name='hidden_2')(hidden_1)

# buat hidden layer ke-3 dengan jumlah node sebanyak 50
hidden_3 = Dense(50, activation='relu', name='hidden_3')(hidden_2)

# membuat output layer dengan jumlah node sebanyak data_output (3)
output_layer = Dense(30, activation='softmax')(hidden_3)
# membuat modelnya dengan memasukan input dan outputnya
animal_model = Model(inputs=input_layer, outputs=output_layer)
animal_model.summary()
opt = optimizers.SGD(lr=0.001, momentum=0.9)
animal_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])


# tentukan nilai learning-rate untuk optimizer yang digunakan
# opt = SGD(lr=1e-4, momentum=0.9)

# compile model dan tentukan fungsi loss, optimizer dan metriks pengujian model yang akan dilatih
# animal_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
# memulai pelatihan

# tentukan jumlah iterasi / perulangan pembelajaran yang digunakan
iterasi = 100

# tentukan jumlah data yang akan dimasukan ke dalam NN setiap satu iterasi
my_batch_size = 10

# memulai training dengan inputan x_train dan y_train
history = animal_model.fit(x=x_train, y=y_train, 
                         batch_size=my_batch_size, 
                         epochs=iterasi, 
                         validation_data=[x_test, y_test])


# melihat nilai loss dan akurasi. metode evaluate menghasilkan dua keluaran yaitu loss dan acc
nilai_loss, nilai_acc = animal_model.evaluate(x=x_test, y=y_test)

# lihat nilai loss akhir
print('loss    = ', nilai_loss)

#melihat nilai akurasi akhir
print('akurasi = ', nilai_acc)


# nilai-nilai loss-nya
loss = history.history['loss']

# nilai-nilai val_loss-nya
val_loss = history.history['val_loss']

# nilai-nilai akurasi-nya
acc = history.history['acc']

# nilai-nilai val_acc-nya
val_acc = history.history['val_acc']

# menentukan sumbu x nya dari 0 sampai dengan banyaknya jumlah nilai loss/acc
sumbu_x = np.arange(0, len(loss))

# memasukan library plotting (matplotlib)
import matplotlib.pyplot as plt

# grafik plot untuk melihat loss
plt.figure(figsize=(15,3))
plt.subplot(1,2,1)
plt.plot(sumbu_x, loss, label='loss')
plt.plot(sumbu_x, val_loss, label='val_loss')
plt.legend()

# grafik plot untuk melihat akurasinya
plt.subplot(1,2,2)
plt.plot(sumbu_x, acc, label='acc')
plt.plot(sumbu_x, val_acc, label='val_acc')
plt.legend()
plt.show()

# simpan model neural network 
animal_model.save('cnn_animal.h5')

# simpan label class-nya
import pickle
label = open('cnnlabel.pickle', 'wb')
label.write(pickle.dumps(lb))
label.close()

# gunakan file data ini sebagai inputan untuk melakukan prediksi di file 'iris_predict.ipynb' 
pd.DataFrame(x_test, columns=data.columns[:-1]).to_csv('data_testing.csv')