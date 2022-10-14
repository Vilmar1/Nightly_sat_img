from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import os
from matplotlib.image import imread
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

# dec=input('Usar a GPU? Digite "n" para "Não" ')
# if dec=='n':
#     os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

##### MODELO
def get_model(x):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=2, activation='relu',input_shape=x[0].shape))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters=32, kernel_size=2, activation='relu',input_shape=x[0].shape))
    model.add(MaxPooling2D(pool_size=2))
    
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    return(model)

###### IMPORTAÇÃO DE DADOS IMAGENS SATÉLITE
paths=os.listdir(r'D:\backup\Desktop\SP\Imagens Diárias POA')
images=[]
for path in paths:
    image_new = imread(rf'D:\backup\Desktop\SP\Imagens Diárias POA\{path}')[:,:,1]
    image_new=image_new.reshape(1,image_new.shape[0],image_new.shape[1])
    images.append(image_new)
images=np.concatenate(images,axis=0)

###### NORMALIZACAO
x=np.array(images)
x=x/255
x=x.reshape(x.shape[0],x.shape[1],x.shape[2],1)

###### IMPORTAÇÃO DE DADOS DE ENERGIA
df=pd.read_excel(r'D:\backup\Desktop\SP\EE municipal\eeSP_POA.xlsx',index_col=0)
y_raw=df.loc[(df['Cidade']=='PORTO ALEGRE') & 
             (df['Data']>='2021-05-01') &
             (df['Data']!='2021-10-31')&
             (df['Data']<='2021-10-31')].iloc[:,-1]

###### NORMALIZACAO
y=(y_raw-min(y_raw))/(max(y_raw)-min(y_raw))


###### TRAIN TEST SPLIT
X_train,X_test,y_train,y_test = train_test_split(x,y,shuffle=True,
                                                 test_size=.25)

###### FIT MODEL
model=get_model(x)                                                 
callback=EarlyStopping(monitor='val_loss',patience=5)
h_callback=model.fit(X_train, y_train, epochs=60, batch_size=50,validation_data=(X_test,y_test),verbose=1,callbacks=[callback])

##### GET ACCURACY
y_pred=model.predict(X_test).reshape((X_test.shape[0]))
y_adj=y_pred*(max(y_raw)-min(y_raw))+min(y_raw)
y_test_adj=y_test*(max(y_raw)-min(y_raw))+min(y_raw)
mape=np.mean(abs(y_adj-y_test_adj)/y_test_adj)
print(mape)

###### PLOT LOSS
def plot_loss(loss,val_loss, title=''):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title(f'Model loss {title}')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.ylim((0,0.2))
    # plt.ylim(0,1)
    plt.legend(['Train', 'Test'], loc='upper right')
    # plt.savefig(f'{title}')
    plt.show()

plot_loss(h_callback.history['loss'], h_callback.history['val_loss'])
