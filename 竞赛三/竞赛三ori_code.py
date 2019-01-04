
# coding: utf-8

# In[1]:


from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, Conv2D, Activation, MaxPool2D
from keras import losses, optimizers
from keras import applications
from keras.callbacks import EarlyStopping
import numpy as  np
from keras.optimizers import Adam
import matplotlib.pyplot as  plt
from PIL import Image
import tensorflow as tf
import pandas as pd
import cv2
import os
import sys
import shutil
from keras.utils import np_utils
import random
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


# # 准备数据

# In[ ]:


for i in range(10000):
    im =  Image.open('F://dataset/train/'+str(i)+'.jpg').convert('L')
    im.save('F://dataset/pretreat/'+str(i)+'.jpg')
for i in range(20000):
    im =  Image.open('F://dataset/train/'+str(i)+'.jpg').convert('L')
    im.save('F://dataset/pretreat_test/'+str(i)+'.jpg')


# # 手动数据分类

# In[11]:


for i in range(1000):
    os.mkdir('F://dataset/trainset/'+str(i)) #训练集
#     os.mkdir('F://dataset/testset/'+str(i))
    os.mkdir('F://dataset/labset/'+str(i)) #验证集


# In[14]:


filename = 'F://dataset/train_labels.csv'    #读取训练集结果，使之与图片一一对应
text = pd.read_csv(filename)

text = np.array(text)
# text = text[:,1]
# text = text.tolist()


# In[14]:


#根据text的值，将图片放入对应的文件夹。文件夹名和图片的数字一致
for i in range(10000):
    shutil.copy('F://dataset/pretreat/'+str(i)+'.jpg','F://dataset/trainset/'+str(text[i])+'/'+str(i)+'.jpg')


# In[2]:


#随机选取2000加入验证集
for i in range(2000):
    a = random.randint(0,9999)
    shutil.copy('F://dataset/pretreat/'+str(a)+'.jpg','F://dataset/labset/'+str(text[a])+'/'+str(a)+'.jpg')


# # 搭建模型

# In[6]:


from keras.layers.pooling import GlobalAveragePooling2D
from keras.applications.mobilenet import preprocess_input


# In[3]:


#使用模型mobilenet，权重选择已预训练过的imagenet
base_model = applications.mobilenet.MobileNet(weights='imagenet', include_top=False, input_shape=(128,128,3))


# In[4]:


x = base_model.output
x = Dropout(0.5)(x) #全连接层。随机的让一些节点不工作，来减少过拟合的情况。
x = GlobalAveragePooling2D()(x) #对整个网路在结构上做正则化防止过拟合,但会造成收敛速度减慢
pre = Dense(1000, activation='softmax')(x) #全连接输出
 
model = Model(inputs=base_model.input, outputs=pre)


# In[6]:


model.compile(optimizer=Adam(),  #使训练数据迭代地更新神经网络权重
              loss="categorical_crossentropy",   #多类的对数损失
              metrics=['accuracy'])


# In[33]:


train_datagen = ImageDataGenerator(   #类别次序根据文件名称的字母顺序来排列
        shear_range=0.2,  #浮点数，剪切强度（逆时针方向的剪切变换角度）
        zoom_range=0.2, #浮点数或形如[lower,upper]的列表，随机缩放的幅度
        preprocessing_function = preprocess_input) #使图片变为矩阵，每一个数据/255

# test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(  #训练集生成器
        'F://dataset/trainset',
        target_size=(128, 128),
        batch_size=32, #每批数据量
        classes=[str(i) for i in range(1000)],  #数字的字符串顺序和数字本身的排序不同
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        'F://dataset/labset',
        target_size=(128, 128),
        batch_size=32,
        classes=[str(i) for i in range(1000)],
        class_mode='categorical')

# test_generator = test_datagen.flow_from_directory(
#         'F://dataset/testset',
#         target_size=(128, 128),
#         batch_size=32,
#         class_mode='categorical')


# # 训练

# In[39]:


BATCH_SIZE = 32
EPOCHS = 30


# In[40]:


model.fit_generator(
        train_generator, 
        steps_per_epoch = BATCH_SIZE,
        epochs = EPOCHS,
        validation_data = validation_generator, 
        validation_steps = BATCH_SIZE,
#         callbacks = [EarlyStopping(monitor='val_loss',patience=0,verbose=0,mode='auto')])
)


# # 保存参数，反复训练

# In[80]:


weights_path = 'mobile.h5'
model.save_weights(weights_path)


# In[86]:


base_model = applications.mobilenet.MobileNet(weights='imagenet', include_top=False, input_shape=(128,128,3))
x = base_model.output
x = Dropout(0.25)(x)
x = GlobalAveragePooling2D()(x)
pre = Dense(1000, activation='softmax')(x)
 
model = Model(inputs=base_model.input, outputs=pre)


# In[87]:


model.load_weights(weights_path)


# In[88]:


model.compile(optimizer=Adam(), 
              loss="categorical_crossentropy",
              metrics=['accuracy'])


# In[89]:


EPOCHS = 30


# In[7]:


model.fit_generator(
        train_generator, 
        steps_per_epoch = BATCH_SIZE,
        epochs = EPOCHS,
        validation_data = validation_generator, 
        validation_steps = BATCH_SIZE,
#         callbacks = [EarlyStopping(monitor='val_loss',patience=0,verbose=0,mode='auto')])
)


# In[48]:


weights_path = 'mobile2.h5'
model.save_weights(weights_path)


# # 预测

# In[49]:


from keras.preprocessing import image


# In[56]:


predict = []
for i in range(20000):
    print(i)
    img_path='F://dataset/pretreat_test/'+str(i)+'.jpg'
    img = image.load_img(img_path)
    img = img.resize((128, 128))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    pre = model.predict(x) #预测
    pre = pre.tolist()
    a = pre[0]  #1000个概率
    a = a.index(max(a))  #选择最大的作为最终预测结果
    predict.append(a)


# In[52]:


# name = []
# for i in range(1000):
#     name.append(str(i))
# name.sort()


# In[53]:


# result = []
# for i in range(len(predict)):
#     a = predict[i]
#     result.append(name[a])


# In[55]:


predict


# In[57]:


numId= [] 


# In[58]:


for i in range(20000):
    numId.append(i)


# In[59]:


dataframe = pd.DataFrame({'id':numId,'y':predict})
dataframe.to_csv("F://dataset/testy.csv",index=False,sep=',')


# # 一晚上的任务

# In[60]:


BATCH_SIZE = 32
EPOCHS = 10


# In[61]:


base_model = applications.mobilenet.MobileNet(weights='imagenet', include_top=False, input_shape=(128,128,3))
x = base_model.output
x = Dropout(0.5)(x)
x = GlobalAveragePooling2D()(x)
pre = Dense(1000, activation='softmax')(x)
 
model = Model(inputs=base_model.input, outputs=pre)

model.load_weights(weights_path)

model.compile(optimizer=Adam(), 
              loss="categorical_crossentropy",
              metrics=['accuracy'])

model.fit_generator(
        train_generator, 
        steps_per_epoch = BATCH_SIZE,
        epochs = EPOCHS,
        validation_data = validation_generator, 
        validation_steps = BATCH_SIZE,
#         callbacks = [EarlyStopping(monitor='val_loss',patience=0,verbose=0,mode='auto')])
)


# In[ ]:


weights_path = 'mobile.h5'
model.save_weights(weights_path)


# In[ ]:


predict = []
for i in range(20000):
    print(i)
    img_path='F://dataset/pretreat_test/'+str(i)+'.jpg'
    img = image.load_img(img_path)
    img = img.resize((128, 128))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    pre = model.predict(x) #预测
    pre = pre.tolist()
    a = pre[0]
    a = a.index(max(a))  #输出最大值下标
    predict.append(a)


# In[ ]:


name = []
for i in range(1000):
    name.append(str(i))
name.sort()


# In[ ]:


result = []
for i in range(len(predict)):
    a = predict[i]
    result.append(name[a])


# In[ ]:


dataframe = pd.DataFrame({'id':numId,'y':result})
dataframe.to_csv("F://dataset/testy2.csv",index=False,sep=',')


# In[ ]:


b_e[0][1]


# In[ ]:


te = iter(train_generator)
b_e = next(te)


# In[8]:


from PIL import Image


# In[31]:


Image.fromarray((b_e[0][2] * 255).astype(np.uint8))


# In[32]:


np.argmax(b_e[1], axis=1)[2]


# In[20]:


get_ipython().run_line_magic('pinfo2', 'train_datagen.flow_from_directory')

