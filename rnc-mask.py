import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#for images
import PIL
#creating dir
import os
import urllib.request
#capture camera
from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode
from google.colab import files


folder_path1 = '/content/drive/My Drive/Colab Notebooks/dataset/with_mask/'
folder1 = sorted(os.listdir(folder_path1))
del folder1[0] 

folder_path2 = '/content/drive/My Drive/Colab Notebooks/dataset/without_mask/'
folder2 = sorted(os.listdir(folder_path2))
del folder2[0] 

print('folder1 : with mask = ', folder1)
print('folder2 : without mask = ', folder2)


with_mask_data = np.empty([0,100 ,100, 3])

i=0
for x in folder1:
  opened_fig = PIL.Image.open(folder_path1 + x)
  opened_fig = opened_fig.resize((100, 100))
  try:
    with_mask_data = np.concatenate((with_mask_data, np.array(opened_fig).reshape(-1, 100,100,3)), axis=0)
    print('----------------------',i)
  except:
    print('!!!!!!!!!  -> ', i)
    pass
  i=i+1
  print(with_mask_data.shape)



print('with_mask_data = ',with_mask_data.shape)


without_mask_data = np.empty([0,100 ,100, 3])

i=0
for x in folder2:
  opened_fig = PIL.Image.open(folder_path2 + x)
  opened_fig = opened_fig.resize((100, 100))
  try:
    without_mask_data = np.concatenate((without_mask_data, np.array(opened_fig).reshape(-1, 100,100,3)), axis=0)
    print('----------------------',i)
  except:
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  -> ', i)
    pass
  i=i+1
  print(without_mask_data.shape)



print('without_mask_data.shape = ',without_mask_data.shape)


with_mask_data_targets = np.ones(with_mask_data.shape[0])
print(with_mask_data_targets.shape)
without_mask_data_targets = np.zeros(without_mask_data.shape[0])
print(without_mask_data_targets.shape)


data  = np.concatenate((with_mask_data, without_mask_data), axis=0)
targets = np.append(with_mask_data_targets, without_mask_data_targets)
print('data.shape = ', data.shape)
print('targets.shape = ', targets.shape)


permutation = np.random.permutation(targets.shape[0])
data = data[permutation, :]
targets = targets[permutation]
print('data.shape = ', data.shape)
print('targets.shape = ', targets.shape)

data = data/255
print(data)


test_size  = int(data.shape[0]/100*(20))
data_test = data[0:test_size, :]
targets_test = targets[0:test_size]

data_train = data[test_size:data.shape[0], :]
targets_train = targets[test_size:targets.shape[0]]

print('data_train = ', data_train.shape)
print('targets_train = ', targets_train.shape)
print('data_test = ', data_test.shape)
print('targets_test = ', targets_test.shape)


rando = np.random.randint(len(data_train)-1)

imgplot = plt.imshow(data_train[rando])
print(targets_train[rando])
print(data_train[rando].shape)


#creat model
model = tf.keras.models.Sequential()
#add  operation to our model
model.add(tf.keras.layers.Conv2D(16, 3, padding= 'same', activation = 'relu') )
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Conv2D(32, 3,padding= 'same', activation = 'relu'))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Conv2D(64, 3, padding= 'same',  activation = 'relu'))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation = 'relu'))
model.add(tf.keras.layers.Dense(2, activation = 'softmax'))
model_output = model.predict(data[10:11])
print(model_output)

model.summary()


#compile the model
model.compile(loss = 'sparse_categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])


#train the model
history = model.fit(data_train,targets_train, epochs=10, batch_size= 32, validation_split= 0.2)


#tracer les courbes de train
loss_curve = history.history['loss']
acc_curve = history.history['accuracy']

loss_val_curve = history.history['val_loss']
acc_val_curve = history.history['val_accuracy']

f1 = plt.figure(1)
plt.plot(loss_curve, label='train')
plt.plot(loss_val_curve , label='val')
plt.legend(loc='upper left')
plt.title('loss')

f2 = plt.figure(2)
plt.plot(acc_curve , label='train')
plt.plot(acc_val_curve , label='val')
plt.legend(loc='upper left')
plt.title('acc')
plt.show


end_test = len(data_test)
result = model.evaluate(data_test[0:end_test], targets_test[0:end_test])
print(round(result[1]*(end_test)), 'good result from : ', (end_test))


#save model
model.save('simple_nn.h5')

#charge model
my_model = tf.keras.models.load_model('simple_nn.h5')


#test on the test_data (images_test and targets_test)
rand = np.random.randint(len(data_test)-1)
my_image = data_test[rand:rand+1]
my_target = targets_test[rand:rand+1]
predict = my_model.predict(my_image)
print('Predict of image N° {} : {}'. format(rand,np.argmax(predict)))
print('Result  of image N° {} : {}'. format(rand,int(my_target)))


# Google code for camera capture
def take_photo(filename='photo.jpg', quality=0.8):
  js = Javascript('''
    async function takePhoto(quality) {
      const div = document.createElement('div');
      const capture = document.createElement('button');
      capture.textContent = 'Capture';
      div.appendChild(capture);

      const video = document.createElement('video');
      video.style.display = 'block';
      const stream = await navigator.mediaDevices.getUserMedia({video: true});

      document.body.appendChild(div);
      div.appendChild(video);
      video.srcObject = stream;
      await video.play();

      // Resize the output to fit the video element.
      google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

      // Wait for Capture to be clicked.
      await new Promise((resolve) => capture.onclick = resolve);

      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      stream.getVideoTracks()[0].stop();
      div.remove();
      return canvas.toDataURL('image/jpeg', quality);
    }
    ''')
  display(js)
  data = eval_js('takePhoto({})'.format(quality))
  binary = b64decode(data.split(',')[1])
  with open(filename, 'wb') as f:
    f.write(binary)
  return filename


  #Test model on camera
  # take a picture
#filename = take_photo() 
  #upload a picture from local
#filename = list(files.upload().keys())[0]


image = PIL.Image.open(filename)
image = image.resize((100, 100))
imgplot = plt.imshow(image)

image = (np.array(image).reshape(-1,100,100,3))/255


predict = my_model.predict(image)
print(predict)
if (np.argmax(predict)== 1):
 print('Predict of {} is {} % MASK'.format(filename, int(np.max(predict)*100)))
elif (np.argmax(predict) == 0):
 print('Predict of {} is {} % NO MASK'.format(filename, int(np.max(predict)*100)))
else:
 print('Erreur')

aaa = PIL.Image.open('no_mask (2).jpg')
aaa = aaa.resize((100, 100))
aaa = (np.array(aaa).reshape(-1,100,100,3))/255
ppp = my_model.predict(aaa) 
print(int(np.max(ppp)*100))
print(np.argmax(ppp))


imm = [0]*100
for i in range(1,6):
  imm[i] = PIL.Image.open('mask ({}).jpg'.format(i))
  imm[i] = imm[i].resize((100, 100))
  imm[i] = (np.array(imm[i]).reshape(-1,100,100,3))/255
  pred = my_model.predict(imm[i])
  if (np.argmax(pred)== 1):
    print('Predict of {} is {} % MASK'.format(i, int(np.max(pred)*100)))
  elif (np.argmax(pred) == 0):
    print('Predict of {} is {} % NO MASK'.format(i, int(np.max(pred)*100)))
  else:
    print('Erreur')
