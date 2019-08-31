import numpy as np 
from PIL import Image

from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt

#%matplotlib inline
def extract_face_from_image(filename,required_size=(224, 224)):
	image=plt.imread(filename)
	detector=MTCNN()
	result_image=detector.detect_faces(image)
	x1,y1,width,height=result_image[0]['box']
	x2,y2=x1+width,y1+height
	face_roi=image[y1:y2,x1:x2]
	image_face = Image.fromarray(face_roi)
	image_face = image_face.resize(required_size)
	image_face = np.asarray(image_face)
	return image_face

def get_images():
	X,Y=[],[]
	train_data_dir="/home/hardik/Desktop/data/"
	train_data=[train_data_dir+i for i in os.listdir(train_data_dir)]
	for img in train_data:
		if "hardik" in img:
			Y.append(0)
		elif "anmol" in img:
			Y.append(1)
		elif "atul" in img:
			Y.append(2)
		elif "himanshu" in img:
			Y.append(3)
		elif "mike" in img:
			Y.append(4)
		elif "swati" in img:
			Y.append(5)
		image_face=extract_face_from_image(img)
		image_face = image_face.astype('float32')
		image_face= np.expand_dims(image_face, axis=0)
# prepare the face for the model, e.g. center pixels
		image_face = preprocess_input(image_face, version=2)
		X.append(image_face)
	X=np.array(X)
	Y=np.array(Y)
	return X,Y
def model_train(X,Y):
	Y= keras.utils.to_categorical(Y, 4)

	resnet = VGGFace(model='resnet50',input_shape=(224, 224, 3))

	layer_name = resnet.layers[-2].name

	out = resnet.get_layer(layer_name).output
	out = Dense(4,activation='softmax')(out)
	resnet_4 = Model(resnet.input, out)

	for layer in resnet_4.layers[:-1]:
		layer.trainable = False

	resnet_4.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

	print (resnet_4.summary())

	resnet_4.fit(X,Y,batch_size=16,epochs=5,shuffle=True)
	resnet_4.save("/home/hardik/Desktop/model_face.h5")


X,Y=get_images()
model_train(X,Y)
