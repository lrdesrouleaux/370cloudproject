from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2 as cv
import os, random
ap=argparse.ArgumentParser()
#model location
ap.add_argument("-m", "--model",default="/Users/lens/Documents/resnet/cloudtrained/resnetmodel" ,required=False,
	help="path to trained model model")
#label locations
ap.add_argument("-l", "--labelbin",default="/Users/lens/Documents/resnet/cloudtrained/resnetasllabels.pickle" ,required=False,
	help="path to label binarizer")
#image locations
ap.add_argument("-i", "--imagepath",default="/Users/lens/Documents/resnet/dataset/archive/asl_alphabet_test/asl_alphabet_test" ,
            required=False,
	help="path to input image")
args = vars(ap.parse_args())
#set the files to a list
files=os.listdir(args["imagepath"])
while(True):
    #pick a random index from that list
	index=random.randrange(0,len(files)-1)
	print(files[index])
	#set it to a a string
	imagefile=args["imagepath"]+"/"+files[index]
	print(imagefile)
	image=cv.imread(imagefile)
	if image is None:
		print("image error trying to grab:",imagefile)
		break
	output=image.copy()
	#change image to model spec
	#convert the image to a 3d numpy array
	image=img_to_array(image)
	#expand the array to make it 3 dimensions
	image=np.expand_dims(image,axis=0)
	#run the model
	print("<info> loading model...")
	model=load_model(args["model"])
	lb=pickle.loads(open(args["labelbin"],"rb").read())
	print("<info> classifying image")
	proba=model.predict(image)
	idx=np.argmax(proba)
	label=lb.classes_[idx]
	filename = files[index][files[index].rfind(os.path.sep) + 1:]
	output = imutils.resize(output, width=400)
	cv.putText(output, label, (10, 25),  cv.FONT_HERSHEY_PLAIN,
	2, (255, 255, 255), 2)
	print("[INFO] {}".format(label))
	cv.imshow("Output", output)
	cv.waitKey()
	cv.destroyAllWindows()