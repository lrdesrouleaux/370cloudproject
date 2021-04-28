# using plaidml backend inorder to use my gpu on training
import cv2 as cv  # image functions
import random  # used to shuffle and reorder data
import pickle  # labels
import argparse  # argument parser
import numpy as np  # numpy arrays and such
from imutils import paths  # image utilities
import matplotlib  # easy plots
import os  # needed to scour paths for data
from keras.models import Model
from resnet import ResNet
import matplotlib.pyplot as plt  # easy plots
# fast training and testing on one dataset, I'll classify on the test dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.image import img_to_array
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import  ImageDataGenerator
matplotlib.use("Agg")
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", default="/Users/lens/Documents/resnet/dataset/archive/asl_alphabet_train/dummy", required=False,
                help="path to input dataset (i.e., directory of images)")
ap.add_argument("-m", "--model", default="/Users/lens/Documents/resnet/resnetmodel", required=False,
                help="path to output model")
ap.add_argument("-l", "--labelbin", default="resnetasllabels.pickle", required=False,
                help="path to output label binarizer")
ap.add_argument("-e", "--epochs", default=100, required=False,
                help="Number of epochs that will be commited")
ap.add_argument("-p", "--plot", type=str, default="resnetplot.png",
                help="path to output accuracy/loss plot")
args = vars(ap.parse_args())
EPOCHS = int(args["epochs"])
INITIAL_LEARNINGRATE = 1E-3
BATCH_SIZE = 4
IMAGE_DIMS = (200, 200, 3)
data = []
labels = []
print("--loading images--")
imagepaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagepaths)
for imagepath in imagepaths:
    # input dataset images into the data
    image = cv.imread(imagepath)
    print('---loading in image {}'.format(imagepath))
    image = img_to_array(image)
    data.append(image)
    # find class label from the image path and update the list
    label = imagepath.split(os.path.sep)[-2]
    print('---labeling the image {}'.format(label))
    labels.append(label)
print("--converting label data to arrays")
labels = np.asarray(labels)
# display size of data
print("-->data matrix: {:.2f} megabytes".format(data.__sizeof__()/1000000))
# binarize the labels
lb = LabelBinarizer()
print("converting data to np")
data = np.asarray(data)
labels = lb.fit_transform(labels)
(trainx, testx, trainy, testy) = train_test_split(
    data, labels, test_size=.2, random_state=42)
idg = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")
print("--compiling model--")
model = ResNet.build(width=IMAGE_DIMS[0], height=IMAGE_DIMS[1], depth=IMAGE_DIMS[2],
                     classes=len(lb.classes_), stages=(3, 4, 6), filters=(8, 16, 32, 64))
model.compile(optimizer=Adam(learning_rate=INITIAL_LEARNINGRATE, decay=INITIAL_LEARNINGRATE/EPOCHS),
               loss="categorical_crossentropy",metrics=['accuracy'])
print("--training network--")
TEMPMODEL = model.fit(
    x=idg.flow(trainx, trainy, batch_size=BATCH_SIZE),
    epochs=EPOCHS, verbose=1, validation_data=(testx, testy), steps_per_epoch=len(trainx)//BATCH_SIZE)
print("<info> serializing network...")
model.save(args["model"])
labelfile = open(args["labelbin"], "wb")
labelfile.write(pickle.dumps(lb))
labelfile.close()
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
# write data plots
plt.plot(np.arange(0, N), TEMPMODEL.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), TEMPMODEL.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), TEMPMODEL.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), TEMPMODEL.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig(args["plot"])
