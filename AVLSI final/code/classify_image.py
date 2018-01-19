from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import Xception # TensorFlow ONLY
from keras.applications import VGG16
from keras.applications import VGG19
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
import argparse
import time
#import cv2


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-model", "--model", type=str, default="vgg16",
	help="name of pre-trained network to use")
args = vars(ap.parse_args())

# define a dictionary that maps model names to their classes
# inside Keras
MODELS = {
	"vgg16": VGG16,
	"vgg19": VGG19,
	"inception": InceptionV3,
	"xception": Xception, # TensorFlow ONLY
	"resnet": ResNet50
}

# esnure a valid model name was supplied via command line argument
if args["model"] not in MODELS.keys():
	raise AssertionError("The --model command line argument should "
		"be a key in the `MODELS` dictionary")

# initialize the input image shape (224x224 pixels) along with
# the pre-processing function (this might need to be changed
# based on which model we use to classify our image)
inputShape = (224, 224)
preprocess = imagenet_utils.preprocess_input
 
if args["model"] in ("inception", "xception"):
	inputShape = (299, 299)
	preprocess = preprocess_input

print("[INFO] loading {}...".format(args["model"]))
Network = MODELS[args["model"]]
model = Network(weights="imagenet")
print("[INFO] basic information of '{}' model".format(args["model"]))


print("[INFO] loading and pre-processing image...")

label = np.loadtxt("label/val_id.txt", dtype=str)
num_image = label.shape[0]
top1 = 0.0
top5 = 0.0
error1 = ''
error5 = ''


label = label.tolist()
for k in range(1,num_image+1):

    path = "ILSVRC2012_img_val/ILSVRC2012_val_000"
    if k < 10:
        pic_name = '0000'+str(k)
    elif 10 <= k and k < 100:
        pic_name = '000'+str(k)
    elif 100 <= k and k < 1000:
        pic_name = '00'+str(k)
    elif 1000 <= k and k < 10000:
        pic_name = '0' +str(k)
    else:
        pic_name = str(k)

    pic_path = path + pic_name + ".JPEG"

    image = load_img(pic_path, target_size=inputShape)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    image = preprocess(image)

    #print("[INFO] classifying image with '{}'...".format(args["model"]))
    preds = model.predict(image)

    P = imagenet_utils.decode_predictions(preds)
    if label[k-1] in P[0][0]:
        top1 += 1
    else:
        error1 += str(k) + '\n'

    inTop5 = False
    for (i, (imagenetID, pred_label, prob)) in enumerate(P[0]):
        if label[k-1] == imagenetID:
            top5 += 1
            inTop5 = True

    if (k%10 == 0):
            print("Progress: {}/50000".format(k),"top1 acc:", round(top1/k,4),"top5 acc:", round(top5/k,4))

    if inTop5 == False:
        error5 += str(k) + '\n'

print('total image', num_image)
print('top_1',top1/num_image)
print('top_5',top5/num_image)
of1 = open("./record/"+args["model"]+ "_error1.txt",'w')
of2 = open("./record/"+args["model"]+ "_error5.txt",'w')
of1.write(error1)
of1.close()
of2.write(error5)
of2.close()


'''
orig = cv2.imread(args["image"])
(imagenetID, label, prob) = P[0][0]
cv2.putText(orig, "Label: {}, {:.2f}%".format(label, prob * 100),
	(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
cv2.imshow("Classification", orig)
cv2.waitKey(0)
'''
