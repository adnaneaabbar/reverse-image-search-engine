#Import dependencies
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import pickle
import config as cfg

from model import ImageSearchModel
from inference import simple_inference

#import Flask dependencies
from flask import Flask, request, render_template, send_from_directory

#Set root dir
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

#Define our model
model = ImageSearchModel(learning_rate=cfg.LEARNING_RATE,
                         size=cfg.IMAGE_SIZE,
                         number_of_classes=cfg.NUMBER_OF_CLASSES)

#Start tf.Session()
session = tf.Session()
session.run(tf.global_variables_initializer())
#Restore session
saver = tf.train.Saver()
saver.restore(session, 'checkpoints/model_epoch_3.ckpt')

#Load training set vectors
with open('pickle_saves/hamming_train_vectors.pickle', 'rb') as f:
    train_vectors = pickle.load(f)

#Load training set paths
with open('pickle_saves/train_images_pickle.pickle', 'rb') as f:
    train_images_paths = pickle.load(f)

#Define Flask app
app = Flask(__name__, static_url_path='/static')


#Define apps home page
@app.route("/")  #www.image-search.com/
def index():
    return render_template("index.html")


#Define upload function
@app.route("/upload", methods=["POST"])
def upload():

    upload_dir = os.path.join(APP_ROOT, "uploads/")

    if not os.path.isdir(upload_dir):
        os.mkdir(upload_dir)

    for img in request.files.getlist("file"):
        img_name = img.filename
        destination = "/".join([upload_dir, img_name])
        img.save(destination)

    #inference
    result = np.array(train_images_paths)[simple_inference(
        model, session, train_vectors, os.path.join(upload_dir, img_name),
        cfg.IMAGE_SIZE)]

    result_final = []

    for img in result:
        result_final.append(
            "images/" + img.split("/")[-1]
        )  #example: dataset/train/0_frog.png -> [dataset, train, 0_frog.png] -> [-1] = 0_frog.png

    return render_template(
        "results.html", image_name=img_name, result_paths=result_final[:-2]
    )  #added [:-2] just to have equal number of images in the result page per row


#Define helper function for finding image paths
@app.route("/upload/<filename>")
def send_image(filename):
    return send_from_directory("uploads", filename)


#Start the application

if __name__ == "__main__":
    app.run(port=5000, debug=True)
