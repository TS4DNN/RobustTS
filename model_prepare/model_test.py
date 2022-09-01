import tensorflow as tf
from traffic_models import *


model = tf.keras.models.load_model("../models/traffic/lenet5.h5")
# model.summary()
# model.save("../models/traffic/vgg16.h5")

test_folder_path = "../datasets/traffic/Test_2/"
test_ds = traffic_generator(test_folder_path, 128)
score = model.evaluate(test_ds)

print(score)
