import tensorflow as tf
from cifar10_vgg16 import *
from ResNet import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class CustomImageDataGen(ImageDataGenerator):  # Inheriting class ImageDataGenerator and manually standardize each input image (x)
    def standardize(self, x):
        if self.featurewise_center:
            x /= 255.
        return x


def kitti_generator(test_folder):
    data_generator = CustomImageDataGen(
        horizontal_flip=True,
        featurewise_center=True
    )
    test_data_generator = data_generator.flow_from_directory(test_folder,
                                                             target_size=(384, 1248),
                                                             batch_size=5,
                                                             seed=1,
                                                             shuffle=True)
    return test_data_generator


def load_kitti_data():
    img_height = 384
    img_width = 1248
    batch_size = 5
    folder_path = "../datasets/kitti/train/"
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        folder_path,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    folder_path = "../datasets/kitti/test/"
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        folder_path,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    return train_ds, test_ds


def kitti_vgg16_model():
    input_shape = (384, 1248, 3)
    model = VGG16_clipped(input_shape=input_shape, rate=0.2, nb_classes=3, drop=False)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Adam(lr=1e-5),
                  metrics=['accuracy'])
    train_ds = kitti_generator("../datasets/kitti/train/")
    test_ds = kitti_generator("../datasets/kitti/test/")
    steps_per_epoch = 289 // 5
    model.fit(
        train_ds,
        steps_per_epoch=steps_per_epoch,
        epochs=10,
    )
    model.save("test.h5")
    score = model.evaluate(test_ds)
    print(score)


def kitti_resnet20_model():
    input_shape = (384, 1248, 3)
    model = resnet20(input_shape, 3)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Adam(lr=1e-5),
                  metrics=['accuracy'])
    train_ds = kitti_generator("../datasets/kitti/train/")
    test_ds = kitti_generator("../datasets/kitti/test/")
    steps_per_epoch = 289 // 5
    model.fit(
        train_ds,
        steps_per_epoch=steps_per_epoch,
        epochs=10,
    )
    # model.save("test.h5")
    score = model.evaluate(test_ds)
    print(score)


def model_eval():
    train_ds, test_ds = load_kitti_data()
    model = tf.keras.models.load_model("test.h5")
    print(model.evaluate(train_ds))
    print(model.evaluate(test_ds))


if __name__ == "__main__":
    # train_ds = kitti_generator("../datasets/kitti/train/")
    # # kitti_resnet20_model()
    # # model_eval()
    # for x, y in train_ds:
    #     print(x[0])
    #     print(y.shape)
    #     break
    # test_ds = kitti_generator("../datasets/kitti/test/")
    # for x, y in test_ds:
    #     print(x[0])
    #     print(y.shape)
    #     break
    kitti_vgg16_model()
