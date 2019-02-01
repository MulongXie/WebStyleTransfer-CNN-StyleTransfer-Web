import tensorflow as tf
import os
import ui_nst_model as model
import ui_nst_image as image
import ui_nst_utils as nu

style_img = image.load_img("style_nahan.jpg")
style_img = nu.reshape_and_normalize_image(style_img)

with tf.Session() as sess:
    vgg = nu.load_vgg_model("D:/datasets/VGG/pretrained-model/imagenet-vgg-verydeep-19.mat")

    # content_img = image.load_img("logo.png")
    # content_img = nu.reshape_and_normalize_image(content_img)
    # sess.run(vgg['input'].assign(content_img))
    # model.NST_model(content_img, style_img, sess, vgg, "output1", 500)

    for i in range(2):
        content_img = image.load_img("con_button" + str(i) + ".jpg")
        content_img = nu.reshape_and_normalize_image(content_img)
        sess.run(vgg['input'].assign(content_img))

        output = "output" + str(i)
        if not os.path.exists(output):
            os.makedirs(output)
        model.NST_model(content_img, style_img, sess, vgg, output, 500)

