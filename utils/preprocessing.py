import tensorflow as tf

def load_image(path, img_size):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (img_size, img_size))
    img = tf.cast(img, tf.float32) / 255.0
    return img

def load_mask(path, img_size):
    mask = tf.io.read_file(path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, (img_size, img_size))
    mask = tf.cast(mask, tf.float32) / 255.0
    return mask

def preprocess_image_mask(img_path, mask_path, img_size):
    img = load_image(img_path, img_size)
    mask = load_mask(mask_path, img_size)
    return img, mask

def preprocess_image_only(img_path, img_size):
    img = load_image(img_path, img_size)
    return img
