from PIL import Image
import numpy as np
import tensorflow as tf

# colour map
label_colours = [(0,0,0) ,(128,0,0),(0,128,0),(128,128,0),(0,0,128),(128,0,128)
                ,(0,128,128),(128,128,128),(64,0,0),(192,0,0),(64,128,0)
                ,(192,128,0),(64,0,128),(192,0,128),(64,128,128),(192,128,128)
                ,(0,64,0),(128,64,0),(0,192,0),(128,192,0),(0,64,128)]

def decode_labels(mask, num_images=1, num_classes=21):
    n, h, w, c = mask.shape
    assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
      img = Image.new('RGB', (len(mask[i, 0]), len(mask[i]))) # Size is given as a (width, height)-tuple.
      pixels = img.load()
      for j_, j in enumerate(mask[i, :, :, 0]):
          for k_, k in enumerate(j):
              if k < num_classes:
                  pixels[k_,j_] = label_colours[k]
      outputs[i] = np.array(img)
    return outputs

def prepare_label(input_batch, new_size, num_classes, one_hot=True):
    with tf.name_scope('label_encode'):
        input_batch = tf.image.resize_nearest_neighbor(input_batch, new_size) # as labels are integer numbers, need to use NN interp.
        input_batch = tf.squeeze(input_batch, squeeze_dims=[3]) # reducing the channel dimension.
        if one_hot:
            input_batch = tf.one_hot(input_batch, depth=num_classes)
    return input_batch

def inv_preprocess(imgs, num_images, img_mean):
    n, h, w, c = imgs.shape
    assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    outputs = np.zeros((num_images, h, w, c), dtype=np.uint8)
    for i in range(num_images):
        outputs[i] = (imgs[i] + img_mean)[:, :, ::-1].astype(np.uint8)
    return outputs
