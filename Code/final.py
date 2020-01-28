
# coding: utf-8

# In[8]:


## Import library
import pandas as pd
import numpy as np
from shapely.wkt import loads           # https://pypi.org/project/Shapely/
from matplotlib.patches import Polygon, Patch  # plot Polygon using Matplotlib:https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.patches.Polygon.html
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_similarity_score
from collections import defaultdict
from shapely.geometry import MultiPolygon, Polygon
import shapely.wkt
import shapely.affinity
import tifffile as tiff
import cv2
import keras
import random
from keras.models import Model, load_model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
#K.set_image_dim_ordering('th')
K.set_image_data_format('channels_first')


# In[24]:


# return all band of image, resize image and concanate all type of images
def all_image(image_id, dims=20, size=800):
    if dims==3:
        filename = "D:/Assignment/three_band/{}.tif".format(image_id)
        img = tiff.imread(filename)
        img = np.rollaxis(img, 0, 3)
        img = cv2.resize(img, (size, size))
        
    elif dims==8:
        filename = "D:/Assignment/sixteen_band/{}_M.tif".format(image_id)
        img = tiff.imread(filename)
        img = np.rollaxis(img, 0, 3)
        img = cv2.resize(img, (size, size))
        
    elif dims==20:
        # for type M 
        img_M = np.transpose(tiff.imread("D:/Assignment/sixteen_band/{}_M.tif".format(image_id)), (1,2,0))
        img_M = cv2.resize(img_M, (size, size))
        # for type A
        img_A = np.transpose(tiff.imread("D:/Assignment/sixteen_band/{}_A.tif".format(image_id)), (1,2,0))
        img_A = cv2.resize(img_A, (size, size))
        # for type P
        img_P = tiff.imread("D:/Assignment/sixteen_band/{}_P.tif".format(image_id))
        img_P = cv2.resize(img_P, (size, size))

        filename = "D:/Assignment/three_band/{}.tif".format(image_id)
        # for RGB 
        img_RGB = tiff.imread(filename)
        img_RGB = np.rollaxis(img_RGB, 0, 3)
        img_RGB = cv2.resize(img_RGB, (size, size))

        img = np.zeros((img_RGB.shape[0], img_RGB.shape[1], dims), "float32")
        img[..., 0:3] = img_RGB
        img[..., 3] = img_P
        img[..., 4:12] = img_M
        img[..., 12:21] = img_A
        
    return img


# In[3]:


def stretch_n(img, lower_percent=5, higher_percent=95):    # https://www.kaggle.com/aamaia/rgb-using-m-bands-example
    out = np.zeros_like(img, dtype=np.float32)
    n = img.shape[2]
    for i in range(n):
        a = 0       # np.min(img)
        b = 1       # np.max(img)
        c = np.percentile(img[:, :, i], lower_percent)
        d = np.percentile(img[:, :, i], higher_percent)
        t = a + (img[:, :, i] - c) * (b - a) / (d - c)
        t[t < a] = a
        t[t > b] = b
        out[:, :, i] = t
    return out.astype(np.float32)


# In[17]:


smooth = 1e-12
def jaccard_coef(y_true, y_pred):
    # # __author__ = Vladimir Iglovikov
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)

def jaccard_coef_int(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred_pos, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)


# In[15]:


ISZ = 160
N_Cls=10
def get_unet():
    inputs = Input((20, ISZ, ISZ))
    conv1 = Conv2D(32, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, 3, activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, 3, activation='relu', padding ='same')(pool4)
    conv5 = Conv2D(512, 3, activation='relu', padding ='same')(conv5)
    
    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=1)
    conv6 = Conv2D(256, 3, activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, 3, activation='relu', padding='same')(conv6)
    
    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=1)
    conv7 = Conv2D(128, 3, activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, 3, activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=1)
    conv8 = Conv2D(64, 3, activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, 3, activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=1)
    conv9 = Conv2D(32, 3, activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, 3, activation='relu', padding='same')(conv9)

    conv10 = Conv2D(N_Cls, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=[jaccard_coef, jaccard_coef_int, 'accuracy'])
    return model


# In[4]:


def mask_for_polygons(polygons, im_size):
    # __author__ = Konstantin Lopuhin
    # https://www.kaggle.com/lopuhin/dstl-satellite-imagery-feature-detection/full-pipeline-demo-poly-pixels-ml-poly
    img_mask = np.zeros(im_size, np.uint8)
    if not polygons:
        return img_mask
    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exteriors = [int_coords(poly.exterior.coords) for poly in polygons]
    interiors = [int_coords(pi.coords) for poly in polygons
                 for pi in poly.interiors]
    cv2.fillPoly(img_mask, exteriors, 1)
    cv2.fillPoly(img_mask, interiors, 0)
    return img_mask


# In[5]:


def mask_to_polygons(mask, epsilon=1, min_area=5):
    # https://www.kaggle.com/lopuhin/dstl-satellite-imagery-feature-detection/full-pipeline-demo-poly-pixels-ml-poly

    # first, find contours with cv2: it's much faster than shapely
    contours, hierarchy = cv2.findContours(((mask == 1) * 255).astype(np.uint8),cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    # create approximate contours to have reasonable submission size
    approx_contours = [cv2.approxPolyDP(cnt, epsilon, True) for cnt in contours]
    if not contours:
        return MultiPolygon()
    # now messy stuff to associate parent and child contours
    cnt_children = defaultdict(list)
    child_contours = set()
    assert hierarchy.shape[0] == 1
    # http://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html
    for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
        if parent_idx != -1:
            child_contours.add(idx)
            cnt_children[parent_idx].append(approx_contours[idx])
    # create actual polygons filtering by area (removes artifacts)
    all_polygons = []
    for idx, cnt in enumerate(approx_contours):
        if idx not in child_contours and cv2.contourArea(cnt) >= min_area:
            assert cnt.shape[1] == 1
            poly = Polygon(
                shell=cnt[:, 0, :],
                holes=[c[:, 0, :] for c in cnt_children.get(idx, [])
                       if cv2.contourArea(c) >= min_area])
            all_polygons.append(poly)
    # approximating polygons might have created invalid ones, fix them
    all_polygons = MultiPolygon(all_polygons)
    if not all_polygons.is_valid:
        all_polygons = all_polygons.buffer(0)
        # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
        # need to keep it a Multi throughout
        if all_polygons.type == 'Polygon':
            all_polygons = MultiPolygon([all_polygons])
    return all_polygons


# In[22]:


trs = [0.5, 0.4, 0.5, 0.4, 0.4, 0.7, 0.3, 0.6, 0.3, 0.3]
def predict_id(id, model, trs):
    img = all_image(id)
    x = stretch_n(img)
  
    cnv = np.zeros((960, 960, 20)).astype(np.float32)
    prd = np.zeros((N_Cls, 960, 960)).astype(np.float32)
    cnv[:img.shape[0], :img.shape[1], :] = x

    for i in range(0, 6):
        line = []
        for j in range(0, 6):
            line.append(cnv[i * ISZ:(i + 1) * ISZ, j * ISZ:(j + 1) * ISZ])

            x = 2 * np.transpose(line, (0, 3, 1, 2)) - 1
            tmp = model.predict(x, batch_size=4)
            for j in range(tmp.shape[0]):
                prd[:, i * ISZ:(i + 1) * ISZ, j * ISZ:(j + 1) * ISZ] = tmp[j]

    for i in range(N_Cls):
        prd[i] = prd[i] > trs[i]

    return prd[:, :img.shape[0], :img.shape[1]]


# In[32]:


def check_predict(id):
    # print('Please enter which class(0 to 9) you want to predict:')
    # i = int(input()) # Class 0  to 9
    model = get_unet()
    model.load_weights('weights/unet_10_jk0.7938')

    msk = predict_id(id, model, trs)
    img = all_image(id)
    img = stretch_n(img)
    plt.figure()
    #ax1 = plt.subplot(131)
    plt.title('image ID:6120_2_3')
    plt.imshow(img[:, :, 5])
    plt.show()
    #ax2 = plt.subplot(132)
    plt.title('predict bldg pixels')
    plt.imshow(msk[0], cmap=plt.get_cmap('gray'))
    plt.show() 
    #ax3 = plt.subplot(133)
    plt.title('predict bldg polygones')
    plt.imshow(mask_for_polygons(mask_to_polygons(msk[0], epsilon=1), img.shape[:2]), cmap=plt.get_cmap('gray'))

    plt.show()


# In[34]:

print('Enter Image id: ')
ids = input()
check_predict(ids)

