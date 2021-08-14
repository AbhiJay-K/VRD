import numpy
import cv2
import numpy as np


def create_gaussian_image_pyramid(image, pyramid_levels,minSize=(30, 30)):
    gauss_copy = numpy.ndarray(shape=image.shape, dtype="float")
    gauss_copy[:] = image
    img_pyramid = [gauss_copy]
    for pyramid_level in range(0,pyramid_levels):
        rows, cols, _channels = map(int, gauss_copy.shape)
        gauss_copy = cv2.pyrDown(gauss_copy,dstsize=(cols // 2, rows // 2))
        img_pyramid.append(gauss_copy)
    return img_pyramid

def create_gaussian_image_pyramid_gray(image, pyramid_levels,minSize=(30, 30)):
    gauss_copy = numpy.ndarray(shape=image.shape, dtype="float")
    gauss_copy[:] = image
    img_pyramid = [gauss_copy]
    for pyramid_level in range(0,pyramid_levels):
        rows, cols = map(int, gauss_copy.shape)
        gauss_copy = cv2.pyrDown(gauss_copy,dstsize=(cols // 2, rows // 2))
        img_pyramid.append(gauss_copy)
    return img_pyramid

def create_gaussian_image_pyramid_nochannel(image, pyramid_levels,minSize=(30, 30)):
    gauss_copy = numpy.ndarray(shape=image.shape, dtype="float")
    gauss_copy[:] = image
    img_pyramid = [gauss_copy]
    for pyramid_level in range(0,pyramid_levels):
        rows, cols = map(int, gauss_copy.shape)
        gauss_copy = cv2.pyrDown(gauss_copy,dstsize=(cols // 2, rows // 2))
        img_pyramid.append(gauss_copy)
    return img_pyramid

#Build Gaussian Pyramid
def build_gaussian_pyramid(src,level=3):
    s=src.copy()
    pyramid=[s]
    for i in range(level):
        s=cv2.pyrDown(s)
        pyramid.append(s)
    return pyramid

def create_laplacian_image_pyramid(image, pyramid_levels):
    gauss_pyramid = create_gaussian_image_pyramid(image, pyramid_levels)
    laplacian_pyramid = []
    for i in range(0,pyramid_levels):
        size = (gauss_pyramid[i].shape[1], gauss_pyramid[i].shape[0])
        laplacian_pyramid.append((gauss_pyramid[i] - cv2.pyrUp(gauss_pyramid[i + 1],dstsize=size)) + 0)

    laplacian_pyramid.append(gauss_pyramid[-1])
    return laplacian_pyramid


def create_gaussian_video_pyramid(video, pyramid_levels):
    return _create_pyramid(video, pyramid_levels, create_gaussian_image_pyramid)

def create_gaussian_video_pyramid_gray(video, pyramid_levels):
    return _create_pyramid_gray(video, pyramid_levels, create_gaussian_image_pyramid_gray)


def create_laplacian_video_pyramid(video, pyramid_levels):
    return _create_pyramid(video, pyramid_levels, create_laplacian_image_pyramid)

def create_laplacian_video_pyramid_gray(video, pyramid_levels):
    return _create_pyramid_gray(video, pyramid_levels, create_laplacian_image_pyramid)


def _create_pyramid(video, pyramid_levels, pyramid_fn):
    vid_pyramid = []
    # frame_count, height, width, colors = video.shape
    for frame_number, frame in enumerate(video):
        frame_pyramid = pyramid_fn(frame, pyramid_levels)

        for pyramid_level, pyramid_sub_frame in enumerate(frame_pyramid):
            if frame_number == 0:
                vid_pyramid.append(
                    numpy.zeros((video.shape[0], pyramid_sub_frame.shape[0], pyramid_sub_frame.shape[1], 3),
                                dtype="float"))

            vid_pyramid[pyramid_level][frame_number] = pyramid_sub_frame

    return vid_pyramid

def _create_pyramid_gray(video, pyramid_levels, pyramid_fn):
    vid_pyramid = []
    # frame_count, height, width, colors = video.shape
    for frame_number, frame in enumerate(video):
        frame_pyramid = pyramid_fn(frame, pyramid_levels)

        for pyramid_level, pyramid_sub_frame in enumerate(frame_pyramid):
            if frame_number == 0:
                vid_pyramid.append(
                    numpy.zeros((video.shape[0], pyramid_sub_frame.shape[0], pyramid_sub_frame.shape[1]),
                                dtype="float"))

            vid_pyramid[pyramid_level][frame_number] = pyramid_sub_frame

    return vid_pyramid


def collapse_laplacian_pyramid(image_pyramid):
    img = image_pyramid.pop()
    while image_pyramid:
        next_img = image_pyramid.pop()
        rows, cols, _channels = map(int, next_img.shape)
        img = cv2.pyrUp(img,dstsize=(cols, rows))
        img = img + (next_img - 0)
    return img


def collapse_g_pyramid(image_pyramid):
    img = image_pyramid.pop()
    while image_pyramid:
        rows, cols, _channels = map(int, image_pyramid.pop().shape)
        img = cv2.pyrUp(img,dstsize=(cols, rows))
    return img


def collapse_laplacian_video_pyramid(pyramid):
    i = 0
    while True:
        try:
            img_pyramid = [vid[i] for vid in pyramid]
            pyramid[0][i] = collapse_laplacian_pyramid(img_pyramid)
            i += 1
        except IndexError:
            break
    return pyramid[0]



def collapse_g_video_pyramid(pyramid):
    i = 0
    while True:
        try:
            img_pyramid = [vid[i] for vid in pyramid]
            pyramid[0][i] = collapse_g_pyramid(img_pyramid)
            i += 1
        except IndexError:
            break
    return pyramid[0]


#reconstract video from original video and gaussian video
def reconstract_video(amp_video, origin_video, levels=None):
    final_video=np.zeros(origin_video.shape)
    for i in range(0, amp_video.shape(0)):
        img = amp_video[i]
        for x in range(levels):
             img=cv2.pyrUp(img)
        img=img+origin_video[i]
        final_video[i]=img
    return final_video