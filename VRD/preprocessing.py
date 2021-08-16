import os

import numpy as np
from cv2 import cv2
from numpy import zeros

from VRD.util import uint8_to_float, float_to_uint8, colour_hist_eq, \
    eulerian_magnification_colour, find_centers, contrast_and_brightness
from VRD.vrd_thread import VRDThread


def load_image(img_path):
    img_path = str(img_path)
    img = cv2.imread(img_path)
    return uint8_to_float(img)


def obtain_roi(video_filename, freq_min_narrow, freq_max_narrow, amplification,pyramid_levels):
    """Load a video into a numpy array"""
    video_filename = str(video_filename)
    print("Loading " + video_filename)
    if not os.path.isfile(video_filename):
        raise Exception("File Not Found: %s" % video_filename)
    # noinspection PyArgumentList
    capture = cv2.VideoCapture(video_filename)

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = get_capture_dimensions(capture)
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    x = 0
    green_frames = zeros((frame_count, height, width, 3), dtype='uint8')
    skin_frames = zeros((frame_count, height, width, 3), dtype='uint8')
    grey_frames = zeros((frame_count, height, width, 3), dtype='uint8')
    actual_vid_list = []
    thread_list = []
    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break
        thread_frame = VRDThread(get_skin_frm_frame, args=(frame,))
        thread_list.append(thread_frame)
        thread_frame.start()
        actual_vid_list.append(frame)

    capture.release()
    for thrd in thread_list:
        thrd.join()
        lst = thrd.get_result()
        green_frames[x] = lst[0]
        skin_frames[x] = lst[2]
        grey_frames[x] = lst[1]
        x += 1
    result = []
    result.append(fps)
    result.append(height)
    result.append(width)
    return_value_greem = wide_bandpass_filtering(uint8_to_float(green_frames), fps, freq_min_narrow,
                                                 freq_max_narrow, amplification, pyramid_levels=3)
    result.append(find_centers(return_value_greem))
    result.append(uint8_to_float(grey_frames))
    result.append(uint8_to_float(skin_frames))
    return result


def wide_bandpass_filtering(vid, fps, freq_min_narrow=0.88, freq_max_narrow=1.0, amplification=300, pyramid_levels=3):
    return eulerian_magnification_colour(vid, fps,
                                         freq_min_narrow=freq_min_narrow,
                                         freq_max_narrow=freq_max_narrow,
                                         amplification=amplification,
                                         pyramid_levels=pyramid_levels)


def get_skin_frm_frame(frame):
    equ = colour_hist_eq(frame)
    thread_YCrCb = VRDThread(get_YCrCb_mask, args=(equ,))
    thread_HSV = VRDThread(get_hsv_mask, args=(equ,))
    thread_YCrCb.start()
    thread_HSV.start()
    thread_YCrCb.join()
    YCrCb_mask = thread_YCrCb.get_result()
    thread_HSV.join()
    HSV_mask = thread_HSV.get_result()
    skinArea = get_final_mask(YCrCb_mask, HSV_mask)
    detectedSkin = cv2.bitwise_and(equ, equ, mask=skinArea)
    lst = []
    thread_green = VRDThread(get_green_frame, args=(detectedSkin,))
    thread_green.start()
    thread_green.join()
    lst.append(thread_green.get_result())
    lst.append(detectedSkin)
    lst.append(equ)
    return lst


def get_red_frame(detectedSkin):
    red_channel = detectedSkin[:, :, 2]
    red_frame = np.zeros(detectedSkin.shape)
    red_frame[:, :, 2] = red_channel
    return red_frame


def get_green_frame(detectedSkin):
    green_channel = detectedSkin[:, :, 1]
    green_frame = np.zeros(detectedSkin.shape)
    green_frame[:, :, 1] = green_channel
    return green_frame


def get_blue_frame(detectedSkin):
    blue_channel = detectedSkin[:, :, 0]
    blue_frame = np.zeros(detectedSkin.shape)
    blue_frame[:, :, 0] = blue_channel
    return blue_frame


def get_gray_frame(frame):
    frame1 = contrast_and_brightness(1.3, 0.4, frame)
    return frame1


def get_hsv_mask(frame):
    # converting from gbr to hsv color space
    img_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    img_HSV = cv2.GaussianBlur(img_HSV, (5, 5), 0)
    # skin color range for hsv color space
    HSV_mask = cv2.inRange(img_HSV, (0, 15, 0), (17, 170, 255))
    HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return HSV_mask


def get_YCrCb_mask(frame):
    # converting from gbr to YCbCr color space
    img_YCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    img_YCrCb = cv2.GaussianBlur(img_YCrCb, (5, 5), 0)
    # skin color range for hsv color space
    YCrCb_mask1 = cv2.inRange(img_YCrCb, (0, 135, 85), (255, 180, 135))
    YCrCb_mask2 = get_contourmask(img_YCrCb)
    YCrCb_mask = cv2.morphologyEx(cv2.bitwise_and(YCrCb_mask2, YCrCb_mask1), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return YCrCb_mask


def get_contourmask(YCrCb):
    (y, cr, cb) = cv2.split(YCrCb)  # Y, Cr, CB value
    cr1 = cv2.GaussianBlur(cr, (5, 5), 0)
    _, skin = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return skin


def get_final_mask(YCrCb_mask, HSV_mask):
    # merge skin detection (YCbCr and hsv)
    global_mask = cv2.bitwise_and(YCrCb_mask, HSV_mask)
    global_mask = cv2.medianBlur(global_mask, 3)
    global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, np.ones((4, 4), np.uint8))
    return global_mask


def get_capture_dimensions(capture):
    """Get the dimensions of a capture"""
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return width, height


def save_video(video, fps, save_filename='media/output.avi'):
    """Save a video to disk"""
    # fourcc = cv2.CAP_PROP_FOURCC('M', 'J', 'P', 'G')
    print(save_filename)
    video = float_to_uint8(video)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(save_filename, fourcc, fps, (video.shape[2], video.shape[1]), 1)
    for x in range(0, video.shape[0]):
        res = cv2.convertScaleAbs(video[x])
        writer.write(res)
