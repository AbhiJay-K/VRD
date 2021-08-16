import logging
import math
import statistics

import cv2
import imutils
import numpy
import numpy as np
import scipy

from scipy.signal import butter, lfilter

from VRD.pyramid import create_gaussian_video_pyramid, collapse_g_video_pyramid


def uint8_to_float(img):
    result = np.ndarray(shape=img.shape, dtype='float')
    result[:] = img * (1. / 255)
    return result


def float_to_uint8(img):
    result = np.ndarray(shape=img.shape, dtype='uint8')
    result[:] = img * 255
    return result


def float_to_int8(img):
    result = np.ndarray(shape=img.shape, dtype='uint8')
    result[:] = (img * 255) - 127
    return result


def temporal_bandpass_filter(data, fps, freq_min=0.833, freq_max=1, axis=0, amplification_factor=1):
    #print("Applying bandpass between " + str(freq_min) + " and " + str(freq_max) + " Hz")
    fft = scipy.fftpack.rfft(data, axis=axis)
    frequencies = scipy.fftpack.fftfreq(data.shape[0], d=1.0 / fps)
    bound_low = (np.abs(frequencies - freq_min)).argmin()
    bound_high = (np.abs(frequencies - freq_max)).argmin()
    fft[:bound_low] = 0
    fft[bound_high:-bound_high] = 0
    fft[-bound_low:] = 0

    result = np.ndarray(shape=data.shape, dtype='float')
    result[:] = scipy.fftpack.ifft(fft, axis=0)
    result *= amplification_factor
    return result


def temporal_bandpass_filter_hp(data, fps, freq_min, freq_max, axis=0, amplification_factor=1):
    #print("Applying bandpass between " + str(freq_min) + " and " + str(freq_max) + " Hz")
    fft = scipy.fftpack.rfft(data, axis=axis)
    frequencies = scipy.fftpack.fftfreq(data.shape[0], d=1.0 / fps)
    bound_low = (np.abs(frequencies - freq_min)).argmin()
    bound_high = (np.abs(frequencies - freq_max)).argmin()
    fft[:bound_low] = 0
    fft[bound_high:-bound_high] = 0
    fft[-bound_low:] = 0

    result = np.ndarray(shape=data.shape, dtype='float')
    result[:] = scipy.fftpack.ifft(fft, axis=0)
    result *= amplification_factor
    return result


def bandpass_filter(data, fps, freq_min, freq_max, axis=0):
    #print("Applying bandpass between " + str(freq_min) + " and " + str(freq_max) + " Hz")
    fft = scipy.fftpack.rfft(data, axis=axis)
    frequencies = scipy.fftpack.fftfreq(data.shape[0], d=1.0 / fps)
    bound_low = (np.abs(frequencies - freq_min)).argmin()
    bound_high = (np.abs(frequencies - freq_max)).argmin()
    fft[:bound_low] = 0
    fft[bound_high:-bound_high] = 0
    fft[-bound_low:] = 0

    result = np.ndarray(shape=data.shape, dtype='float')
    result[:] = scipy.fftpack.ifft(fft, axis=0)
    return result


def colour_hist_eq(frame):
    img_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    # equalize the histogram of the Y channel
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    # convert the YUV image back to RGB format
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)


def high_pass(data, fps, freq_max, axis=0):
    print("Applying hipass at ", str(freq_max), " Hz")
    fft = scipy.fftpack.rfft(data, axis=axis)
    frequencies = scipy.fftpack.fftfreq(data.shape[0], d=1.0 / fps)
    bound_high = (np.abs(frequencies - freq_max)).argmin()
    fft[bound_high:-bound_high] = 0
    result = np.ndarray(shape=data.shape, dtype='float')
    result[:] = scipy.fftpack.ifft(fft, axis=0)
    return result


def low_pass(data, fps, freq_min, axis=0):
    print("Applying hipass at ", str(freq_min), " Hz")
    fft = scipy.fftpack.rfft(data, axis=axis)
    frequencies = scipy.fftpack.fftfreq(data.shape[0], d=1.0 / fps)
    bound_low = (np.abs(frequencies - freq_min)).argmin()
    fft[:bound_low] = 0
    fft[-bound_low:] = 0
    result = np.ndarray(shape=data.shape, dtype='float')
    result[:] = scipy.fftpack.ifft(fft, axis=0)
    return result


def contrast_and_brightness(alpha, beta, img):
    blank = np.zeros(img.shape, img.dtype)
    # dst = alpha * img + (1-alpha) * blank + beta
    dst = cv2.addWeighted(img, alpha, blank, 1-alpha, beta)
    return dst


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5,amplification=1):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def Sort_Tuple(tup):
    # getting length of list of tuples
    lst = len(tup)
    for i in range(0, lst):

        for j in range(0, lst - i - 1):
            if (tup[j][1] > tup[j + 1][1]):
                temp = tup[j]
                tup[j] = tup[j + 1]
                tup[j + 1] = temp
    return tup


def eulerian_magnification_colour(vid_data, fps,freq_min_narrow, freq_max_narrow, amplification, pyramid_levels=3,skip_levels_at_top=1):
    # vid_data_c = butter_bandpass_filter(vid_data, 0.6, 3, 50, order=5,amplification=1520)
    vid_pyramid = create_gaussian_video_pyramid(vid_data, pyramid_levels=pyramid_levels)
    vid_pyramid_new = []
    for i, vid in enumerate(vid_pyramid):
        if i < 1 or i >= len(vid_pyramid) - 1:
            # ignore the top and bottom of the pyramid. One end has too much noise and the other end is the
            # gaussian representation
            continue
        bandpassed = temporal_bandpass_filter_hp(vid, fps, freq_min_narrow, freq_max_narrow,amplification_factor=amplification)
        vid_pyramid_new.append(bandpassed)
    vid_data_c = collapse_g_video_pyramid(vid_pyramid_new)
    return vid_data_c


def eulerian_magnification_butter(vid_data, fps,freq_min_narrow, freq_max_narrow, amplification, pyramid_levels=3,skip_levels_at_top=1):
    # vid_data_c = butter_bandpass_filter(vid_data, 0.6, 3, 50, order=5,amplification=1520)
    vid_pyramid = create_gaussian_video_pyramid(vid_data, pyramid_levels=pyramid_levels)
    vid_pyramid_new = []
    for i, vid in enumerate(vid_pyramid):
        if i < 1 or i >= len(vid_pyramid) - 1:
            # ignore the top and bottom of the pyramid. One end has too much noise and the other end is the
            # gaussian representation
            continue
        bandpassed = temporal_bandpass_filter_hp(vid, fps, freq_min_narrow, freq_max_narrow,amplification_factor=amplification)
        vid_pyramid_new.append(bandpassed)
    vid_data_c = collapse_g_video_pyramid(vid_pyramid_new)
    return butter_bandpass_filter(vid_data_c, freq_min_narrow, freq_max_narrow, 1000, order=5)

def find_centers(vid_data):
    video = float_to_uint8(vid_data)
    i = 0
    for frame in video:
        frame = center_of_contours(frame)
        video[i] = uint8_to_float(frame)
        i = i + 1
    return video


def center_of_contours(frame):
    # load the image, convert it to grayscale, blur it slightly,
    # and threshold it
    image = frame
    h, w, c = image.shape
    blank_image = np.zeros((h, w, 3), np.uint8)
    equ = blank_image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    for c in cnts:
        # compute the center of the contour
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
        cY = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0
        # draw the contour and center of the shape on the image
        cv2.drawContours(blank_image, [c], -1, (0, 255, 0), 2)
        cv2.circle(blank_image, (cX, cY), 5, (255, 255, 255), -1)
        equ = colour_hist_eq(blank_image)
        #cv2.putText(blank_image, "center", (cX - 20, cY - 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return equ

def brightness(frame):
    r = frame[:, :, 2].sum()
    g = frame[:, :, 1].sum()
    b = frame[:, :, 0].sum()
    return np.float64(math.sqrt(0.241 * (r ** 2) + 0.691 * (g ** 2) + 0.068 * (b ** 2)))


def get_mad(fq_list, median):
    median_diff_list = []
    for item in fq_list:
        logging.debug("Median diff :", abs(item - median))
        median_diff_list.append(abs(item - median))
    if len(median_diff_list) > 0:
        logging.debug("MAD median :", statistics.median(median_diff_list))
        return statistics.median(median_diff_list)
    else:
        return 0.0


def check_dark(gray_img, max_darkness):
    # Convert the picture to a single-channel grayscale image
    # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Get the number of rows and columns of the grayscale image matrix
    r, c = gray_img.shape[:2]
    piexs_sum = r * c  # The number of pixels in the entire radian map is r*c

    # Get dark pixels(Means0~19The gray value is dark)  The threshold can be modified here
    dark_points = (gray_img < 20)
    target_array = gray_img[dark_points]
    dark_sum = target_array.size
    # Determine the percentage of gray value that is dark
    dark_prop = dark_sum / (piexs_sum)
    if dark_prop >= max_darkness:
        dark_prop = 1.0
    return dark_prop


def get_frame_dimensions(frame):
    """Get the dimensions of a single frame"""
    height, width = frame.shape[:2]
    return width, height






