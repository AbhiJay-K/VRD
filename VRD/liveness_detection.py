import statistics
import traceback

import cv2
import numpy as np

from VRD.vrd_plot import show_frequencies_plot, plot_fq_fft
from VRD.preprocessing import save_video, obtain_roi
from VRD.util import float_to_uint8, eulerian_magnification_butter, get_mad
import logging


def write_pyramid(pyramid):
    i = 0
    while True:
        try:
            for level, vid in enumerate(pyramid):
                cv2.imshow('Level %i' % level, vid[i])
            i += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except IndexError:
            break


def liveness_detection(row, freq_min, freq_max, amplification, pyramid_levels, result_folder_name):
    vid_list = obtain_roi(row[0], freq_min, freq_max, amplification, pyramid_levels)
    fps = vid_list[0]
    green_vid = vid_list[3]
    logging.debug("done loading file")
    save_video(green_vid, fps, result_folder_name + '/vid_' + str(row[0]).split(".")[0] + "_colour_amp.avi")
    full_list_green = show_frequencies_plot(green_vid, fps, None, str(row[0]).split(".")[0] + "_green",
                                            result_folder_name)
    count = 1.0
    max_list = get_max_frequency(full_list_green, str(row[0]).split(".")[0] + "_filtered", count)
    prev_list = [max_list]
    sum_of_fq_count = 0.0
    logging.debug("Continue ", count, max_list)
    while max_list[3] > 1:
        logging.debug(max_list)
        if max_list[0] == 0.0 and max_list[0] == 0.0:
            return check_frames(green_vid, row, count, prev_list)
        green_vid = eulerian_magnification_butter(green_vid, fps, max_list[0], max_list[1], max_list[2] * amplification,
                                                  pyramid_levels)
        full_list_green = show_frequencies_plot(green_vid, fps, None,
                                                str(row[0]).split(".")[0] + "_green_" + str(count),
                                                result_folder_name)
        max_list = get_max_frequency(full_list_green, str(row[0]).split(".")[0] + "_filtered_" + str(count), count)
        count = count + 1.0
        if prev_list[len(prev_list) - 1][3] == max_list[3] or max_list[0] > 3.3:
            break
        prev_list.append(max_list)
        logging.debug("Continue ", count, max_list)
        sum_of_fq_count = sum_of_fq_count + max_list[3]
    return check_frames(green_vid, row, count, prev_list)


def check_same_range(ar1, ar2):
    if (abs(ar1[0] - ar2[0])) == 0 and (abs(ar1[1] - ar2[1])) == 0:
        return True
    else:
        return False


def check_frames(vid_green, row, count, max_list):
    video_green = float_to_uint8(vid_green)
    result_green = check_dark_vid(video_green)
    low = max_list[len(max_list) - 2][0]
    high = max_list[len(max_list) - 2][1]
    mean_freq = (low + high) / 2.0
    logging.debug("mean frequency : ", mean_freq)
    if result_green is not None:
        frequency = result_green[0][0]
        intesity_score = result_green[0][1]
        if count > 1.0:
            if 0.8 <= frequency <= 3.3 and \
                    0.8 <= mean_freq <= 3.3:
                if intesity_score == 1:
                    prediction = 1
                else:
                    prediction = 0
            else:
                prediction = 0
        else:
            prediction = 0
    else:
        frequency = 0.0
        prediction = 0
    if len(max_list) > 1 and not len(max_list) == 0:
        logging.info([frequency, count, max_list[len(max_list) - 2][0], max_list[len(max_list) - 2][1],
               max_list[len(max_list) - 2][3], prediction, row[1], row[0]])
        return [frequency, count, max_list[len(max_list) - 2][0], max_list[len(max_list) - 2][1],
                max_list[len(max_list) - 2][3], prediction, row[1], row[0]]
    else:
        logging.info([frequency, count, max_list[0][0], max_list[0][1], max_list[0][3], prediction, row[1], row[0]])
        return [frequency, count, max_list[0][0], max_list[0][1], max_list[0][3], prediction, row[1], row[0]]


def check_dark_vid(full_list_greem_motion):
    frequency = get_freq(full_list_greem_motion)
    logging.debug([frequency])
    return [frequency]


def get_max_frequency(full_list_c, filename, count, result_folder_name):
    logging.debug("full list for ", count, "run >>>", full_list_c)
    amp = 1
    try:
        if full_list_c.__len__() > 0:
            full_fqlist, full_fftlist = zip(*full_list_c)
            if len(full_fqlist) > 2 and len(full_fftlist) > 2:
                if 1 > min(full_fftlist) > 0:
                    amp = 1.0 / min(full_fftlist)
                else:
                    amp = min(full_fftlist)
                logging.debug("min fft:", min(full_fftlist))
                logging.debug("amp:", amp)
                fft_mean = statistics.mean(full_fftlist)
                fft_std_dev = statistics.stdev(full_fftlist)
                logging.debug(" fft mean and std dev >>>> ", fft_mean, fft_std_dev)
                small_fq_list = []
                small_fft_list = []
                lesser_fq_list = []
                lesser_fft_list = []
                for fq, ft in full_list_c:
                    if (fft_mean + (2 * fft_std_dev)) <= ft < (fft_mean + (3 * fft_std_dev)):
                        logging.debug("fft above ", (fft_mean + (3 * fft_std_dev)), " :>", ft)
                        small_fq_list.append(fq)
                        small_fft_list.append(ft)
                    if ft >= (fft_mean + fft_std_dev):
                        lesser_fft_list.append(ft)
                        lesser_fq_list.append(fq)
                if len(small_fft_list) == 0:
                    small_fq_list = lesser_fq_list.copy()
                    small_fft_list = lesser_fft_list.copy()
                    logging.debug(len(small_fq_list))
                if len(small_fq_list) > 2:
                    logging.debug("Ploting done")
                    fq_median = statistics.median(small_fq_list)
                    fq_mean = statistics.mean(small_fq_list)
                    fq_stddev = statistics.stdev(small_fq_list)
                    logging.debug("median frquency:", fq_median)
                    mad = get_mad(small_fq_list, fq_median)
                    logging.debug(fq_median, mad)
                    if count == 1:
                        logging.debug("mean : ", fq_mean, "std dev:", statistics.stdev(small_fq_list))
                        if (fq_mean + fq_stddev) > 3.3:
                            low = (fq_median - mad) if (fq_median - mad) > 0 else min(small_fq_list)
                            high = (fq_median + mad)
                            plot_fq_fft(low, high, fft_mean, fq_median, fft_std_dev, filename,
                                        small_fq_list, small_fft_list, result_folder_name)
                            return [0, 0, amp, len(small_fq_list)]
                    logging.debug("Length of small list : ", len(small_fq_list))
                else:
                    if len(small_fq_list) > 1:
                        unsorted_fq_list = small_fq_list.copy()
                        small_fq_list.sort()
                        logging.debug("Small list : ", small_fq_list)
                        low = small_fq_list[0]
                        high = small_fq_list[1]
                        fq_median = statistics.median(small_fq_list)
                        plot_fq_fft(low, high, fft_mean, fq_median, fft_std_dev, filename,
                                    unsorted_fq_list, small_fft_list, result_folder_name)
                        return [small_fq_list[0] - 0.05, small_fq_list[1] + 0.05, amp, len(small_fq_list)]
                    else:
                        low = small_fq_list[0]
                        high = small_fq_list[0]
                        fq_median = small_fq_list[0]
                        plot_fq_fft(low, high, fft_mean, fq_median, fft_std_dev, filename,
                                    small_fq_list, small_fft_list, result_folder_name)
                        return [small_fq_list[0], small_fq_list[0], amp, len(small_fq_list)]
                low = (fq_median - mad) if (fq_median - mad) > 0 else min(small_fq_list)
                high = (fq_median + mad)
                plot_fq_fft(low, high, fft_mean, fq_median, fft_std_dev, filename, small_fq_list,
                            small_fft_list, result_folder_name)
                return [low, high, amp, len(small_fq_list)]
            else:
                low = full_fqlist[len(full_fqlist) - 1] - 0.2
                high = full_fqlist[len(full_fqlist) - 1] + 0.2
                fft_mean = statistics.mean(full_fftlist)
                fq_median = statistics.median(full_fqlist)
                plot_fq_fft(low, high, fft_mean, fq_median, fft_mean + 0.01, filename, full_fqlist,
                            full_fftlist, result_folder_name)
                return [full_fqlist[len(full_fqlist) - 1] - 0.2, full_fqlist[len(full_fqlist) - 1] + 0.2, 1,
                        len(full_fqlist)]
        else:
            return [0, 0, amp, 0]
    except Exception as er:
        traceback.print_exc()
        return [0, 0, amp, 0]


def get_freq(full_list_c):
    colour_max_fq = 0.0
    prom_fft = 0.0
    logging.debug(full_list_c)
    if full_list_c.__len__() > 0:
        full_fqlist, full_fftlist = zip(*full_list_c)
        fq_median = statistics.median(full_fqlist[len(full_fqlist) - 10:len(full_fqlist) - 1])
        logging.debug("Median of frequency : ", fq_median)
        fq_mad = get_mad(full_fqlist[len(full_fqlist) - 10:len(full_fqlist) - 1], fq_median)
        logging.debug("Median absolute deviation of frequency : ", fq_mad)
        if len(full_fftlist) > 0:
            if 0.8 < full_fqlist[len(full_fqlist) - 1] < 3.3:
                logging.debug("Highest Frequency : ", full_fqlist[len(full_fqlist) - 1])
                colour_max_fq = full_fqlist[len(full_fqlist) - 1]
                prom_fft = 1
    return [colour_max_fq, prom_fft]


def convert_nd_list(nd_arr):
    a = np.array(nd_arr)
    at = a.T
    at = np.array(at)
    return list(zip(at[:0], at[:1]))
