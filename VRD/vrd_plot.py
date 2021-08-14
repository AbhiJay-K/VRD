
import numpy as np
import scipy
from matplotlib import pyplot


def show_frequencies_plot(vid_data, fps, bounds=None,filename="output",folder_name="Result"):
    """Graph the average value of the video as well as the frequency strength"""
    averages = np.array([])
    if bounds:
        for x in range(1, vid_data.shape[0] - 1):
            averages = np.insert(averages,vid_data[x, bounds[2]:bounds[3], bounds[0]:bounds[1], :].mean())
    else:
        for x in range(1, vid_data.shape[0] - 1):
            print(vid_data[x, :, :, :].sum(),type(vid_data[x, :, :, :].sum()))
            #averages.append(brightness(vid_data[x, :, :, :]))
            averages = np.append(averages,vid_data[x, :, :, :].mean())
    averages = averages - min(averages)
    charts_x = 1
    charts_y = 2
    #pyplot.rc('text',usetex=True)
    pyplot.figure(figsize=(10,6))
    pyplot.subplots_adjust(hspace=.6)
    pyplot.subplot(charts_y, charts_x, 1)
    #pyplot.title("FFT PLOT "+filename)
    pyplot.xlabel("Time (Sec)", weight='bold',fontsize=14)
    pyplot.ylabel("Intensity (Pixel value)", weight='bold',fontsize=14)
    time = np.arange(0, len(averages)/fps, 1 / fps)
    print(len(averages) / fps)
    print(len(time))
    print(len(averages))
    if len(time) > len(averages):
        time = time[0:len(averages)]
    print(len(averages)/fps)
    print(len(time))
    print(len(averages))
    pyplot.plot(time,averages,linewidth=1,color='k', ls='solid')
    freqs = scipy.fftpack.fftfreq(len(averages), d=1/fps)
    fft = abs(scipy.fftpack.fft(averages))
    idx = np.argsort(freqs)
    pyplot.subplot(charts_y, charts_x, 2)
    pyplot.title("FFT")
    pyplot.xlabel("Freq (Hz)", weight='bold',fontsize=14)
    pyplot.ylabel("Intensity (Pixel value)", weight='bold', fontsize=14)
    freqs = freqs[idx]

    fft = fft[idx]

    freqs = freqs[len(freqs) // 2 + 1:]
    fft = fft[len(fft) // 2 + 1:]
    pyplot.plot(freqs, fft,linewidth=1,color='k', ls='solid')
    pyplot.savefig(folder_name +'/plot_'+filename+'.pdf')
    fqftlist = []
    top_fqftlist = []
    top_fqftlist.sort(reverse=True)
    for fq, ft in np.array(list(zip(freqs, fft))):
        fqftlist.append((fq, ft))
    nd_fqlist = np.array(fqftlist)
    if len(nd_fqlist) != 0:
        nd_fqlist = nd_fqlist[nd_fqlist[:, 1].argsort()]
    return nd_fqlist


def plot_fq_fft(low,high,fft_median,fq_median,mad_fft,filename,small_fq_list,small_fft_list,folder_name="Result"):
    pyplot.scatter(small_fq_list, small_fft_list,color='tab:cyan')
    pyplot.axvline(fq_median, color='tab:blue', linestyle='dashed', linewidth=1,label='Frequency median')
    pyplot.axvline(low, color='cadetblue', linestyle='dashed', linewidth=1,label='Frequency low')
    pyplot.axvline(high, color='steelblue', linestyle='dashed', linewidth=1,label='Frequency high')
    pyplot.axhline(fft_median, color='khaki', linestyle='dashed', linewidth=1,label='Mean')
    pyplot.axhline((fft_median + mad_fft), color='darkgoldenrod', linestyle='dashed', linewidth=1,label='Mean + std dev')
    pyplot.axhline((fft_median + 3 * mad_fft), color='slategrey', linestyle='dashed', linewidth=1,label='Mean + 3 std dev')
    pyplot.legend(prop={"size":10})
    pyplot.savefig(folder_name+'/plot_' + filename + '.pdf')
    pyplot.close("all")