import argparse
import logging

from VRD.liveness_detection import liveness_detection

long_options = ["help",
                "output=", "verbose"]

file_handler = logging.FileHandler('vrd.log')
formatter    = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
file_handler.setFormatter(formatter)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-fmin", dest='frq_min', help="minimum frequency for wide temporal filtering",
                        type=float)
    parser.add_argument("-fmax", dest='frq_max',help="maximum frequency for wide temporal filtering",
                        type=float)
    parser.add_argument("-a", dest='amp_factor',help="amplification factor for temporal filtering",
                        type=int)
    parser.add_argument("-p", dest='pyramid_levels',help="number of pyramid levels for gaussian pyramid",
                        type=int)
    parser.add_argument("-i", dest='input_file', help="input video file name",
                        type=str)
    parser.add_argument("-o",  dest='output_folder',help="folder to output the result",
                        type=str)
    parser.add_argument("-v","--verbosity", help="increase output verbosity",action="store_true")
    parser.add_argument('--version', action='version', version='%(prog)s 1.0')
    args = parser.parse_args()
    if args.verbosity:
        logging.basicConfig(level=logging.DEBUG,filename="vrd.log")
    else:
        logging.basicConfig(level=logging.error(),filename="vrd.log")
    row = [args.input_file,1]
    liveness_detection(row, args.frq_min, args.frq_max, args.amp_factor, args.pyramid_levels, args.output_folder)