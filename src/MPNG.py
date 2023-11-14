'''Motion PNG. Provides entropy Eecoding of video using PNG (Portable Network Graphics). '''

import io
from skimage import io as skimage_io # pip install scikit-image
import main
import logging
import numpy as np
import cv2 as cv
import parser
import entropy_video_coding as EVC
import av

# Default IOs
ENCODE_INPUT = "http://www.hpca.ual.es/~vruiz/videos/mobile_352x288x30x420x300.avi"
ENCODE_OUTPUT = "/tmp/encoded"
DECODE_INPUT = ENCODE_OUTPUT
DECODE_OUTPUT = "/tmp/decoded.avi"

# Encoder parser
parser.parser_encode.add_argument("-i", "--input", type=parser.int_or_str, help=f"Input video (default: {ENCODE_INPUT})", default=ENCODE_INPUT)
parser.parser_encode.add_argument("-o", "--output", type=parser.int_or_str, help=f"Output video (default: {ENCODE_OUTPUT})", default=f"{ENCODE_OUTPUT}")

# Decoder parser
parser.parser_decode.add_argument("-i", "--input", type=parser.int_or_str, help=f"Input video (default: {DECODE_INPUT})", default=f"{DECODE_INPUT}")
parser.parser_decode.add_argument("-o", "--output", type=parser.int_or_str, help=f"Output video (default: {DECODE_OUTPUT})", default=f"{DECODE_OUTPUT}")    

parser.parser.parse_known_args()

COMPRESSION_LEVEL = 9

class CoDec(EVC.CoDec):

    def __init__(self, args):
        super().__init__(args)

    def compress(self, vid):
        '''Input a H.264 AVI-file and output a sequence of PNG images.'''
        container = av.open(fn)
        for frame in container.decode(video=0):
            img = frame.to_image()
            img_fn = os.path.join(ENCODE_OUTPUT, "_%04d.png" % frame.index)
            print(img_fn)
            img.save(img_fn)
            # cv2.imwrite(img_fn, img)
        compressed_vid = vid.copy()
        compressed_vid.prefix = ENCODE_OUTPUT
        return compressed_vid

    def decompress(self, compressed_vid):
        '''Input a sequence of PNG images and output a H.264 AVI-file.'''
        imgs = [i for i in os.listdir(compressed_vid.prefix)]
        
        container = av.open(DECODE_OUTPUT, 'w', format='avi')
        video_stream = container.add_stream('libx264', rate=framerate)
        img_0 = Image.open(os.path.join(ENCODE_INPUT, "_0000.png")).convert('RGB')
        width, height = img_0.size
        video_stream.width = width
        video_stream.height = height
        for i in imgs:
            frame = av.VideoFrame.from_image(i)
            packet = video_stream.encode(frame)
            container.mux(packet)
        container.close()
        vid = compressed_vid
        vid.prefix = DECODE_OUTPUT
        return vid

if __name__ == "__main__":
    main.main(parser.parser, logging, CoDec)
