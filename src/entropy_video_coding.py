'''Shared code among the video entropy codecs. "Uncompressed" IO uses
lossless H.264 encapsulated in AVI.'''

import os
import io
from skimage import io as skimage_io # pip install scikit-image
from PIL import Image # pip install 
import numpy as np
import logging
import subprocess
import cv2 as cv
import main
import urllib

from information_theory import distortion # pip install "information_theory @ git+https://github.com/vicente-gonzalez-ruiz/information_theory"

class CoDec:

    def __init__(self, args):
        self.args = args
        logging.debug(f"args = {self.args}")
        if args.subparser_name == "encode":
            self.encoding = True
        else:
            self.encoding = False
        logging.debug(f"self.encoding = {self.encoding}")
        self.input_bytes = 0
        self.output_bytes = 0

    def __del__(self):
        logging.info(f"Total {self.input_bytes} bytes read")
        logging.info(f"Total {self.output_bytes} bytes written")
        if self.encoding:
            N_frames = self.vid_shape[0]
            BPP = (self.output_bytes*8)/(self.vid_shape[1]*self.vid_shape[2])
            logging.info(f"N_frames = {N_frames}")
            logging.info(f"rate = {BPP} bits/pixel")
            with open(f"{self.args.output}_BPP.txt", 'w') as f:
                f.write(f"{N_frames}")
                f.write(f"{BPP}")
        else:
            vid = self.encode_read_fn("file:///tmp/original.avi")
            y = self.encode_read_fn(self.args.output)
            total_RMSE = 0
            for i in vid:
                total_RMSE += distortion.RMSE(i, y)
            RMSE = total_RMSE / self.vid_shape[0]
            logging.info(f"RMSE = {RMSE}")
            with open(f"{self.args.input}_BPP.txt", 'r') as f:
                N_frames = float(f.read())
                BPP = float(f.read())
            J = BPP + RMSE
            logging.info(f"J = R + D = {J}")

    def encode(self):
        vid = self.encode_read()
        compressed_vid = self.compress(vid)
        self.encode_write(compressed_vid)
        #logging.info(f"BPP = {BPP}")
        #return BPP

    def decode(self):
        compressed_vid = self.decode_read()
        vid = self.decompress(compressed_vid)
        #compressed_img_diskimage = io.BytesIO(compressed_img)
        #img = np.load(compressed_img_diskimage)['a']
        #decompressed_data = zlib.decompress(compressed_img)
        #img = io.BytesIO(decompressed_data))
        self.decode_write(vid)
        #logging.debug(f"output_bytes={self.output_bytes}, img.shape={img.shape}")
        #self.BPP = (self.output_bytes*8)/(img.shape[0]*img.shape[1])
        #return rate, 0
        #logging.info("RMSE = 0")

    def encode_read(self):
        '''Read the video specified in the class attribute <args.input>.'''
        vid = self.encode_read_fn(self.args.input)
        self.decode_write_fn(vid, "/tmp/original.avi")
        self.output_bytes = 0
        self.vid_shape = vid.shape
        return vid

    def decode_read(self):
        compressed_vid = self.decode_read_fn(self.args.input)
        return compressed_vid

    def encode_write(self, compressed_vid):
        '''Save to disk the video specified in the class attribute <
        args.output>.'''
        self.encode_write_fn(compressed_vid, self.args.output)

    def decode_write(self, vid):
        return self.decode_write_fn(vid, self.args.output)
        
    def encode_read_fn(self, fn):
        '''Read the video <fn>.'''
        #img = skimage_io.imread(fn) # https://scikit-image.org/docs/stable/api/skimage.io.html#skimage.io.imread
        #img = Image.open(fn) # https://pillow.readthedocs.io/en/stable/handbook/tutorial.html#using-the-image-class
        try:
            input_size = os.path.getsize(fn)
            self.input_bytes += input_size 
            img = cv.imread(fn, cv.IMREAD_UNCHANGED)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        except:
            req = urllib.request.Request(fn, method='HEAD')
            f = urllib.request.urlopen(req)
            input_size = int(f.headers['Content-Length'])
            self.input_bytes += input_size
            img = skimage_io.imread(fn) # https://scikit-image.org/docs/stable/api/skimage.io.html#skimage.io.imread
        logging.info(f"Read {input_size} bytes from {fn} with shape {img.shape} and type={img.dtype}")
        return img

    def decode_read_fn(self, fn_without_extention):
        fn = fn_without_extention + self.file_extension
        input_size = os.path.getsize(fn)
        self.input_bytes += input_size
        logging.info(f"Read {os.path.getsize(fn)} bytes from {fn}")
        data = open(fn, "rb").read()
        return data

    def encode_write_fn(self, data, fn_without_extention):
        data.seek(0)
        fn = fn_without_extention + self.file_extension
        with open(fn, "wb") as output_file:
            output_file.write(data.read())
        self.output_bytes += os.path.getsize(fn)
        logging.info(f"Written {os.path.getsize(fn)} bytes in {fn}")

    def decode_write_fn(self, img, fn):
        skimage_io.imsave(fn, img)
        self.output_bytes += os.path.getsize(fn)
        logging.info(f"Written {os.path.getsize(fn)} bytes in {fn} with shape {img.shape} and type {img.dtype}")

