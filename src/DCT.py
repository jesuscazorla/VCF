'''Exploiting spatial (perceptual) redundancy with the 2D Discrete Cosine Transform.'''

import io
from skimage import io as skimage_io # pip install scikit-image
import numpy as np
import pywt
import os
import logging
import main
import parser
import importlib

#from DWT import color_dyadic_DWT as DWT
from DCT2D.block_DCT import analyze as space_analyze # pip install "DCT2D @ git+https://github.com/vicente-gonzalez-ruiz/DCT2D"
from DCT2D.block_DCT import synthesize as space_synthesize
from DCT2D.block_DCT import get_subbands
from DCT2D.block_DCT import get_blocks

from color_transforms.YCoCg import from_RGB # pip install "color_transforms @ git+https://github.com/vicente-gonzalez-ruiz/color_transforms"
from color_transforms.YCoCg import to_RGB

default_block_size = 8
default_CT = "YCoCg"

parser.parser_encode.add_argument("-b", "--block_size", type=parser.int_or_str, help=f"Block size (default: {default_block_size})", default=default_block_size)
parser.parser_encode.add_argument("-t", "--color_transform", type=parser.int_or_str, help=f"Color transform (default: \"{default_CT}\")", default=default_CT)
parser.parser_decode.add_argument("-b", "--block_size", type=parser.int_or_str, help=f"Block size (default: {default_block_size})", default=default_block_size)
parser.parser_decode.add_argument("-t", "--color_transform", type=parser.int_or_str, help=f"Color transform (default: \"{default_CT}\")", default=default_CT)

args = parser.parser.parse_known_args()[0]
CT = importlib.import_module(args.color_transform)

class CoDec(CT.CoDec):

    def __init__(self, args):
        super().__init__(args)
        self.block_size = args.block_size
        logging.info(f"block_size = {self.block_size}")

    def encode(self):
        img = self.encode_read().astype(np.int16)
        CT_img = from_RGB(img)
        DCT_img = space_analyze(CT_img, self.block_size, self.block_size)
        decom_img = get_subbands(DCT_img, self.block_size, self.block_size)
        decom_k = self.quantize_decom(decom_img)
        decom_k = self.compress(decom_k.astype(np.uint8))
        self.encode_write(decom_k)
        rate = (self.output_bytes*8)/(img.shape[0]*img.shape[1])
        return rate

    def decode(self):
        decom_k = self.decode_read()
        decom_k = self.decompress(decom_k).astype(np.int16)
        decom_y = self.dequantize_decom(decom_k)
        DCT_y = get_blocks(decom_y, self.block_size, self.block_size)
        CT_y = space_synthesize(DCT_y, self.block_size, self.block_size)
        y = to_RGB(CT_y)
        y = np.clip(y, 0, 255).astype(np.uint8)
        self.decode_write(y)
        rate = (self.input_bytes*8)/(y.shape[0]*y.shape[1])
        return rate

    def quantize_decom(self, decom):
        subbands_in_y = self.block_size
        subbands_in_x = self.block_size
        subband_y_size = int(decom.shape[0]/self.block_size)
        subband_x_size = int(decom.shape[1]/self.block_size)
        #decom_k = np.empty_like(decom, dtype=np.int16)
        decom_k = decom
        for by in range(subbands_in_y):
            for bx in range(subbands_in_x):
                subband = decom[by*subband_y_size:(by+1)*subband_y_size,
                                bx*subband_x_size:(bx+1)*subband_x_size,
                                :]
                subband_k = self.quantize(subband)
                #subband_k += 128
                decom_k[by*subband_y_size:(by+1)*subband_y_size,
                        bx*subband_x_size:(bx+1)*subband_x_size,
                        :] = subband_k
        #decom_k[0:subband_y_size, 0:subband_x_size, 1] += 128
        #decom_k[0:subband_y_size, 0:subband_x_size, 2] += 128
        return decom_k

    def dequantize_decom(self, decom_k):
        subbands_in_y = self.block_size
        subbands_in_x = self.block_size
        subband_y_size = int(decom_k.shape[0]/self.block_size)
        subband_x_size = int(decom_k.shape[1]/self.block_size)
        #decom_y = np.empty_like(decom_k, dtype=np.int16)
        decom_y = decom_k
        #decom_y[0:subband_y_size, 0:subband_x_size, 1] -= 128
        #decom_y[0:subband_y_size, 0:subband_x_size, 2] -= 128
        for by in range(subbands_in_y):
            for bx in range(subbands_in_x):
                subband_k = decom_k[by*subband_y_size:(by+1)*subband_y_size,
                                    bx*subband_x_size:(bx+1)*subband_x_size,
                                    :]
                #subband_k -= 128
                subband_y = self.dequantize(subband_k)
                decom_k[by*subband_y_size:(by+1)*subband_y_size,
                        bx*subband_x_size:(bx+1)*subband_x_size,
                        :] = subband_y
        return decom_y

if __name__ == "__main__":
    main.main(parser.parser, logging, CoDec)
