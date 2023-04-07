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

from information_theory import distortion # pip install "information_theory @ git+https://github.com/vicente-gonzalez-ruiz/information_theory"

default_block_size = 8
default_CT = "YCoCg"
perceptual_quantization = False

parser.parser_encode.add_argument("-b", "--block_size", type=parser.int_or_str, help=f"Block size (default: {default_block_size})", default=default_block_size)
parser.parser_encode.add_argument("-t", "--color_transform", type=parser.int_or_str, help=f"Color transform (default: \"{default_CT}\")", default=default_CT)
parser.parser_encode.add_argument("-p", "--perceptual_quantization", action='store_true', help=f"Use perceptual quantization (default: \"{perceptual_quantization}\")", default=perceptual_quantization)
parser.parser_encode.add_argument("-L", "--Lambda", type=parser.int_or_str, help="Relative weight between the rate and the distortion. If provided (float), the block size is RD-optimized between {2**i; i=1,2,3,4,5,6,7}. For example, if Lambda=1.0, then the rate and the distortion have the same weight.")
parser.parser_decode.add_argument("-b", "--block_size", type=parser.int_or_str, help=f"Block size (default: {default_block_size})", default=default_block_size)
parser.parser_decode.add_argument("-t", "--color_transform", type=parser.int_or_str, help=f"Color transform (default: \"{default_CT}\")", default=default_CT)
parser.parser_decode.add_argument("-p", "--perceptual_quantization", action='store_true', help=f"Use perceptual dequantization (default: \"{perceptual_quantization}\")", default=perceptual_quantization)

args = parser.parser.parse_known_args()[0]
CT = importlib.import_module(args.color_transform)

class CoDec(CT.CoDec):

    def __init__(self, args):
        super().__init__(args)
        self.block_size = args.block_size
        logging.info(f"block_size = {self.block_size}")
        if args.perceptual_quantization:
            self.quantize_decom = self.perceptual_quantize_decom
            logging.info("using perceptual quantization")
            # Luma
            self.Y_QSSs = np.array([[16, 11, 10, 16, 24, 40, 51, 61], 
                                    [12, 12, 14, 19, 26, 58, 60, 55],
                                    [14, 13, 16, 24, 40, 57, 69, 56],
                                    [14, 17, 22, 29, 51, 87, 80, 62],
                                    [18, 22, 37, 56, 68, 109, 103, 77],
                                    [24, 35, 55, 64, 81, 104, 113, 92],
                                    [49, 64, 78, 87, 103, 121, 120, 101],
                                    [72, 92, 95, 98, 112, 100, 103, 99]])
            # Chroma
            self.C_QSSs = np.array([[17, 18, 24, 47, 99, 99, 99, 99], 
                                    [18, 21, 26, 66, 99, 99, 99, 99],
                                    [24, 26, 56, 99, 99, 99, 99, 99],
                                    [47, 66, 99, 99, 99, 99, 99, 99],
                                    [99, 99, 99, 99, 99, 99, 99, 99],
                                    [99, 99, 99, 99, 99, 99, 99, 99],
                                    [99, 99, 99, 99, 99, 99, 99, 99],
                                    [99, 99, 99, 99, 99, 99, 99, 99]])
        if self.encoding:
            if args.Lambda is not None:
                self.Lambda = float(args.Lambda)
                logging.info("optimizing the block size")
                self.optimize_block_size()
                logging.info(f"optimal block_size={self.block_size}")

    def encode(self):
        img = self.encode_read().astype(np.float32)
        CT_img = from_RGB(img)
        DCT_img = space_analyze(CT_img, self.block_size, self.block_size)
        decom_img = get_subbands(DCT_img, self.block_size, self.block_size)
        decom_k = self.quantize_decom(decom_img)
        decom_k += 128
        decom_k = decom_k.astype(np.uint8)
        #decom_k = np.clip(decom_k, 0, 255).astype(np.uint8)
        decom_k = self.compress(decom_k)
        self.encode_write(decom_k)
        rate = (self.output_bytes*8)/(img.shape[0]*img.shape[1])
        return rate

    def decode(self):
        decom_k = self.decode_read()
        decom_k = self.decompress(decom_k).astype(np.float32)
        decom_k -= 128
        decom_y = self.dequantize_decom(decom_k)
        DCT_y = get_blocks(decom_y, self.block_size, self.block_size)
        CT_y = space_synthesize(DCT_y, self.block_size, self.block_size)
        y = to_RGB(CT_y)
        y = np.clip(y, 0, 255).astype(np.uint8)
        self.decode_write(y)
        rate = (self.input_bytes*8)/(y.shape[0]*y.shape[1])
        return rate

    def quantize_decom(self, decom):
        decom_k = self.quantize(decom)
        return decom_k

    def dequantize_decom(self, decom_k):
        decom_y = self.dequantize(decom_k)
        return decom_y
    
    def perceptual_quantize_decom(self, decom):
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
                subband_k = np.empty_like(subband, dtype=np.int16)
                self.QSS *= (self.Y_QSSs[by,bx]/121)
                subband_k[..., 0] = self.quantize(subband[..., 0])
                self.QSS *= (self.C_QSSs[by,bx]/99)
                subband_k[..., 1] = self.quantize(subband[..., 1])
                subband_k[..., 2] = self.quantize(subband[..., 2])
                decom_k[by*subband_y_size:(by+1)*subband_y_size,
                        bx*subband_x_size:(bx+1)*subband_x_size,
                        :] = subband_k
        return decom_k

    def perceptual_dequantize_decom(self, decom_k):
        subbands_in_y = self.block_size
        subbands_in_x = self.block_size
        subband_y_size = int(decom_k.shape[0]/self.block_size)
        subband_x_size = int(decom_k.shape[1]/self.block_size)
        #decom_y = np.empty_like(decom_k, dtype=np.int16)
        decom_y = decom_k
        for by in range(subbands_in_y):
            for bx in range(subbands_in_x):
                subband_k = decom_k[by*subband_y_size:(by+1)*subband_y_size,
                                    bx*subband_x_size:(bx+1)*subband_x_size,
                                    :]
                subband_y = np.empty_like(subband_k, dtype=np.int16)
                self.QSS *= (self.Y_QSSs[by,bx]/121)
                subband_y[..., 0] = self.dequantize(subband_k[..., 0])
                self.QSS *= (self.C_QSSs[by,bx]/99)
                subband_y[..., 1] = self.dequantize(subband_k[..., 1])
                subband_y[..., 2] = self.dequantize(subband_k[..., 2])
                decom_k[by*subband_y_size:(by+1)*subband_y_size,
                        bx*subband_x_size:(bx+1)*subband_x_size,
                        :] = subband_y
        return decom_y

    def optimize_block_size(self):
        min = 1000000
        img = self.encode_read().astype(np.float32)
        for block_size in [2**i for i in range(1, 7)]:
            #block_size = 2**i
            CT_img = from_RGB(img)
            DCT_img = space_analyze(CT_img, block_size, block_size)
            decom_img = get_subbands(DCT_img, block_size, block_size)
            decom_k = self.quantize_decom(decom_img)
            decom_k += 128
            decom_k_bytes = self.compress(decom_k.astype(np.uint8))
            decom_k_bytes.seek(0)
            rate = len(decom_k_bytes.read())
            decom_k -= 128
            decom_y = self.dequantize_decom(decom_k)
            DCT_y = get_blocks(decom_y, block_size, block_size)
            CT_y = space_synthesize(DCT_y, block_size, block_size)
            y = to_RGB(CT_y)
            y = np.clip(y, 0, 255).astype(np.uint8)
            RMSE = distortion.RMSE(img, y)
            J = rate + self.Lambda*RMSE
            logging.info(f"J={J} for block_size={block_size}")
            if J < min:
                min = J
                self.block_size = block_size

if __name__ == "__main__":
    main.main(parser.parser, logging, CoDec)
