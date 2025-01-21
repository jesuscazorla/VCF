'''Exploiting color (perceptual) redundancy with the DCT transform.'''

import numpy as np
import logging
import main
import importlib
with open("/tmp/description.txt", 'w') as f:  # Used by parser.py
    f.write(__doc__)
import parser

from color_transforms.DCT import from_RGB # pip install "color_transforms @ git+https://github.com/vicente-gonzalez-ruiz/color_transforms"
from color_transforms.DCT import to_RGB

default_quantizer = "deadzone"

parser.parser_encode.add_argument("-c", "--quantizer", help=f"Quantizer (default: {default_quantizer})", default=default_quantizer)
parser.parser_decode.add_argument("-c", "--quantizer", help=f"Quantizer (default: {default_quantizer})", default=default_quantizer)

args = parser.parser.parse_known_args()[0]
Q = importlib.import_module(args.quantizer)

class CoDec(Q.CoDec):

    def __compress(self, img):
        DCT_img = from_RGB(img)
        compressed_k = super().compress(DCT_img)
        return compressed_k

    def __decompress(self, compressed_k):
        DCT_y = super().decompress(compressed_k)
        y = to_RGB(DCT_y)
        y = np.clip(y, 0, 255)
        y = y.astype(np.uint8)
        return y

    def encode(self):
        img = self.encode_read()#.astype(np.int16)
        #img -= 128
        img = img.astype(np.int16) - 128
        #img = img.astype(np.uint8)
        DCT_img = from_RGB(img)
        k = self.quantize(DCT_img)
        print("------------->", k.dtype, np.max(k), np.min(k))
        k += 128
        k = k.astype(np.uint16)
        print("------------->", k.dtype, np.max(k), np.min(k))
        if np.max(k) > 255:
            logging.warning(f"k[{np.unravel_index(np.argmax(k),k.shape)}]={np.max(k)}")
        if np.min(k) < 0:
            logging.warning(f"k[{np.unravel_index(np.argmin(k),k.shape)}]={np.min(k)}")
        #k = k.astype(np.uint16)
        compressed_k = self.compress(k)
        self.encode_write(compressed_k)
        #self.BPP = (self.output_bytes*8)/(img.shape[0]*img.shape[1])
        #logging.info(f"BPP = {BPP}")

    def decode(self):
        compressed_k = self.decode_read()
        k = self.decompress(compressed_k)
        k = k.astype(np.int32)
        k -= 128
        #k = self.read()
        #k -= 32768
        print("------------->", k.dtype, np.max(k), np.min(k))
        DCT_y = self.dequantize(k)
        #y_128 = to_RGB(DCT_y.astype(np.int16))
        #DCT_y = DCT_y.astype(np.uint8)
        y = to_RGB(DCT_y)
        y = y.astype(np.int16) + 128
        #y += 128
        #y = (y_128.astype(np.int16) + 128)
        if np.max(y) > 255:
            logging.warning(f"y[{np.unravel_index(np.argmax(y),y.shape)}]={np.max(y)}")
        if np.min(y) < 0:
            logging.warning(f"y[{np.unravel_index(np.argmin(y),y.shape)}]={np.min(y)}")
        y = np.clip(y, 0, 255).astype(np.uint8)
        self.decode_write(y)
        #self.BPP = (self.input_bytes*8)/(k.shape[0]*k.shape[1])
        #RMSE = distortion.RMSE(self.encode_read(), y)
        #logging.info(f"RMSE = {RMSE}")

if __name__ == "__main__":
    main.main(parser.parser, logging, CoDec)
