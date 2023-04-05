'''Exploiting color (perceptual) redundancy with the YCoCg transform.'''

import numpy as np
import logging
import main
import importlib
import parser

from color_transforms.YCoCg import from_RGB # pip install "color_transforms @ git+https://github.com/vicente-gonzalez-ruiz/color_transforms"
from color_transforms.YCoCg import to_RGB

default_quantizer = "deadzone"

parser.parser_encode.add_argument("-c", "--quantizer", help=f"Quantizer (default: {default_quantizer})", default=default_quantizer)
parser.parser_decode.add_argument("-c", "--quantizer", help=f"Quantizer (default: {default_quantizer})", default=default_quantizer)

args = parser.parser.parse_known_args()[0]
Q = importlib.import_module(args.quantizer)

class CoDec(Q.CoDec):

    def encode(self):
        img = self.encode_read()
        # Specific for solving the issue https://github.com/vicente-gonzalez-ruiz/scalar_quantization/issues/1
        #img_128 = img.astype(np.int16) - 128
        #YCoCg_img_128 = from_RGB(img_128)
        #YCoCg_img = YCoCg_img_128 + 128
        YCoCg_img = from_RGB(img.astype(np.int16))
        k = self.quantize(YCoCg_img)
        compressed_k = self.compress(k)
        self.encode_write(compressed_k)
        rate = (self.output_bytes*8)/(img.shape[0]*img.shape[1])
        return rate

    def decode(self):
        compressed_k = self.decode_read()
        k = self.decompress(compressed_k)
        YCoCg_y = self.dequantize(k)
        # Specific for solving the issue https://github.com/vicente-gonzalez-ruiz/scalar_quantization/issues/1
        #YCoCg_y_128 = YCoCg_y.astype(np.int16) - 128
        #y_128 = to_RGB(YCoCg_y_128)
        #y = y_128 + 128
        y = to_RGB(YCoCg_y.astype(np.int16))
        y = np.clip(y, 0, 255).astype(np.uint8)
        self.decode_write(y)
        rate = (self.input_bytes*8)/(k.shape[0]*k.shape[1])
        return rate

if __name__ == "__main__":
    main.main(parser.parser, logging, CoDec)
