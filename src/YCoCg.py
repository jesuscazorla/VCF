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
        img = self.encode_read().astype(np.int16)
        # Specific for solving the issue https://github.com/vicente-gonzalez-ruiz/scalar_quantization/issues/1
        #img_128 = img.astype(np.int16) - 128
        #YCoCg_img_128 = from_RGB(img_128)
        #YCoCg_img = YCoCg_img_128 + 128
        YCoCg_img = from_RGB(img)
        #YCoCg_img[..., 1] += 128
        #YCoCg_img[..., 2] += 128
        #logging.debug(f"max(YCoCg_img)={np.max(YCoCg_img)}, min(YCoCg_img)={np.min(YCoCg_img)}")
        #assert (YCoCg_img < 256).all()
        #assert (YCoCg_img >= 0).all()
        k = self.quantize(YCoCg_img)
        logging.debug(f"k.shape={k.shape}, k.type={k.dtype}")
        #k = YCoCg_img
        k[..., 1] += 128
        k[..., 2] += 128
        compressed_k = self.compress(k.astype(np.uint8))
        self.encode_write(compressed_k)
        rate = (self.output_bytes*8)/(img.shape[0]*img.shape[1])
        return rate

    def decode(self):
        compressed_k = self.decode_read()
        k = self.decompress(compressed_k).astype(np.int16)
        logging.debug(f"k.shape={k.shape}, k.type={k.dtype}")
        k[..., 1] -= 128
        k[..., 2] -= 128
        YCoCg_y = self.dequantize(k)
        #YCoCg_y = k
#        logging.debug(f"max(YCoCg_y)={np.max(YCoCg_y)}, min(YCoCg_y)={np.min(YCoCg_y)}")
#        assert (YCoCg_y < 256).all()
#        assert (YCoCg_y >= 0).all()
#        logging.debug(f"YCoCg_y.shape={YCoCg_y.shape}, YCoCg_y.type={YCoCg_y.dtype}")
        # Specific for solving the issue https://github.com/vicente-gonzalez-ruiz/scalar_quantization/issues/1
        #YCoCg_y_128 = YCoCg_y.astype(np.int16) - 128
        #y_128 = to_RGB(YCoCg_y_128)
        #y = y_128 + 128
        #YCoCg_y[..., 1] -= 128
        #YCoCg_y[..., 2] -= 128        
        y = to_RGB(YCoCg_y)
        logging.debug(f"y.shape={y.shape}, y.type={y.dtype}")
        y = np.clip(y, 0, 255).astype(np.uint8)
        self.decode_write(y)
        rate = (self.input_bytes*8)/(k.shape[0]*k.shape[1])
        return rate

if __name__ == "__main__":
    main.main(parser.parser, logging, CoDec)
