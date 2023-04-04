'''Entropy Encoding of images using Zlib.'''

import entropy_encoding_image as EEI
import io
import numpy as np
import main
import logging

class CoDec (EEI.CoDec):

    def compress(self, img):
        compressed_img = io.BytesIO()
        np.savez_compressed(file=compressed_img, a=img)
        return compressed_img

    def decompress(self, compressed_img):
        compressed_img = io.BytesIO(compressed_img)
        img = np.load(compressed_img)['a']
        return img

    def _encode_write_fn(self, compressed_img, fn):
        '''Write to disk the image <compressed_img> with filename <fn>.'''
        compressed_img.seek(0)
        with open(fn, "wb") as output_file:
            output_file.write(compressed_img.read())
        self.output_bytes += os.path.getsize(fn)
        logging.info(f"Written {os.path.getsize(fn)} bytes in {fn}")

    def _encode(self):
        '''Read an image, compress it with Zlib, and save it in the disk.
        '''
        img = self.encode_read()
        compressed_img = self.compress(img)
        #compressed_img = io.BytesIO()
        #np.savez_compressed(file=compressed_img, a=img)
        #compressed_img.seek(0)
        #np.save(file=img_for_disk, arr=img)
        #compressed_img = zlib.compress(img_for_disk, COMPRESSION_LEVEL)
        #print(len(compressed_img.read()))
        #with open("/tmp/1.Zlib", "wb") as output_file:
        #    output_file.write(compressed_img)
        #x = zlib.decompress(compressed_img)
        self.encode_write(compressed_img)
        logging.debug(f"output_bytes={self.output_bytes}, img.shape={img.shape}")
        rate = (self.output_bytes*8)/(img.shape[0]*img.shape[1])
        return rate

    def _decode(self):
        '''Read a compressed image, decompress it, and save it.'''
        compressed_img = self.decode_read()
        compressed_img_diskimage = io.BytesIO(compressed_img)
        img = np.load(compressed_img_diskimage)['a']
        #decompressed_data = zlib.decompress(compressed_img)
        #img = io.BytesIO(decompressed_data))
        self.decode_write(img)
        logging.debug(f"output_bytes={self.output_bytes}, img.shape={img.shape}")
        rate = (self.output_bytes*8)/(img.shape[0]*img.shape[1])
        return rate

if __name__ == "__main__":
    main.main(EEI.parser, logging, CoDec)
