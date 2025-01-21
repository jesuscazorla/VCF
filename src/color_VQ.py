'''Image (spatial 2D) quantization using a vector quantizer.'''

import os
import numpy as np
import logging
import main
with open("/tmp/description.txt", 'w') as f:  # Used by parser.py
    f.write(__doc__)
import parser
from sklearn import cluster # pip install scikit-learn
from sklearn.utils import shuffle
import entropy_image_coding as EIC
import importlib

from information_theory import information  # pip install "information_theory @ git+https://github.com/vicente-gonzalez-ruiz/information_theory"


  
default_EIC = "PNG"
default_N_clusters = 8

parser.parser_encode.add_argument("-e", "--entropy_image_codec", help=f"Entropy Image Codec (default: {default_EIC})", default=default_EIC)
parser.parser_decode.add_argument("-e", "--entropy_image_codec", help=f"Entropy Image Codec (default: {default_EIC})", default=default_EIC)
parser.parser_encode.add_argument("-n", "--N_clusters", type=parser.int_or_str, help=f"Number of clusters (default: {default_N_clusters})", default=default_N_clusters)
parser.parser_decode.add_argument("-n", "--N_clusters", type=parser.int_or_str, help=f"Number of clusters (default: {default_N_clusters})", default=default_N_clusters)

args = parser.parser.parse_known_args()[0]
EC = importlib.import_module(args.entropy_image_codec)

class CoDec(EC.CoDec):

    def __init__(self, args, min_index_val=0, max_index_val=255):
        super().__init__(args)
        logging.debug(f"args = {self.args}")
        self.N_clusters = args.N_clusters
        self.input = args.input
        self.output = args.output

    def encode(self):
        img = self.encode_read()
        #k, centroids = self.quantize(img)
        k = self.quantize(img)
        compressed_k = self.compress(k)
        self.encode_write(compressed_k)
        #self.encode_write_fn(compressed_k, self.output + "_labels")
        #compressed_centroids = self.compress(centroids)
        #self.encode_write_fn(compressed_centroids, self.output + "_centroids")
    
    def decode(self):
        compressed_k = self.decode_read()
        #compressed_centroids = self.decode_read_fn(self.input + "_centroids")
        #compressed_k = self.decode_read_fn(self.input + "_labels")
        #centroids = self.decompress(compressed_centroids)
        k = self.decompress(compressed_k)
        #y = self.dequantize(k, centroids)
        y = self.dequantize(k)
        self.decode_write(y)

    def quantize(self, img):
        img = np.array(img, dtype=np.float64) / 255
        w,h,d = tuple(img.shape)
        assert d == 3
        image_array = np.reshape(img, (w * h,d))        
        image_array_sample = shuffle(image_array, random_state = 0, n_samples=1_000)
        kmeans = cluster.KMeans(n_clusters=self.N_clusters, random_state=0).fit(image_array_sample)
        labels = kmeans.predict(image_array)
        labels = labels.reshape((w,h))
        labels = labels.astype(np.uint8)
        centroids = kmeans.cluster_centers_
        fn = self.output + "_centroids.npz"
        np.savez_compressed(file=fn, a=centroids)
        self.output_bytes += os.path.getsize(fn)
        return labels

    def dequantize(self, labels):
        fn = self.input + "_centroids.npz"
        self.input_bytes += os.path.getsize(fn)
        centroids = np.load(file=fn)['a']
        return (centroids[labels].reshape(labels.shape[0], labels.shape[1], -1) * 255).astype(np.uint8)

if __name__ == "__main__":
    main.main(parser.parser, logging, CoDec)