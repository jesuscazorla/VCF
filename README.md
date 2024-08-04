# Visual Coding Framework
A programming environment to develop and test image and video compression algorithms.

## Install and configuration

Supposing a Python interpreter and Git available:

   python -m venv ~/envs/VCF  # Only if you want a new environment
   git clone git@github.com:Sistemas-Multimedia/VCF.git
   cd VCF
   source ~/envs/VCF/bin/activate
   pip install -r requirements
   cd src
   python PNG.py encode
   python PNG.py decode
   display /tmp/decoded.png

## Programming

### Image Coding

The simplest solution is to implement the methods `compressed_img = compress(img)` and `img = decompress(compressed_img)`, defined in the `entropy_image_coding` class interface. Notice that it is not necessary to read `img` when encoding, nor write `compressed_img` when decoding. See `src/PNM.py`.

### Video Coding

Again, it is necessary to implement the methods `None = compress()` and `None = decompress()`, defined in the `entropy_video_coding` class interface. See `src/MPNG.py`.