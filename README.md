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