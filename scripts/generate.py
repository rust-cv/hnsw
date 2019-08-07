"""generate

Usage: generate.py <image>...

Options:
  -h --help         Show this screen.
  --version         Show the version.

"""

import cv2
import os
import sys
import os.path as path
from progress.bar import Bar
from docopt import docopt

if __name__ == '__main__':
    arguments = docopt(__doc__, options_first=True, version='1.0.0')
    images = arguments["<image>"]
    bar = Bar('Progress', max=len(images))

    for filename in images:
        image = cv2.imread(filename)
        akaze = cv2.AKAZE.create()
        keypoints, descriptors = akaze.detectAndCompute(image, None)
        for descriptor in descriptors:
            sys.stdout.buffer.write(descriptor)
        bar.next()
    bar.finish()
