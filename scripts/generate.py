"""generate

Usage: generate.py <image>...

Options:
  -h --help         Show this screen.
  --version         Show the version.

"""

import sys
import cv2
from progress.bar import Bar
from docopt import docopt

if __name__ == '__main__':
    ARGUMENTS = docopt(__doc__, options_first=True, version='1.0.0')
    IMAGES = ARGUMENTS["<image>"]
    BAR = Bar('Progress', max=len(IMAGES))

    for filename in IMAGES:
        image = cv2.imread(filename)
        akaze = cv2.AKAZE.create()
        keypoints, descriptors = akaze.detectAndCompute(image, None)
        for descriptor in descriptors:
            sys.stdout.buffer.write(descriptor)
        BAR.next()
    BAR.finish()
