"""
generate

Usage:
  generate.py <descriptor> <image>...

Options:
  -h --help         Show this screen.
  --version         Show the version.

<descriptor> can be one of the following:
  * akaze (outputs 61 bytes)
  * kaze (outputs 64 32-bit floats)
"""

import sys
import struct
import itertools
import cv2
from progress.bar import Bar
from docopt import docopt

if __name__ == '__main__':
    ARGUMENTS = docopt(__doc__, options_first=True, version='1.0.0')
    IMAGES = ARGUMENTS["<image>"]
    DESCRIPTOR = ARGUMENTS["<descriptor>"]
    BAR = Bar('Progress', max=len(IMAGES))

    if DESCRIPTOR == "akaze":
        DESCRIPTOR_OBJECT = cv2.AKAZE.create()
        BINARY = True
    elif DESCRIPTOR == "kaze":
        DESCRIPTOR_OBJECT = cv2.KAZE.create()
        BINARY = False
    else:
        sys.exit("unknown descriptor {}".format(DESCRIPTOR))

    for filename in IMAGES:
        image = cv2.imread(filename)
        keypoints, descriptors = DESCRIPTOR_OBJECT.detectAndCompute(
            image, None)

        for descriptor in descriptors:
            if BINARY:
                sys.stdout.buffer.write(descriptor)
            else:
                sys.stdout.buffer.write(bytes(itertools.chain.from_iterable(
                    map(lambda f: struct.pack('<f', f), descriptor))))
        BAR.next()
    BAR.finish()
