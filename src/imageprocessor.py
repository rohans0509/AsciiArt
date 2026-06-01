"""
    Converts images to Ascii form

"""
import math

import cv2


class ImageProcessor:
    # Here all images are np arrays
    def imageToAscii(self, image, quality, scale):
        self.setparams(image, quality)
        image = self.fontresize(image)
        # todo create new black background
        # todo figure out a way to type on this new image

    def setparams(self, image, quality):
        self.quality = quality
        self.oldwidth = image.shape[1]
        self.oldheight = image.shape[0]
        self.oldsize = (self.oldwidth, self.oldheight)
        # Resize the image on the basis of the proportions of the font and scale it down by the quality factor
        charWidth = 10
        charHeight = 18
        aspectChar = charWidth / charHeight
        adjustedwidth = int(self.oldwidth * quality)
        adjustedheight = int(self.oldheight * quality * aspectChar)
        # Each pixel of the the adjusted image will be represented by an ascii character
        self.newsize = (adjustedwidth * charWidth, adjustedheight * charHeight)

    def getchar(self, scale, greyvalue):
        index = math.floor(greyvalue / 255 * len(scale))
        return scale[index]

    def luminance(self, r, g, b):
        # Returns converted value to gray scale
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    def fontresize(self, image):
        return cv2.resize(image, self.newsize)
