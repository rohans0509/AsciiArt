"""
    Handles video conversions by sending individual frames to image_processor

"""
import cv2
from tqdm import tqdm

from AsciiArt.imageprocessor import ImageProcessor


class VideoProcessor:

    def __init__(self, file):
        video = cv2.VideoCapture(file)
        self.video = video
        self.width = int(video.get(3))
        self.height = int(video.get(4))
        self.size = (self.width, self.height)
        self.fps = video.get(cv2.CAP_PROP_FPS)
        self.length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    def videoToAscii(self, output="out.mp4", quality=0.1, start=0, end=0, show=False, asciiscale=""):
        self.setparams(output, asciiscale, start)

        print("Converting video please wait...")

        improcessor = ImageProcessor()

        for frame_number in tqdm(range(start, end)):
            exists, frame = self.video.read()
            if exists:
                # Image to Ascii returns np array
                frame = self.format(frame)
                outframe = improcessor.imageToAscii(frame, quality, self.scale)
                self.result.write(outframe)

                # Shows rendering
                if show:
                    cv2.imshow("Frame", outframe)

                if cv2.waitKey(1) & 0xFF == ord('s'):
                    break

            else:
                break

        # When everything done, close everything
        self.video.release()
        self.result.release()
        cv2.destroyAllWindows()

        print(f"The video was successfully saved as {output}")
        return self.result

    def setparams(self, output, asciiscale, start):
        if asciiscale == "":
            self.scale = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,"'. "'
        else:
            self.scale = asciiscale

        self.video.set(cv2.CAP_PROP_FRAME_COUNT, start - 1)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.result = cv2.VideoWriter(output, fourcc, self.fps, self.size)

        print(f"Size of video is {self.width}x{self.height} at {self.fps} fps")

    def format(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
