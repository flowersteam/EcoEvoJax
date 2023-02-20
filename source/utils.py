""" This script contains general utilities.
"""


import moviepy.editor as mvp
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from IPython.display import HTML, display, clear_output
import numpy as np
from moviepy.editor import *



def merge_videos(directory, num_gens):
    """ Merge multiple videos into a single one.

    Attributes
    ----------
    directory: str
        name of directory where videos are saved

    num_gens: int
        last generation
    """
    gens = range(0, num_gens, 50) # loading every 50 generations
    L = []

    for gen in gens:
        file_path = "projects/" + directory + "/train/media/gen_" + str(gen) + ".mp4"
        video = VideoFileClip(file_path)
        L.append(video)

    final_clip = concatenate_videoclips(L)
    final_clip.to_videofile("projects/" + directory + "/total_training.mp4", fps=24, remove_temp=False)



class VideoWriter:
    """ Class for saving videos.
    """
    def __init__(self, filename, fps=30.0, **kw):
        self.writer = None
        self.params = dict(filename=filename, fps=fps, **kw)

    def add(self, img):
        img = np.asarray(img)
        if self.writer is None:
            h, w = img.shape[:2]
            self.writer = FFMPEG_VideoWriter(size=(w, h), **self.params)
        if img.dtype in [np.float32, np.float64]:
            img = np.uint8(img.clip(0, 1) * 255)
        if len(img.shape) == 2:
            img = np.repeat(img[..., None], 3, -1)
        self.writer.write_frame(img)

    def close(self):
        if self.writer:
            self.writer.close()

    def __enter__(self):
        return self

    def __exit__(self, *kw):
        self.close()

    def show(self, **kw):
        self.close()
        fn = self.params['filename']
        display(mvp.ipython_display(fn, **kw))

