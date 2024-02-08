# Preprocessing
#   Beat Saber: Flip(Depth)
#   Cartoon Network: Flip(Depth, Frame)
#   Epic Roller Coasters: Flip(Depth, Frame), OneEye
#   Job Sim: FLip(Frame), No Depth
#   Mini Motor Racing: Flip(Depth)
#   Monster Awakens: Flip(Depth, Frame), OneEye
#   Pottery: Flip(Depth)
#   Traffic Cop: Flip(Depth, Frame), OneEye
#   Voxel Shot: Flip(Depth, Frame), OneEye
#   Rome: FLip(Depth, Frame), OneEye
from PIL import Image, ImageDraw
import os
import re

debug = True  # Print logs, do not overwrite images.


def rename_debug(path):
    """make a copy of the file."""
    im = Image.open(path)
    path = path.split('/')
    path[-1] = "debug_" + path[-1]
    path = '/'.join(path)
    im.save(path)
    return path


def resize_image(im: Image, factor: int = 0.5):
    """Take an image defined by path and scale it by factor(0.5 is halfed)"""
    width, height = im.size
    new_width, new_height = int(width * factor), int(height * factor)
    return im.resize((new_width, new_height), resample=Image.NEAREST)


def vertical_flip(im: Image):
    """Flip an image vertically(on horizontal axis)"""
    return im.transpose(Image.FLIP_TOP_BOTTOM)


def to_twoeye(im: Image, name: str, reflect=False, flipped=True):
    """Convert a single eye image to a two eye image(by duplication)"""
    new_size = (2016, 1042)
    result = Image.new("RGB", new_size)
    im = im.crop((0, 0, im.size[0] - 16, im.size[1]))
    result.paste(im, (0, 0))
    result.paste(im, (int(2016 / 2), 0))
    result.show()
    return result


def to_oneeye(im: Image, name: str):
    """Convert a two eye image to a single eye image(by cropping)"""
    return im.crop((1008, 0, im.size[0] + 16, im.size[1]))


if __name__ == "__main__":
    root_path = "/home/lambda8/Desktop/preproc_debug"
    paths = [str(os.path.join(dirpath, f)) for (dirpath, dirnames, filenames) in os.walk(root_path) for f in filenames]
    for path in paths:
        if debug:
            path = rename_debug(path)
        name = path.split('/')[-1]
        img = Image.open(path)
        img = to_twoeye(img, name)
        # img = to_oneeye(img, name)  # Run Eye conversions before any other operation.
        # img = resize_image(img)
        # img = vertical_flip(img)
        img.save(path)
    if debug:
        input("Press enter to delete debug.")
        paths = [os.path.join(dirpath, f) for (dirpath, dirnames, filenames) in os.walk(root_path) for f in filenames]
        for path in paths:
            if "debug" in path.split('/')[-1]:
                os.remove(path)
