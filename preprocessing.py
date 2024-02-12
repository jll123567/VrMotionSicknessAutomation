# Preprocessing
#   Beat Saber: Flip(Depth)
#   Cartoon Network: Flip(Depth, Frame)
#   Epic Roller Coasters: Flip(Depth, Frame), OneEye
#   Job Sim: FLip(Frame), No Depth <== Skipping
#   Mini Motor Racing: Flip(Depth)
#   Monster Awakens: Flip(Depth, Frame), OneEye
#   Pottery: Flip(Depth)
#   Traffic Cop: Flip(Depth, Frame), OneEye
#   Voxel Shot: Flip(Depth, Frame), OneEye
#   Rome: FLip(Depth, Frame), OneEye
from PIL import Image
import os
import re

debug = False  # Print logs, do not overwrite images.
eyes = 2


def rename_debug(fp):
    """make a copy of the file."""
    im = Image.open(fp)
    fp = fp.split('/')
    fp[-1] = "debug_" + fp[-1]
    fp = '/'.join(fp)
    im.save(fp)
    return fp


def resize_image(im: Image, factor: int = 0.5):
    """Take an image defined by path and scale it by factor(0.5 is halved)"""
    width, height = im.size
    new_width, new_height = int(width * factor), int(height * factor)
    return im.resize((new_width, new_height), resample=Image.NEAREST)


def vertical_flip(im: Image):
    """Flip an image vertically(on horizontal axis)"""
    return im.transpose(Image.FLIP_TOP_BOTTOM)


def to_twoeye(im: Image):
    """Convert a single eye image to a two eye image(by duplication)"""
    new_size = (2016, 1042)
    result = Image.new("RGB", new_size)
    im = im.crop((0, 0, im.size[0] - 16, im.size[1]))
    result.paste(im, (0, 0))
    result.paste(im, (int(2016 / 2), 0))
    return result


def to_oneeye(im: Image):
    """Convert a two eye image to a single eye image(by cropping)"""
    return im.crop((1008, 0, im.size[0] + 16, im.size[1]))


def enforce_eyes(im: Image, current_eyes: int):
    if eyes > 2 or eyes < 1 or current_eyes > 2 or current_eyes < 1:
        raise Exception(
            f"Eyes settings are out of range. Please fix. {eyes} and {current_eyes} should be either 1 or 2")
    if eyes == 2 and current_eyes == 1:
        return to_twoeye(im)
    elif eyes == 1 and current_eyes == 2:
        return to_oneeye(im)
    else:
        return im


if __name__ == "__main__":
    root_path = "/home/lambda8/ledbetterj1_VRMotionSickness/dataset/VRNetDataCollection"
    print("Getting Paths")
    paths = [str(os.path.join(dirpath, f)) for (dirpath, dirnames, filenames) in os.walk(root_path) for f in filenames
             if re.search(r'[ds][0-9]+\.jpg', f)]
    d_paths = [p for p in paths if re.search(r'd[0-9]+\.jpg', p)]
    print(f"Got {len(paths)} images of which {len(d_paths)} are depth.")
    for path in paths:
        if debug:
            path = rename_debug(path)
        img = Image.open(path)
        if re.search(r'Beat_Saber', path):
            print(f"[Beat_Saber, F D, 2] {path}")
            img = enforce_eyes(img, 2)
            img = resize_image(img)
            if path in d_paths:
                img = vertical_flip(img)
        elif re.search(r'Cartoon_Network', path):
            print(f"[Cartoon_Network, F SD, 2] {path}")
            img = enforce_eyes(img, 2)
            img = resize_image(img)
            img = vertical_flip(img)
        elif re.search(r'Epic_Roller_Coasters', path):
            print(f"[Epic_Roller_Coasters, F SD, 1] {path}")
            img = enforce_eyes(img, 1)
            img = resize_image(img)
            img = vertical_flip(img)
        elif re.search(r'Job_Simulator', path):
            print(f"[Job_Simulator, Skip] {path}")
            continue
        elif re.search(r'Mini_Motor_Racing', path):
            print(f"[Mini_Motor_Racing, F D, 2] {path}")
            img = enforce_eyes(img, 2)
            img = resize_image(img)
            if path in d_paths:
                img = vertical_flip(img)
        elif re.search(r'Monster_Awakens', path):
            print(f"[Monster_Awakens, F SD, 1] {path}")
            img = enforce_eyes(img, 1)
            img = resize_image(img)
            img = vertical_flip(img)
        elif re.search(r'Pottery', path):
            print(f"[Pottery, F D, 2] {path}")
            img = enforce_eyes(img, 2)
            img = resize_image(img)
            if path in d_paths:
                img = vertical_flip(img)
        elif re.search(r'Traffic_Cop', path):
            print(f"[Traffic_Cop, F SD, 1] {path}")
            img = enforce_eyes(img, 1)
            img = resize_image(img)
            img = vertical_flip(img)
        elif re.search(r'Voxel_Shot_VR', path):
            print(f"[Voxel_Shot_VR, F SD, 1] {path}")
            img = enforce_eyes(img, 1)
            img = resize_image(img)
            img = vertical_flip(img)
        elif re.search(r'VR_Rome', path):
            print(f"[VR_Rome, F SD, 1] {path}")
            img = enforce_eyes(img, 1)
            img = resize_image(img)
            img = vertical_flip(img)
        else:
            raise Exception(f"Started operating on an invalid game at path {path}")
        img.save(path)
    if debug:
        input("Press enter to delete debug.")
        paths = [os.path.join(dirpath, f) for (dirpath, dirnames, filenames) in os.walk(root_path) for f in filenames]
        for path in paths:
            if "debug" in path.split('/')[-1]:
                os.remove(path)
