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

# Missing Camera.csv(do not import)
#   P3 VRLOG-5051017
#   P3 VRLOG-5051033
#   P20 VRLOG-6051722
#   P19 VRLOG-6051703
#   P21 VRLOG-6051736
#   P18 VRLOG-6051153
from PIL import Image
import os
import re
import pandas as pd
import numpy as np

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


def preprocess_images(root_path):
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


def get_nearest_frame(ts: int, camera: pd.DataFrame) -> pd.Series:
    """Get the nearest framecount and timestamp from camera that's greater equal to a given timestamp."""
    ts = ts / 1000  # Remove some digits of precision to make comparison work.

    # The row with a timestamp greater or equal to ts, only the framecount and timestamp
    return camera[camera["timestamp"] >= ts].iloc[0, 0]


def preprocess_voice(root_path: str) -> None:
    # Load voice.
    voice = pd.read_csv(root_path + "/voice.csv", header=None)
    voice.columns = ["timestamp", "rating", "method"]

    # Load camera(decent refrence for framecount and timestamps).
    camera = pd.read_csv(root_path + "/camera.csv")

    # Add framecounts to voice from camera.
    voice["nearest_frame"] = voice.apply(lambda r: get_nearest_frame(r.iloc[0], camera), axis=1)

    # Expand voice for each third frame (what's recorded in camera)
    last_frame = camera.iloc[-1, 0]  # Last col(nearest frame) of last row
    new_index = pd.Index(np.arange(3, last_frame + 3, 3), name="nearest_frame")
    voice = voice.set_index(["nearest_frame"]).reindex(new_index).reset_index()
    voice["timestamp"] = camera["timestamp"]  # Copy timestamps from voice to camera (precision is lower but w/e)

    # Add values to begin and end for interpolation.
    voice.iloc[0, 2] = 1  # Assume starting with no sickness.
    voice.iloc[-1, 2] = voice["rating"][voice["rating"].notnull()].iloc[-1]  # Assume ends with last recorded rating.

    # Interpolate, round ratings.
    voice = voice.interpolate(method="slinear").astype({"rating": int})

    # Fill method on interpolated.
    voice.loc[voice["method"].isnull(), "method"] = "interpolated"

    # write out new
    voice.to_csv(root_path + "/voice_preproc.csv", index=False)


def preprocess_all_voice(root_path: str) -> None:
    paths = [str(os.path.join(dirpath, d)) for (dirpath, dirnames, filenames) in os.walk(root_path) for d in dirnames
             if re.match(r'P[0-9]{1,2} VRLOG-[0-9]{7}', d)]
    paths = set(paths)
    for i, path in enumerate(paths):
        print(f"{i}/{len(paths)}Preprocessing voice.csv in: {path}")
        try:
            preprocess_voice(path)
        except FileNotFoundError:
            print("  No camera.csv, skipping.")


if __name__ == "__main__":
    # preprocess_images("/home/lambda8/ledbetterj1_VRMotionSickness/dataset/VRNetDataCollection")
    # preprocess_all_voice("/home/lambda8/ledbetterj1_VRMotionSickness/dataset/VRNetDataCollection")
    pass
