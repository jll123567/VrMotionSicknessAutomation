# Worklog

## Dataset cleaning

The VRNet dataset is somewhat disorganized.
The dataset contains a set of recordings for each game.
Each recording has a set of frames(s###.jpg), their motion vectors(m###.zfp) and depth(d###.jpg), as well as a 'gamedata.log' and a voice.csv(the motion sickness ratings).
Each game also has .7z with more .csv files for the camera, controllers, in game objects, lights, etc.
The dataset comes as a large collection of zip files, some are un-organized recordings while others contain folders for each game, the game folders containing the game.7z and maybe some recordings.

This meant that once all the .zips were decompressed, they needed to be sorted into their respective game folders.
Once this was done, the game.7zs needed to be decompressed and their contents put into the correct recording folders.

The format of the .zips was inconsistent, especially data for the 'Monster Awakens' game,
where some recordings were put into folders labeled 'Participant #' unlike the others (P## VRLOG-#######/VRLOG-#######).
A fair amount of time was also spent making all the names uniform (game_name/P## VRLOG-#######).

Also included was SelfReportData.zip but these were just copies of each recording's voice.csv.
Because of this, it made sense to not include the redundant data.

The dataset images are inconsistent from game to game.
Some games have both eyes, some only have one.
Some games flip the frame and depth, others only the depth.
(Why are any of the images flipped anyway!!!!????)

THe images are huge too.
I'm going to half the resolution.
They are jpg compressed to hell and back, but still, resolution is resolution.

## Feature Engineering/Selection

I still need to figure out what data I do/don't need.

I likely won't need lighting data as this tends to remain static in most scenes.
Depth seems iffy to me, but I can see cases where it might be useful.

I definitely want the camera, controller, frame, and motion vector data though.
camera data for point of view and viewpoint motion (separate from HMD motion).
controller for the player's real world movements.
frame and motion vector data will be a different matter.

Using Pillow(python library) to flip images and resize them to half-resolution.
I also need to either crop them in or double them for the two/one eye problem.
If I double the "one-eyes" I might flip the opposite eye.

Went to the motionsickness ratings in each recording's voice.csv and did some preprocessing.
Using camera.csv as a template, I added the nearest frame to each rating and expanded over the entire recording.
Timestamps now folow those in camera.csv as well.
Since the ratings are kinda sparse, I interpolated (linear, rounded to 0 decimals) between each rating given.
I made an assumption that ratings start at one at the beginning of the recording (user is not sick)
and end with the last rating held to the end.

## Loading Data
ZFP is a nightmare to load if the files don't have headers, which they don't.
I emailed the guy listed as the designated contact(feb 12), so help me god they get back to me.
Haven't heard back yet, emailed again, but to a different email they have listed at their personal site(feb 15)

Loading each csv file into pandas and then converting it to a nicer format.
First was voice, easy enough, grab the rating from voice_preproc.csv
Next was camera, had to do some things to convert the projection and view matricies since they were strings.
Then was the controllers, I needed to dump a bunch of rows and columns, then needed to merge three rows into one.
Then the pose, which was a collection of both problems from camera and control.
These three are the numeric inputs (X_n).
Load_images was straightforward, mostly copy-paste from last time(autoencodevr), though this time more efficient since I read the tensorflow docs more closely.
