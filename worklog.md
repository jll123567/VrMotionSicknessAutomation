# Worklog

## Dataset cleaning

The VRNet dataset is somewhat disorganized.
The dataset contains a set of recordings for each game.
Each recording has a set of frames(s###.jpg), their motion vectors(m###.zfp) and depth(d###.jpg), as well as a 'gamedata.log' and a voice.csv(the motion sickness ratings).
Each game also has .7z with more .csv files for the camera, controllers, in game objects, lights, etc.
The dataset comes as a large collection of zip files, some are un-organized recordings while others contain folders for each game, the game folders containing the game.7z and maybe some recordings.

This meant that once all the .zips were decompressed they needed to be sorted into their respective game folders.
Once this was done, the game.7zs needed to be decompressed and their contents put into the correct recording folders.

The format of the .zips were inconsistent, especially data for the 'Monster Awakens' game, where some recodings were put into folders labeld 'Participant #' unlike the others (P## VRLOG-#######/VRLOG-#######).
A fair amount of time was also spent making all the names uniform (game_name/P## VRLOG-#######).

Also included was SelfReportData.zip but these were just copies of each recording's voice.csv.
Because of this it made sense to not include the redundant data.

The datasets images are inconsistent from game to game.
Some games have both eyes, some only have one.
Some games flip the frame and depth, others only the depth.
(Why are any of the images flipped anyway!!!!????)

THe images are huge too.
I'm going to half the resolution.
They are jpg compressed to hell and back but still , resolution is resolution.

## Feature Engineering/Selection

I still need to figure out what data I do/don't need.

I likely won't need lighting data as this tends to remain static in most scenes.
Depth seems iffy to me, but I can  see cases where it might be useful.

I definitely want the camera, controller, frame, and motion vector data though.
camera data for point of view and viewpoint motion(separate from HMD motion).
controller for the player's real world movements.
frame and motion vector data will be a different matter.

Using Pillow(python library) to flip images and resize them to half-resolution.
I also need to either crop them in or double them for the two/one eye problem.
If I double the oneyes I might flip the opisite eye.
