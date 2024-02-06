# Worklog

## Dataset cleaning

The VRNet dataset is somewhat disorganized.
The dataset contains a set of recordings for each game.
Each recording has a set of frames(s###.jpg), their motion vectors(m###.zfp) and depth(d###.jpg), as well as a 'gamedata.log' and a voice.csv(the motion sickness ratings).
Each game also has .7z with more .csv files for the camera, controllers, in game objects, lights, etc.
The dataset comes as a large colletion of zip files, some are un-organized recordings while others contain folders for each game, the game folders containing the game.7z and maybe some recordings.

This meant that once all the .zips were decompressed they needed to be sorted into their respective game folders.
Once this was done, the game.7zs needed to be decompressed and their contents put into the correct recording folders.

The format of the .zips were inconsistant, especially data for the 'Monster Aweakens' game, where some recodings were put into folders labeld 'Participant #' unlike the others (P## VRLOG-#######/VRLOG-#######).
A fair amount of time was also spent making all the names uniform (game_name/P## VRLOG-#######).

Also included was SelfReportData.zip but these were jsut copies of each recording's voice.csv.
Because of this it made sense to not include thee redundant data.


As you can tell rustdesk is not happy I type so fast.

## Feature Engineering/Selection

I still need to figure out what data I do/dont need.

I likely wont need lighting data as this tends to remain static in most scenes.
Depth seems iffy to me buut I can  see casessss where i might be useful.

I definitely want the camera, controler, frame, and motion vector dataaa though.
camera data for point of view and viewpoint motion(seperate from HMD motion).
controler for the player's real world movements.
frame and motion vector data will be a different matter.

