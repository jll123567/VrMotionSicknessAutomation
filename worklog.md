# Worklog(Thesis)

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

Went to the motion-sickness ratings in each recording's voice.csv and did some preprocessing.
Using camera.csv as a template, I added the nearest frame to each rating and expanded over the entire recording.
Timestamps now follow those in camera.csv as well.
Since the ratings are kinda sparse, I interpolated (linear, rounded to 0 decimals) between each rating given.
I made an assumption that ratings start at one at the beginning of the recording (user is not sick)
and end with the last rating held to the end.

## Loading Data
ZFP is a nightmare to load if the files don't have headers, which they don't.
I emailed the guy listed as the designated contact(feb 12), so help me god they get back to me.
Haven't heard back yet, emailed again, but to a different email they have listed at their personal site(feb 15)

Loading each csv file into pandas and then converting it to a nicer format.
First was voice, easy enough, grab the rating from voice_preproc.csv
Next was camera, had to do some things to convert the projection and view matrices since they were strings.
Then was the controllers, I needed to dump a bunch of rows and columns, then needed to merge three rows into one.
Then the pose, which was a collection of both problems from camera and control.
These three are the numeric inputs (X_n).
Load_images was straightforward, mostly copy-paste from last time(autoencodevr), though this time more efficient since I read the tensorflow docs more closely.
Had to make sure that I don't output a list from map functions but a tensor, since a list gets a tensor of all elements (which are also tensors) instead of a tensor of just the values.

## Model Creation
I'm just following a tutorial on the keras site.
Three conv1d layers, relu activation, batch normalization.
Global average pooling and dense at the end.

conv layers are +2 of the dim they are labeled(1d -> x by t by batch)
so I had to double batch the dataset.


The ratings (labels/targets) are assigned for each set of steps (like a chunk of the time to slide a window over).
I set the chunk to be some number of seconds(1), but I needed to convert that to frames since each row of data is a frame and div by how often a frame is recorded.
This left steps = 1 * 60(fps)/3(every frame polled)
I then converted the ratings from 1-5 to a one_hot_encoding.

I cannot figure out why, but logits shape(network output) is always [batch size, classes] which makes sense.
But the labels are instead [batch size*classes] for some reason.
I'm batching the labels and inputs the same, I don't understand, sending an email to memo.
So, it's the loss function, using categorical_crossentropy instead, IDK.

Getting images to work wasn't too bad, I just had to up the dimensions for the images(1d to 3d).

I'm combining the numeric and image models with a concatenate layer and then three dense layers.

## Training model
Set the train-test split to 0.8, wow that was bad, it was overwriting like hell.
At 0.5 it's significantly better, 0.54 ish val_acc, IE it's more than 50% right where guessing would come up 20%.
Not awful, could be better.

For images, I need to reduce the period and batches and resolution I don't have enough RAM.
My ram limit is about 45g anything more, and the model won't load into VRAM.
The Numeric model is about a gig, and the image is about 39 when the period is 3,
batch-size is 2, and the image scale is 1/4.
Numeric gets 0.55 ish accuracy with these settings.

WHY DO I NOT HAVE CUDA!?
I didn't notice at first since the numeric data was easy, but with images I need cuda working.
Updated nvidia drivers to 550, re-installed cuda stuff,
then forced the script run with the CUDNN_PATH and LD_LIBRARY_PATH envvars.

Got 30% accuracy on a full run, not great... gonna shuffle, make a bigger split(0.8), and make period larger(5s)
holy! 80% accuracy!

for full: loss: 0.4411 - categorical_accuracy: 0.8131
numeric: loss: 1.6717 - categorical_accuracy: 0.4393, categorical_accuracy: 0.5048 - loss: 3.7492
image: loss: 1.3780 - categorical_accuracy: 0.4911, categorical_accuracy: 0.5695 - loss: 1.2036

# For ML4DS

Adding LSTM layers with 64 units to the imaage and numeric portions of the full model.
Removing datset shuffle on train.
Setting period to like 3

For numeric: LSTM must be before GlobalAveragePooling since that reduces dims, replacing GAP with LSTM
For images: LSTM needs more dims so i'm using a ConvLSTM2D to convolve over images, over time
~~For images some gap is needed since Conv3D layers return a 5d tensor and I need a 3d, so I'll use some gap here.~~
Still GAPing after the LSTM

Starts with really high accuracy, then dies down.
Since I load multiple recordings and there's no shuffling, the accuracy loss seems to correspond with when it encounters a new reocrding and therefore a new environment.
Dying a quarter of the way through each epoch, five recordings with 80% train, test split, hmmm

It immeadiately does better with one application but it's stagnating pretty hard and, probably because it's overfitting really hard.

trying with a large period, train acc is better already.

## New stats
full: categorical_accuracy: 0.9803 - loss: 0.1689(deadass forgot to set last few dense layers to relu)