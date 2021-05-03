# Manual Classification GUI Guide

This app aims to improve the efficiency for manual crack labeling.

## How to run?

**1.** Firstly, make sure your Python 3 environment has installed `dearpygui`, `PIL` and `tensorflow`.

If not, you can install by using `pip`.

```shell
pip install dearpygui
pip install PIL
pip install tenserflow
```

**2.** Then, create a new folder and move all images that need to be labled to that folder. For example, move all images to `/Users/root/Desktop/cracks`.

**3.** Then, run in termial using the following command.

```shell
<your-python3-interpretor-path> <path-to-app.py> <full-path-to-image-folder> <number-of-cracks>
```

For example,

```shell
cd Classification\ GUI 
python3 app.py /Users/root/Desktop/cracks 30
```

**4.** The parameter `<number-of-cracks>` defines how many cracks to label for current schedule. Suppse there are 100 images in total in `/Users/root/Desktop/cracks`, and you wish to label all images within 3 times (e.g.,  30, 30, 40 images for each schedule), you just need to run the following command .

```
# Schedule 1
python3 app.py /Users/root/Desktop/cracks 30
# Schedule 2
python3 app.py /Users/root/Desktop/cracks 30
# Schedule 3
python3 app.py /Users/root/Desktop/cracks 40
```

## How it works?

**1.** The app will save all lables in a `csv` file with name `labels.csv` in sub-folder `save`. The `csv` file has the following format.

 ```txt
 image_name,crack_1,crack_2,crack_3
 00001.JPG,long,lat,none
 00002.JPG,long,long,croc
 00003.JPG,long,lat,croc
 ...
 ```

**2.** Each time you run this app, the system will load this `csv` file and determine from which image to start labelling. Don't worry if your input `<number-of-cracks>` is more than the unlabelled images in the folder, the system will handle it. For example, for 100 images in the labels, it is perfectly fine to run the following command.

```
# Schedule 1
python3 app.py /Users/root/Desktop/cracks 30
# Schedule 2
python3 app.py /Users/root/Desktop/cracks 30
# Schedule 3
python3 app.py /Users/root/Desktop/cracks 50
```

**3.** In the app, you have 3 opearion choices.

> **a.** `LOAD NEXT`: To load the image in the system, the system will split the image into 3 smaller images.
>
> **b**. `SHOW NEXT`: To show the next smaller image. For every split image, you need to manually click this button to show it. When all the 3 smaller images haven been shown, you need to click `LOAD NEXT` to load next image.
>
> **c.** Select crack type. You have to select one of the four types: `LONG`, `LAT`, `CROC` and `NONE`. Each split image need to be classified once.

The image of this pannel is shown below. 

![panel](panel.png)

## What if I jumped over one image?

**1.** Don't worry, the system will not allow you to do so. You will have to label one image before showing the next split image and you will have to label all 3 smaller images before loading the next image. All operations violating the rules will be prevented by the system.

**2.** The window title will also give you some tips for what you should do next and will show your current operation. (It is indeed annoying and will make you dizzy when labelling hunderds of images)

## What if I mis-labeled one image?

Sorry, you can't go back and re-label. However, you can open the `labels.csv` file and re-label it anytime during the process.

## What if the app crashed during the labeling process?

The system will save the result every time you click a crack botton. If app crashed during the labeling process, check if the `labels.csv` file ends with a complete entry or not. For example:

```txt
# OK
00001.JPG,long,lat,none
# NOT OK
00001.JPG,long
```

If the last line is complete, you don't have to do anything but start you next schedule of labelling. If not, delete the last line **together with the newline `\n` character for the last second line** , then start you next schedule of labelling.