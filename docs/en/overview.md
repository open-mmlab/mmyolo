# Overview

This chapter introduces you to the overall framework of MMYOLO and provides links to detailed tutorials.

## What is  MMYOLO

![pic](https://user-images.githubusercontent.com/12907710/137271636-56ba1cd2-b110-4812-8221-b4c120320aa9.png)

MMYOLO is a YOLO series algorithm toolbox, which currently implements only the target detection task and will subsequently support various tasks such as instance segmentation, panoramic segmentation and key point detection. It includes a rich set of target detection algorithms and related components and modules, and the following is its overall framework.

MMYOLO file structure is identical to the MMDetection. To allow full reuse of the MMDetection code, MMYOLO includes only custom content, which consists of 3 main parts:datasets、models、engine.

- **datasets** supports a variety of data sets for target detection
  - **transforms** includes various data enhancement transforms
- **models** is the most important part of the detector, which include different component of it.
  - **detectors** define all detection model classes
  - **data_preprocessors** is used to preprocess the dataset of the model
  - **backbones** include various backbone networks
  - **necks** include various neck components
  - **dense_heads** include various dense head of different tasks
  - **losses** include various loss functions
  - **task_modules** provide component for testing tasks,such as assigners、samplers、box coders and prior generators。
  - **layers** provide some basic network layers
- **engine** is a component of running
  - **optimizers** provide optimizers and packages for optimizers
  - **hooks** provide hooks for runner

## How to use this tutorial

The detailed instruction of MMYOLO is as following

1. Look up install instruction through [start your first step](get_started.md)

2. Basic method of how to use MMYOLO can be found here:

   - [Training and testing](https://mmyolo.readthedocs.io/en/latest/user_guides/index.html#train-test)
   - [From getting started to deployment tutorial](https://mmyolo.readthedocs.io/en/latest/user_guides/index.html#from-getting-started-to-deployment-tutorial)
   - [Useful Tools](https://mmyolo.readthedocs.io/en/latest/user_guides/index.html#useful-tools)

3. YOLO series of tutorials on algorithm implementation and full analysis:

   - [Essential Basics](https://mmyolo.readthedocs.io/en/latest/algorithm_descriptions/index.html#essential-basics)
   - [Full explanation of the model and implementation](https://mmyolo.readthedocs.io/en/latest/algorithm_descriptions/index.html#algorithm-principles-and-implementation)

4. Refer to the following tutorials for an in-depth look:

   - [How to](https://mmyolo.readthedocs.io/en/latest/advanced_guides/index.html#how-to)
