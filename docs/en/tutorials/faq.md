# Frequently Asked Questions

We list some common problems many users face and their corresponding solutions here. Feel free to enrich the list if you find any frequent issues and have ways to help others to solve them. If the contents here do not cover your issue, please create an [issue](https://github.com/open-mmlab/mmyolo/issues/new/choose) and make sure you fill in all the required information in the template.

## Why do we need to launch MMYOLO?

Why do we need to launch MMYOLO? Why do we need to open a separate repository instead of putting it directly into MMDetection? Since the open source, we have been receiving similar questions from our community partners, and the answers can be summarized in the following three points.

**(1) Unified operation and inference platform**

At present, there are very many improved algorithms for YOLO in the field of target detection, and they are very popular, but such algorithms are based on different frameworks for different back-end implementations, and there are large differences, lacking a unified and convenient fair evaluation process from training to deployment.

**(2) Protocol limitations**

As we all know, YOLOv5 and its derived algorithms such as YOLOv6 and YOLOv7 are GPL 3.0 protocols, which are different from the Apache protocol of MMDetection. Due to the protocol issue, it is not possible to incorporate MMYOLO directly into MMDetection.

**(3) Multitasking support**

There is another far-reaching reason: **MMYOLO tasks are not limited to MMDetection**, and more tasks will be supported in the future, such as MMPose based keypoint related applications and MMTracking based tracking related applications, so it is not suitable to be directly incorporated into MMDetection.
