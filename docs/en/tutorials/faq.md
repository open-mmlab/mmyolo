# Frequently Asked Questions

We list some common problems many users face and their corresponding solutions here. Feel free to enrich the list if you find any frequent issues and have ways to help others to solve them. If the contents here do not cover your issue, please create an [issue](https://github.com/open-mmlab/mmyolo/issues/new/choose) and make sure you fill in all the required information in the template.

## Why do we need to launch MMYOLO?

Why do we need to launch MMYOLO? Why do we need to open a separate repository instead of putting it directly into MMDetection? Since the open source, we have been receiving similar questions from our community partners, and the answers can be summarized in the following three points.

**(1) Unified operation and inference platform**

At present, there are very many improved algorithms for YOLO in the field of target detection, and they are very popular, but such algorithms are based on different frameworks for different back-end implementations, and there are significant differences, lacking a unified and convenient fair evaluation process from training to deployment.

**(2) Protocol limitations**

As we all know, YOLOv5 and its derived algorithms, such as YOLOv6 and YOLOv7 are GPL 3.0 protocols, which differ from the Apache protocol of MMDetection. Therefore, due to the protocol issue, it is not possible to incorporate MMYOLO directly into MMDetection.

**(3) Multitasking support**

There is another far-reaching reason: **MMYOLO tasks are not limited to MMDetection**, and more tasks will be supported in the future, such as MMPose based keypoint-related applications and MMTracking based tracking related applications, so it is not suitable to be directly incorporated into MMDetection.

## What is the projects folder used for?

The `projects` folder is newly introduced in OpenMMLab 2.0. There are three primary purposes:

1. facilitate community contributors: Since OpenMMLab series codebases have a rigorous code management process, this inevitably leads to long algorithm reproduction cycles, which is not friendly to community contributions.
2. facilitate rapid support for new algorithms: A long development cycle can also lead to another problem users may not be able to experience the latest algorithms as soon as possible.
3. facilitate rapid support for new approaches and features: New approaches or new features may be incompatible with the current design of the codebases and cannot be quickly incorporated.

In summary, the `projects` folder solves the problems of slow support for new algorithms and complicated support for new features due to the long algorithm reproduction cycle. Each folder in `projects` is an entirely independent project, and community users can quickly support some algorithms in the current version through `projects`. This allows the community to quickly use new algorithms and features that are difficult to adapt in the current version. When the design is stable or the code meets the merge specification, it will be considered to merge into the main branch.

## Why does the performance drop significantly by switching the YOLOv5 backbone to Swin?

In [Replace the backbone network](../recommended_topics/replace_backbone.md), we provide many tutorials on replacing the backbone module. However, you may not get a desired result once you replace the module and start directly training the model. This is because different networks have very distinct hyperparameters. Take the backbones of Swin and YOLOv5 as an example. Swin belongs to the transformer family, and the YOLOv5 is a convolutional network. Their training optimizers, learning rates, and other hyperparameters are different. If we force using Swin as the backbone of YOLOv5 and try to get a moderate performance, we must modify many parameters.

## How to use the components implemented in all MM series repositories?

In OpenMMLab 2.0, we have enhanced the ability to use different modules across MM series libraries. Currently, users can call any module that has been registered in MM series algorithm libraries via `MM Algorithm Library A. Module Name`. We demonstrated using MMClassification backbones in the [Replace the backbone network](../recommended_topics/replace_backbone.md). Other modules can be used in the same way.

## Can pure background pictures be added in MMYOLO for training?

Adding pure background images to training can suppress the false positive rate in most scenarios, and this feature has already been supported for most datasets. Take `YOLOv5CocoDataset` as an example. The control parameter is `train_dataloader.dataset.filter_cfg.filter_empty_gt`. If `filter_empty_gt` is True, the pure background images will be filtered out and not used in training, and vice versa. Most of the algorithms in MMYOLO have added this feature by default.

## Is there a script to calculate the inference FPS in MMYOLO?

MMYOLO is based on MMDet 3.x, which provides a [benchmark script](https://github.com/open-mmlab/mmdetection/blob/3.x/tools/analysis_tools/benchmark.py) to calculate the inference FPS. We recommend using `mim` to run the script in MMDet directly across the library instead of copying them to MMYOLO. More details about `mim` usages can be found at [Use mim to run scripts from other OpenMMLab repositories](../common_usage/mim_usage.md).

## What is the difference between MMDeploy and EasyDeploy?

MMDeploy is developed and maintained by the OpenMMLab deployment team to provide model deployment solutions for the OpenMMLab series algorithms, which support various inference backends and customization features. EasyDeploy is an easier and more lightweight deployment project provided by the community. However, it does not support as many features as MMDeploy. Users can choose which one to use in MMYOLO according to their needs.

## How to check the AP of every category in COCOMetric?

Just set `test_evaluator.classwise` to True or add `--cfg-options test_evaluator.classwise=True` when running the test script.

## Why doesn't MMYOLO support the auto-learning rate scaling feature as MMDet?

It is because the YOLO series algorithms are not very well suited for linear scaling. We have verified on several datasets that the performance is better without the auto-scaling based on batch size.

## Why is the weight size of my trained model larger than the official one?

The reason is that user-trained weights usually include extra data such as `optimizer`, `ema_state_dict`, and `message_hub`, which are removed when we publish the models. While on the contrary, the weight users trained by themselves are kept. You can use the [publish_model.py](https://github.com/open-mmlab/mmyolo/blob/main/tools/misc/publish_model.py) to remove these unnecessary components.

## Why does the RTMDet cost more graphics memory during the training than YOLOv5?

It is due to the assigner in RTMDet. YOLOv5 uses a simple and efficient shape-matching assigner, while RTMDet uses a dynamic soft label assigner for entire batch computation. Therefore, it consumes more memory in its internal cost matrix, especially when there are too many labeled bboxes in the current batch. We are considering solving this problem soon.

## Do I need to reinstall MMYOLO after modifying some code?

Without adding any new python code, and if you installed the MMYOLO by `mim install -v -e .`, any new modifications will take effect without reinstalling. However, if you add new python codes and are using them, you need to reinstall with `mim install -v -e .`.

## How to use multiple versions of MMYOLO to develop?

If users have multiple versions of the MMYOLO, such as mmyolo-v1 and mmyolo-v2. They can specify the target version of their MMYOLO by using this command in the shell:

```shell
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
```

Users can unset the `PYTHONPATH` when they want to reset to the default MMYOLO by this command:

```shell
unset PYTHONPATH
```

## How to save the best checkpoints during the training?

Users can choose what metrics to filter the best models by setting the `default_hooks.checkpoint.save_best` in the configuration. Take the COCO dataset detection task as an example. Users can customize the `default_hooks.checkpoint.save_best` with these parameters:

1. `auto` works based on the first evaluation metric in the validation set.
2. `coco/bbox_mAP` works based on `bbox_mAP`.
3. `coco/bbox_mAP_50` works based on `bbox_mAP_50`.
4. `coco/bbox_mAP_75` works based on `bbox_mAP_75`.
5. `coco/bbox_mAP_s` works based on `bbox_mAP_s`.
6. `coco/bbox_mAP_m` works based on `bbox_mAP_m`.
7. `coco/bbox_mAP_l` works based on `bbox_mAP_l`.

In addition, users can also choose the filtering logic by setting `default_hooks.checkpoint.rule` in the configuration. For example, `default_hooks.checkpoint.rule=greater` means that the larger the indicator is, the better it is. More details can be found at [checkpoint_hook](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/checkpoint_hook.py).

## How to train and test with non-square input sizes?

The default configurations of the YOLO series algorithms are mostly squares like 640x640 or 1280x1280. However, if users want to train with a non-square shape, they can modify the `image_scale` to the desired value in the configuration. A more detailed example could be found at [yolov5_s-v61_fast_1xb12-40e_608x352_cat.py](https://github.com/open-mmlab/mmyolo/tree/dev/configs/yolov5/yolov5_s-v61_fast_1xb12-40e_608x352_cat.py).
