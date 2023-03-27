# Common Warning Notes

The purpose of this document is to collect warning messages that users often find confusing, and provide explanations to facilitate understanding.

## xxx registry in mmyolo did not set import location

The warning message complete information is that The xxx registry in mmyolo did not set import location. Fallback to call `mmyolo.utils.register_all_modules` instead.

This warning means that a module was not set with an import location when importing it, making it impossible to determine its location. Therefore, `mmyolo.utils.register_all_modules` is automatically called to trigger the package import.
This warning belongs to the very low-level module warning in MMEngine, which may be difficult for users to understand, but it has no impact on the actual use and can be ignored directly.

## save_param_schedulers is true but self.param_schedulers is None

The following information is an example using the YOLOv5 algorithm. This is because the parameter scheduler strategy `YOLOv5ParamSchedulerHook` has been rewritten in YOLOv5, so the ParamScheduler designed in MMEngine is not used. However, `save_param_schedulers` is not set to False in the YOLOv5 configuration.

First of all, this warning has no impact on performance and resuming training. If users think this warning affects experience, you can set `default_hooks.checkpoint.save_param_scheduler` to False, or set `--cfg-options default_hooks.checkpoint.save_param_scheduler=False` when training via the command line.

## The loss_cls will be 0. This is a normal phenomenon.

This is related to specific algorithms. Taking YOLOv5 as an example, its classification loss only considers positive samples. If the number of classes is 1, then the classification loss and object loss are functionally redundant. Therefore, in the design, when the number of classes is 1, the loss_cls is not calculated and is always 0. This is a normal phenomenon.

## The model and loaded state dict do not match exactly

Whether this warning will affect performance needs to be determined based on more information. If it occurs during fine-tuning, it is a normal phenomenon that the COCO pre-trained weights of the Head module cannot be loaded due to the user's custom class differences, and it will not affect performance.
