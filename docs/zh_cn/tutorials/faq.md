# 常见问题解答

我们在这里列出了使用时的一些常见问题及其相应的解决方案。 如果您发现有一些问题被遗漏，请随时提 PR 丰富这个列表。 如果您无法在此获得帮助，请创建 [issue](https://github.com/open-mmlab/mmyolo/issues/new/choose) 提问，但是请在模板中填写所有必填信息，这有助于我们更快定位问题。

## 为什么要推出 MMYOLO？

为什么要推出 MMYOLO？ 为何要单独开一个仓库而不是直接放到 MMDetection 中？ 自从开源后，不断收到社区小伙伴们类似的疑问，答案可以归纳为以下三点：

**(1) 统一运行和推理平台**

目前目标检测领域出现了非常多 YOLO 的改进算法，并且非常受大家欢迎，但是这类算法基于不同框架不同后端实现，存在较大差异，缺少统一便捷的从训练到部署的公平评测流程。

**(2) 协议限制**

众所周知，YOLOv5 以及其衍生的 YOLOv6 和 YOLOv7 等算法都是 GPL 3.0 协议，不同于 MMDetection 的 Apache 协议。由于协议问题，无法将 MMYOLO 直接并入 MMDetection 中。

**(3) 多任务支持**

还有一层深远的原因： **MMYOLO 任务不局限于 MMDetection**，后续会支持更多任务例如基于 MMPose 实现关键点相关的应用，基于 MMTracking 实现追踪相关的应用，因此不太适合直接并入 MMDetection 中。

## YOLOv5 backbone 替换为 Swin 后效果很差

## MM 系列开源库中有很多组件，如何在 MMYOLO 中使用？

## MMYOLO 中是否可以加入纯背景图片进行训练？

## MMYOLO 是否有计算模型推理 FPS 脚本？

## MMDeploy 和 EasyDeploy 有啥区别？

## 如果指定某张 GPU 进行训练或测试？

## COCOMetric 中如何查看每个类的 AP

## MMYOLO 中为何没有支持 MMDet 类似的自动学习率缩放功能？

## 如何进行分布式训练或者测试

## 自己训练的模型权重尺寸为啥比官方发布的大？

## RTMDet 为何训练所占显存比 YOLOv5 多很多？
