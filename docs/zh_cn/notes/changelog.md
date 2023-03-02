# 更新日志

## v0.5.0 (2/3/2023)

### 亮点

1. 支持了 [RTMDet-R](https://github.com/open-mmlab/mmyolo/blob/dev/configs/rtmdet/README.md#rotated-object-detection) 旋转框目标检测任务和算法
2. [YOLOv8](https://github.com/open-mmlab/mmyolo/blob/dev/configs/yolov8/README.md) 支持使用 mask 标注提升目标检测模型性能
3. 支持 [MMRazor](https://github.com/open-mmlab/mmyolo/blob/dev/configs/razor/subnets/README.md) 搜索的 NAS 子网络作为 YOLO 系列算法的 backbone
4. 支持调用 [MMRazor](https://github.com/open-mmlab/mmyolo/blob/dev/configs/rtmdet/distillation/README.md) 对 RTMDet 进行知识蒸馏
5. [MMYOLO](https://mmyolo.readthedocs.io/zh_CN/dev/) 文档结构优化，内容全面升级
6. 基于 RTMDet 训练超参提升 YOLOX 精度和训练速度
7. 支持模型参数量、FLOPs 计算和提供 T4 设备上 GPU 延时数据，并更新了 [Model Zoo](https://github.com/open-mmlab/mmyolo/blob/dev/docs/zh_cn/model_zoo.md)
8. 支持测试时增强 TTA
9. 支持 RTMDet、YOLOv8 和 YOLOv7 assigner 可视化

### 新特性

01. 支持 RTMDet 实例分割任务的推理 (#583)
02. 美化 MMYOLO 中配置文件并增加更多注释 (#501, #506, #516, #529, #531, #539)
03. 重构并优化中英文文档 (#568, #573, #579, #584, #587, #589, #596, #599, #600)
04. 支持 fast 版本的 YOLOX (#518)
05. EasyDeploy 中支持 DeepStream，并添加说明文档 (#485, #545, #571)
06. 新增混淆矩阵绘制脚本 (#572)
07. 新增单通道应用案例 (#460)
08. 支持 auto registration (#597)
09. Box CAM 支持 YOLOv7、YOLOv8 和 PPYOLOE (#601)
10. 新增自动化生成 MM 系列 repo 注册信息和 tools 脚本 (#559)
11. 新增 YOLOv7 模型结构图 (#504)
12. 新增如何指定特定 GPU 训练和推理文档 (#503)
13. 新增训练或者测试时检查 `metainfo` 是否全为小写 (#535)
14. 增加 Twitter、Discord、Medium 和 YouTube 等链接 (#555)

### Bug 修复

1. 修复 isort 版本问题 (#492, #497)
2. 修复 assigner 可视化模块的 type 错误 (#509)
3. 修复 YOLOv8 文档链接错误 (#517)
4. 修复 EasyDeploy 中的 RTMDet Decoder 错误 (#519)
5. 修复一些文档链接错误 (#537)
6. 修复 RTMDet-Tiny 权重路径错误 (#580)

### 完善

1. 完善更新 `contributing.md`
2. 优化 `DetDataPreprocessor` 支使其支持多任务 (#511)
3. 优化 `gt_instances_preprocess` 使其可以用于其他 YOLO 算法 (#532)
4. 新增 `yolov7-e6e` 权重转换脚本 (#570)
5. 参考 YOLOv8 推理代码修改 PPYOLOE (#614)

### 贡献者

总共 22 位开发者参与了本次版本

@triple-Mu, @isLinXu, @Audrey528, @TianWen580, @yechenzhi, @RangeKing, @lyviva, @Nioolek, @PeterH0323, @tianleiSHI, @aptsunny, @satuoqaq, @vansin, @xin-li-67, @VoyagerXvoyagerx,
@landhill, @kitecats, @tang576225574, @HIT-cwh, @AI-Tianlong, @RangiLyu, @hhaAndroid

## v0.4.0 (18/1/2023)

### 亮点

1. 实现了 [YOLOv8](https://github.com/open-mmlab/mmyolo/blob/dev/configs/yolov8/README.md) 目标检测模型，并通过 [projects/easydeploy](https://github.com/open-mmlab/mmyolo/blob/dev/projects/easydeploy) 支持了模型部署
2. 新增了中英文版本的 [YOLOv8 原理和实现全解析文档](https://github.com/open-mmlab/mmyolo/blob/dev/docs/zh_cn/algorithm_descriptions/yolov8_description.md)

### 新特性

1. 新增 YOLOv8 和 PPYOLOE 模型结构图 (#459, #471)
2. 调整最低支持 Python 版本从 3.6 升级为 3.7 (#449)
3. TensorRT-8 中新增新的 YOLOX decoder 写法 (#450)
4. 新增学习率可视化曲线脚本 (#479)
5. 新增脚本命令速查表 (#481)

### Bug 修复

1. 修复 `optimize_anchors.py` 脚本导入错误问题 (#452)
2. 修复 `get_started.md` 中安装步骤错误问题 (#474)
3. 修复使用 `RTMDet` P6 模型时候 neck 报错问题 (#480)

### 视频

1. 发布了 [玩转 MMYOLO 之实用篇（四）：顶会第一步 · 模块自定义](https://www.bilibili.com/video/BV1yd4y1j7VD/)

### 贡献者

总共 9 位开发者参与了本次版本

谢谢 @VoyagerXvoyagerx, @tianleiSHI, @RangeKing, @PeterH0323, @Nioolek, @triple-Mu, @lyviva, @Zheng-LinXiao, @hhaAndroid

## v0.3.0 (8/1/2023)

### 亮点

1. 实现了 [RTMDet](https://github.com/open-mmlab/mmyolo/blob/dev/configs/rtmdet/README.md) 的快速版本。RTMDet-s 8xA100 训练只需要 14 个小时，训练速度相比原先版本提升 2.6 倍。
2. 支持 [PPYOLOE](https://github.com/open-mmlab/mmyolo/blob/dev/configs/ppyoloe/README.md) 训练。
3. 支持 [YOLOv5](https://github.com/open-mmlab/mmyolo/blob/dev/configs/yolov5/crowdhuman/yolov5_s-v61_8xb16-300e_ignore_crowdhuman.py) 的 `iscrowd` 属性训练。
4. 支持 [YOLOv5 正样本分配结果可视化](https://github.com/open-mmlab/mmyolo/blob/dev/projects/assigner_visualization/README.md)
5. 新增 [YOLOv6 原理和实现全解析文档](https://github.com/open-mmlab/mmyolo/blob/dev/docs/zh_cn/algorithm_descriptions/yolov6_description.md)

### 新特性

01. 新增 `crowdhuman` 数据集 (#368)
02. EasyDeploy 中支持 TensorRT 推理 (#377)
03. 新增 `YOLOX` 结构图描述 (#402)
04. 新增视频推理脚本 (#392)
05. EasyDeploy 中支持 `YOLOv7` 部署 (#427)
06. 支持从 CLI 中的特定检查点恢复训练 (#393)
07. 将元信息字段设置为小写（#362、#412）
08. 新增模块组合文档 (#349, #352, #345)
09. 新增关于如何冻结 backbone 或 neck 权重的文档 (#418)
10. 在 `how_to.md` 中添加不使用预训练权重的文档 (#404)
11. 新增关于如何设置随机种子的文档 (#386)
12. 将 `rtmdet_description.md` 文档翻译成英文 (#353)

### Bug 修复

01. 修复设置 `--class-id-txt` 时输出注释文件中的错误 (#430)
02. 修复 `YOLOv5` head 中的批量推理错误 (#413)
03. 修复某些 head 的类型提示（#415、#416、#443）
04. 修复 expected a non-empty list of Tensors 错误 (#376)
05. 修复 `YOLOv7` 训练中的设备不一致错误（#397）
06. 修复 `LetterResize` 中的 `scale_factor` 和 `pad_param` 值 (#387)
07. 修复 readthedocs 的 docstring 图形渲染错误 (#400)
08. 修复 `YOLOv6` 从训练到验证时的断言错误 (#378)
09. 修复 `np.int` 和旧版 builder.py 导致的 CI 错误 (#389)
10. 修复 MMDeploy 重写器 (#366)
11. 修复 MMYOLO 单元测试错误 (#351)
12. 修复 `pad_param` 错误 (#354)
13. 修复 head 推理两次的错误（#342）
14. 修复自定义数据集训练 (#428)

### 完善

01. 更新 `useful_tools.md` (#384)
02. 更新英文版 `custom_dataset.md` (#381)
03. 重写函数删除上下文参数 (#395)
04. 弃用 `np.bool` 类型别名 (#396)
05. 为自定义数据集添加新的视频链接 (#365)
06. 仅为模型导出 onnx (#361)
07. 添加 MMYOLO 回归测试 yml (#359)
08. 更新 `article.md` 中的视频教程 (#350)
09. 添加部署 demo (#343)
10. 优化 debug 模式下大图的可视化效果(#346)
11. 改进 `browse_dataset` 的参数并支持 `RepeatDataset` (#340, #338)

### 视频

1. 发布了 [基于 sahi 的大图推理](https://www.bilibili.com/video/BV1EK411R7Ws/)
2. 发布了 [自定义数据集从标注到部署保姆级教程](https://www.bilibili.com/video/BV1RG4y137i5)

### 贡献者

总共 28 位开发者参与了本次版本

谢谢 @RangeKing, @PeterH0323, @Nioolek, @triple-Mu, @matrixgame2018, @xin-li-67, @tang576225574, @kitecats, @Seperendity, @diplomatist, @vaew, @wzr-skn, @VoyagerXvoyagerx, @MambaWong, @tianleiSHI, @caj-github, @zhubochao, @lvhan028, @dsghaonan, @lyviva, @yuewangg, @wang-tf, @satuoqaq, @grimoire, @RunningLeon, @hanrui1sensetime, @RangiLyu, @hhaAndroid

## v0.2.0（1/12/2022)

### 亮点

1. 支持 [YOLOv7](https://github.com/open-mmlab/mmyolo/tree/dev/configs/yolov7) P5 和 P6 模型
2. 支持 [YOLOv6](https://github.com/open-mmlab/mmyolo/blob/dev/configs/yolov6/README.md) 中的 ML 大模型
3. 支持 [Grad-Based CAM 和 Grad-Free CAM](https://github.com/open-mmlab/mmyolo/blob/dev/demo/boxam_vis_demo.py)
4. 基于 sahi 支持 [大图推理](https://github.com/open-mmlab/mmyolo/blob/dev/demo/large_image_demo.py)
5. projects 文件夹下新增 [easydeploy](https://github.com/open-mmlab/mmyolo/blob/dev/projects/easydeploy/README.md) 项目
6. 新增 [自定义数据集教程](https://github.com/open-mmlab/mmyolo/blob/dev/docs/zh_cn/user_guides/custom_dataset.md)

### 新特性

1. `browse_dataset.py` 脚本支持可视化原图、数据增强后和中间结果功能 (#304)
2. `image_demo.py` 新增预测结果保存为 labelme 格式功能 (#288, #314)
3. 新增 labelme 格式转 COCO 格式脚本 `labelme2coco` (#308, #313)
4. 新增 COCO 数据集切分脚本 `coco_split.py` (#311)
5. `how-to.md` 文档中新增两个 backbone 替换案例以及更新 `plugin.md` (#291)
6. 新增贡献者文档 `contributing.md` and 代码规范文档 `code_style.md` (#322)
7. 新增如何通过 mim 跨库调用脚本文档 (#321)
8. `YOLOv5` 支持 RV1126 设备部署 (#262)

### Bug 修复

1. 修复 `MixUp` padding 错误 (#319)
2. 修复 `LetterResize` 和 `YOLOv5KeepRatioResize` 中 `scale_factor` 参数顺序错误 (#305)
3. 修复 `YOLOX Nano` 模型训练错误问题 (#285)
4. 修复 `RTMDet` 部署没有导包的错误 (#287)
5. 修复 int8 部署配置错误 (#315)
6. 修复 `basebackbone` 中 `make_stage_plugins` 注释 (#296)
7. 部署模块支持切换为 deploy 模式功能 (#324)
8. 修正 `RTMDet` 模型结构图中的错误 (#317)

### 完善

1. `test.py` 中新增 json 格式导出选项 (#316)
2. `extract_subcoco.py` 脚本中新增基于面积阈值过滤规则 (#286)
3. 部署相关中文文档翻译为英文 (#289)
4. 新增 `YOLOv6` 算法描述大纲文档 (#252)
5. 完善 `config.md` (#297, #303)
6. 完善 `mosiac9` 的 docstring (#307)
7. 完善 `browse_coco_json.py` 脚本输入参数 (#309)
8. 重构 `dataset_analysis.py` 中部分函数使其更加通用 (#294)

### 视频

1. 发布了 [工程文件结构简析](https://www.bilibili.com/video/BV1LP4y117jS)
2. 发布了 [10分钟换遍主干网络文档](https://www.bilibili.com/video/BV1JG4y1d7GC)

### 贡献者

总共 14 位开发者参与了本次版本

谢谢 @fcakyon, @matrixgame2018, @MambaWong, @imAzhou, @triple-Mu, @RangeKing, @PeterH0323, @xin-li-67, @kitecats, @hanrui1sensetime, @AllentDan, @Zheng-LinXiao, @hhaAndroid, @wanghonglie

## v0.1.3（10/11/2022)

### 新特性

1. 支持 CBAM 插件并提供插件文档 (#246)
2. 新增 YOLOv5 P6 模型结构图和相关说明 (#273)

### Bug 修复

1. 基于 mmengine 0.3.1 修复保存最好权重时训练失败问题
2. 基于 mmdet 3.0.0rc3 修复 `add_dump_metric` 报错 (#253)
3. 修复 backbone 不支持 `init_cfg` 问题 (#272)
4. 基于 mmdet 3.0.0rc3 改变 typing 导入方式 (#261)

### 完善

1. `featmap_vis_demo` 支持文件夹和 url 输入 (#248)
2. 部署 docker 文件完善 (#242)

### 贡献者

总共 10 位开发者参与了本次版本

谢谢 @kitecats, @triple-Mu, @RangeKing, @PeterH0323, @Zheng-LinXiao, @tkhe, @weikai520, @zytx121, @wanghonglie, @hhaAndroid

## v0.1.2（3/11/2022)

### 亮点

1. 支持 ONNXRuntime 和 TensorRT 的 [YOLOv5/YOLOv6/YOLOX/RTMDet 部署](https://github.com/open-mmlab/mmyolo/blob/main/configs/deploy)
2. 支持 [YOLOv6](https://github.com/open-mmlab/mmyolo/blob/main/configs/yolov6) s/t/n 模型训练
3. YOLOv5 支持 [P6 大分辨率 1280 尺度训练](https://github.com/open-mmlab/mmyolo/blob/main/configs/yolov5)
4. YOLOv5 支持 [VOC 数据集训练](https://github.com/open-mmlab/mmyolo/blob/main/configs/yolov5/voc)
5. 支持 [PPYOLOE](https://github.com/open-mmlab/mmyolo/blob/main/configs/ppyoloe) 和 [YOLOv7](https://github.com/open-mmlab/mmyolo/blob/main/configs/yolov7) 模型推理和官方权重转化
6. How-to 文档中新增 YOLOv5 替换 [backbone 教程](https://github.com/open-mmlab/mmyolo/blob/dev/docs/zh_cn/advanced_guides/how_to.md#%E8%B7%A8%E5%BA%93%E4%BD%BF%E7%94%A8%E4%B8%BB%E5%B9%B2%E7%BD%91%E7%BB%9C)

### 新特性

1. 新增 `optimize_anchors` 脚本 (#175)
2. 新增 `extract_subcoco` 脚本 (#186)
3. 新增 `yolo2coco` 转换脚本 (#161)
4. 新增 `dataset_analysis` 脚本 (#172)
5. 移除 Albu 版本限制 (#187)

### Bug 修复

1. 修复当设置 `cfg.resume` 时候不生效问题 (#221)
2. 修复特征图可视化脚本中不显示 bbox 问题 (#204)
3. 更新 RTMDet 的 metafile (#188)
4. 修复 test_pipeline 中的可视化错误 (#166)
5. 更新 badges (#140)

### 完善

1. 优化 Readthedoc 显示页面 (#209)
2. 为 base model 添加模块结构图的 docstring (#196)
3. 支持 LoadAnnotations 中不包括任何实例逻辑 (#161)
4. 更新 `image_demo` 脚本以支持文件夹和 url 路径 (#128)
5. 更新 pre-commit hook (#129)

### 文档

1. 将 `yolov5_description.md`、 `yolov5_tutorial.md` 和 `visualization.md` 翻译为英文 (#138, #198, #206)
2. 新增部署相关中文文档 (#220)
3. 更新 `config.md`、`faq.md` 和 `pull_request_template.md` (#190, #191, #200)
4. 更新 `article` 页面 (#133)

### 视频

1. 发布了[特征图可视化视频](https://www.bilibili.com/video/BV188411s7o8)
2. 发布了 [YOLOv5 配置文件解读视频](https://www.bilibili.com/video/BV1214y157ck)
3. 发布了 [RTMDet-s 特征图可视化 demo 视频](https://www.bilibili.com/video/BV1je4y1478R)
4. 发布了[源码解读和必备调试技巧视频](https://www.bilibili.com/video/BV1N14y1V7mB)

### 贡献者

总共 14 位开发者参与了本次版本

谢谢 @imAzhou, @triple-Mu, @RangeKing, @PeterH0323, @xin-li-67, @Nioolek, @kitecats, @Bin-ze, @JiayuXu0, @cydiachen, @zhiqwang, @Zheng-LinXiao, @hhaAndroid, @wanghonglie

## v0.1.1（29/9/2022)

基于 MMDetection 的 RTMDet 高精度低延时目标检测算法，我们也同步发布了 RTMDet，并提供了 RTMDet 原理和实现全解析中文文档

### 亮点

1. 支持了 [RTMDet](https://github.com/open-mmlab/mmyolo/blob/main/configs/rtmdet)
2. 新增了 [RTMDet 原理和实现全解析中文文档](https://github.com/open-mmlab/mmyolo/blob/main/docs/zh_cn/algorithm_descriptions/rtmdet_description.md)
3. 支持对 backbone 自定义插件，并更新了 How-to 文档 (#75)

### Bug 修复

1. 修复一些文档错误 (#66, #72, #76, #83, #86)
2. 修复权重链接错误 (#63)
3. 修复 `LetterResize` 使用 `imscale` api 时候输出不符合预期的 bug (#105)

### 完善

1. 缩减 docker 镜像尺寸 (#67)
2. 简化 BaseMixImageTransform 中 Compose 逻辑 (#71)
3. test 脚本支持 dump 结果 (#84)

#### 贡献者

总共 13 位开发者参与了本次版本

谢谢 @wanghonglie, @hhaAndroid, @yang-0201, @PeterH0323, @RangeKing, @satuoqaq, @Zheng-LinXiao, @xin-li-67, @suibe-qingtian, @MambaWong, @MichaelCai0912, @rimoire, @Nioolek

## v0.1.0（21/9/2022)

我们发布了 MMYOLO 开源库，其基于 MMEngine, MMCV 2.x 和 MMDetection 3.x 库. 目前实现了目标检测功能，后续会扩展为多任务。

### 亮点

1. 支持 YOLOv5/YOLOX 训练，支持 YOLOv6 推理。部署即将支持。
2. 重构了 MMDetection 的 YOLOX，提供了更快的训练和推理速度。
3. 提供了详细入门和进阶教程, 包括 YOLOv5 从入门到部署、YOLOv5 算法原理和实现全解析、 特征图可视化等教程。
