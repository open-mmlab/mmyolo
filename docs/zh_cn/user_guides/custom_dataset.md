# 自定义数据集 训练+部署 全流程

本章节会介绍从 用户自己的图片数据集 到 最终进行训练 的整体流程。打标签使用的软件是 [labelme](https://github.com/wkentaro/labelme)
流程步骤概览如下：

1. 打标签
2. 使用脚本转换成 COCO 数据集格式
3. 数据集划分
4. 根据数据集内容新建 config 文件
5. 训练
6. 测试
7. 部署

下面详细介绍每一步。

## 1. 打标签

通常，打标签有2种方法：

- 辅助 + 人工打标签
- 纯人工打标签

## 1.1 辅助 + 人工打标签

辅助打标签的原理是用已有模型进行推理，将得出的推理信息保存为打标签软件的标签文件格式。

**Tips**：如果已有模型没有新数据集的类，可以先人工打100张左右的图片标签，训练个初始模型，然后再进行辅助打标签。

人工操作打标签软件加载生成好的标签文件，只需要检查每张图片的目标是否标准，以及是否有漏掉的目标。

【辅助 + 人工打标签】这种方式可以节省很多时间和精力，达到降本提速的目的。

下面会分别介绍其过程：

### 1.1.1 辅助打标签

MMYOLO 提供模型推理生成 labelme 格式标签文件的脚本 `demo/image_demo.py`，具体用法如下：

```shell
image_demo.py img \
              config \
              checkpoint
              [-h] \
              [--out-dir OUT_DIR] \
              [--device DEVICE] \
              [--show] \
              [--deploy] \
              [--score-thr SCORE_THR] \
              [--to-labelme]
```

其中：

- `img`： 图片的路径，支持文件夹、文件、URL；
- `config`：用到的模型 config 文件路径；
- `checkpoint`：用到的模型权重文件路径；
- `--out-dir`：推理结果输出到指定目录下，默认为 `./output`，当 `--show` 参数存在时，不保存检测结果；
- `--device`：使用的计算资源，包括 `CUDA`, `CPU` 等，默认为 `cuda:0`；
- `--show`：使用该参数表示在屏幕上显示检测结果，默认为 `False`；
- `--deploy`：是否切换成 deploy 模式；
- `--score-thr`：置信度阈值，默认为 `0.3`；
- `--to-labelme`：是否导出 `labelme` 格式的标签文件，不可以与 `--show` 参数同时存在

eg.

```shell
python demo/image_demo.py \
    /data/cat/images \
    configs/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py \
    work_dirs/yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth \
    --out-dir /data/cat/labels \
    --to-labelme
```

生成的标签文件会在 `--out-dir` 中:

```shell
.
└── $OUT_DIR
    ├── image1.json
    ├── image1.json
    └── ...
```

### 1.1.2 人工打标签

本教程使用的打标签软件是 [labelme](https://github.com/wkentaro/labelme)

- 安装 labelme

```shell
pip install labelme
```

- 启动 labelme

```shell
labelme ${图片文件夹路径（即上一步的图片文件夹）} \
        --output ${label文件所处的文件夹路径（即上一步的 --out-dir）} \
        --autosave \
        --nodata
```

eg.

```shell
labelme /data/cat/images --output /data/cat/labels --autosave --nodata
```

输入命令之后 labelme 就会启动，然后进行标签检查即可。

**注意：打标签的时候务必使用 `rectangle`**

## 1.2 人工打标签

步骤和 【1.1.2 人工打标签】 相同，只是这里是直接打标签，没有预先生成的标签。

## 2. 使用脚本转换成 COCO 数据集格式

### 2.1 使用脚本转换

MMYOLO 提供脚本将 labelme 的标签转换为 COCO 标签

```shell
python tools/dataset_converters/labelme2coco.py --img-dir ${图片文件夹路径} \
                                                --label-dir ${标签文件夹位置} \
                                                --out ${输出COC标签json路径}
```

### 2.2 检查转换的 COCO 标签

使用下面的命令可以将 COCO 的标签在图片上进行显示，这一步可以验证刚刚转换是否有问题：

```shell
python tools/analysis_tools/browse_coco_json.py --img-dir ${图片文件夹路径} \
                                                --ann-file ${COCO标签json路径}
```

关于 `tools/analysis_tools/browse_coco_json.py` 的更多用法请参考 [可视化 COCO 标签](useful_tools.md)。

## 3. 数据集划分

```shell
python tools/analysis_tools/browse_coco_json.py --json ${COCO标签json路径} \
                                                --out-dir ${划分标签json保存路径} \
                                                --ratio ${划分比例} \
                                                [--shuffle] \
                                                [--seed ${划分的随机种子}]
```

其中：

- `--ratio`：划分的比例，如果只设置了2个，则划分为 `trainval + test`，如果设置为 3 个，则划分为 `train + val + test`。支持两种格式：整数和小数：
  - 小数：划分为比例，但是**全部比例加起来必须等于 `1`**。例子： `--ratio 0.8 0.1 0.1` or `--ratio 0.8 0.2`
  - 整数：按比分进行划分，代码中会进行归一化之后划分数据集。例子： `--ratio 2 1 1`（代码里面会转换成 `0.5 0.25 0.25`） or `--ratio 3 1`（代码里面会转换成 `0.75 0.25`）
- `--shuffle`: 是否打乱数据集再进行划分；
- `--seed`：可以设定划分的随机种子，不设置的话自动生成随机种子。

## 4. 根据数据集内容新建 config 文件

- 修改 config data_root
- data_prefix
- 根据 classes.txt 配置 metainfo=metainfo,num_classes
- 根据自己的GPU情况，修改 train_batch_size_per_gpu， base_lr
- 推荐使用 nGPU x 4 的 train_num_workers
- 设置 pretrain 的文件路径 or URL

## 5. 训练

使用 YOLOv5s 作为例子：

```shell
python tools/train.py configs/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py
```

## 6. 推理查看

```shell
python demo/image_demo.py /data/cat_split/test \
                          configs/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py \
                          work_dirs/yolov5_s-v61_syncbn_fast_8xb16-300e_coco/last.pth \
                          --out-dir /data/cat_split/test_output
```

## 7. 部署
