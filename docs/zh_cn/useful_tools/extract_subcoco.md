# 提取 COCO 子集

COCO2017 数据集训练数据集包括 118K 张图片，验证集包括 5K 张图片，数据集比较大。在调试或者快速验证程序是否正确的场景下加载 json 会需要消耗较多资源和带来较慢的启动速度，这会导致程序体验不好。

`extract_subcoco.py` 脚本提供了按指定图片数量、类别、锚框尺寸来切分图片的功能，用户可以通过 `--num-img`, `--classes`, `--area-size` 参数来得到指定条件的 COCO 子集，从而满足上述需求。

例如通过以下脚本切分图片：

```shell
python tools/misc/extract_subcoco.py \
    ${ROOT} \
    ${OUT_DIR} \
    --num-img 20 \
    --classes cat dog person \
    --area-size small
```

会切分出 20 张图片，且这 20 张图片只会保留同时满足类别条件和锚框尺寸条件的标注信息, 没有满足条件的标注信息的图片不会被选择，保证了这 20 张图都是有 annotation info 的。

注意： 本脚本目前仅仅支持 COCO2017 数据集，未来会支持更加通用的 COCO JSON 格式数据集

输入 root 根路径文件夹格式如下所示：

```text
├── root
│   ├── annotations
│   ├── train2017
│   ├── val2017
│   ├── test2017
```

1. 仅仅使用 5K 张验证集切分出 10 张训练图片和 10 张验证图片

```shell
python tools/misc/extract_subcoco.py ${ROOT} ${OUT_DIR} --num-img 10
```

2. 使用训练集切分出 20 张训练图片，使用验证集切分出 20 张验证图片

```shell
python tools/misc/extract_subcoco.py ${ROOT} ${OUT_DIR} --num-img 20 --use-training-set
```

3. 设置全局种子，默认不设置

```shell
python tools/misc/extract_subcoco.py ${ROOT} ${OUT_DIR} --num-img 20 --use-training-set --seed 1
```

4. 按指定类别切分图片

```shell
python tools/misc/extract_subcoco.py ${ROOT} ${OUT_DIR} --classes cat dog person
```

5. 按指定锚框尺寸切分图片

```shell
python tools/misc/extract_subcoco.py ${ROOT} ${OUT_DIR} --area-size small
```
