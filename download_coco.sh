# Dataset origin
cp -r /dataset/chenjn/dataset/coco ./
# tar -xvf coco/annotations.tar.gz -C coco/
ln -s /dataset/houbowei/coco/annotations ./coco/annotations
tar -xvf coco/train2017.tar.gz -C coco/
tar -xvf coco/val2017.tar.gz -C coco/
rm coco/*.gz