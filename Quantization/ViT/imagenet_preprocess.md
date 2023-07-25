## 下载

训练集（ILSVRC2012_img_train.tar）

```
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar --no-check-certificate
```

验证集（ILSVRC2012_img_val.tar）

```
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar --no-check-certificate
```

标签映射文件（ILSVRC2012_devkit_t12.tar.gz）

```
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz --no-check-certificate
```



## 数据集处理

### 1.解压

```
mkdir train
mkdir val
tar xvf ILSVRC2012_img_train.tar -C ./train
tar xvf ILSVRC2012_img_val.tar -C ./val
```

### 2. 二次解压train

```sh
# 利用脚本批量解压
dir=./train 
for x in `ls $dir/*tar` 
do     
  filename=`basename $x .tar`     
  mkdir $dir/$filename     
  tar -xvf $x -C $dir/$filename 
done 
rm $dir/*.tar
```

### 3. 修改val集映射标签

```python
from scipy import io
import os
import shutil

def move_valimg(val_dir='./val', devkit_dir='./ILSVRC2012_devkit_t12'):
    """
    move valimg to correspongding folders.
    val_id(start from 1) -> ILSVRC_ID(start from 1) -> WIND
    organize like:
    /val
       /n01440764
           images
       /n01443537
           images
        .....
    """
    # load synset, val ground truth and val images list
    synset = io.loadmat(os.path.join(devkit_dir, 'data', 'meta.mat'))
    
    ground_truth = open(os.path.join(devkit_dir, 'data', 'ILSVRC2012_validation_ground_truth.txt'))
    lines = ground_truth.readlines()
    labels = [int(line[:-1]) for line in lines]
    
    root, _, filenames = next(os.walk(val_dir))
    for filename in filenames:
        # val image name -> ILSVRC ID -> WIND
        val_id = int(filename.split('.')[0].split('_')[-1])
        ILSVRC_ID = labels[val_id-1]
        WIND = synset['synsets'][ILSVRC_ID-1][0][1][0]
        print("val_id:%d, ILSVRC_ID:%d, WIND:%s" % (val_id, ILSVRC_ID, WIND))

        # move val images
        output_dir = os.path.join(root, WIND)
        if os.path.isdir(output_dir):
            pass
        else:
            os.mkdir(output_dir)
        shutil.move(os.path.join(root, filename), os.path.join(output_dir, filename))

if __name__ == '__main__':
    move_valimg()
```

### 4. 加载数据集

```python
import os
import torch
import torchvision.datasets as datasets

root = '/vg_data/ice/dataset/imagenet/1k'
def get_imagenet(root, train = True, transform = None, target_transform = None):
    if train:
        root = os.path.join(root, 'train')
    else:
        root = os.path.join(root, 'val')
    return datasets.ImageFolder(root = root,
                               transform = transform,
                               target_transform = target_transform)
```


