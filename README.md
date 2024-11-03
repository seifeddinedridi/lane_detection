# Lane Detection
The first part of this project is an implementation of [ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation](https://arxiv.org/pdf/1606.02147.pdf).

While the second part is an implementation of [Towards End-to-End Lane Detection: an Instance Segmentation
Approach](https://arxiv.org/pdf/1802.05591.pdf).

# The ENet model
The model consists of 7 stages:

- Stage 0: Downsamples the input image
- Stages 1, 2, 3: encoder stages
- Stages 4, 5: decoder stages
- Transpose CNN: upsamples the output of stage 5 into the desired target size

# The LaneNet model
There are 2 branches in the LaneNet model. The first one is the binary segmentation branch, and the second one is the instance segmentation branch.

Both branches share the first 2 stages of the Enet neural network, leaving stage 3 and the full decoder as the backbone of each branch.

The binary segmentation branch finds the pixels forming the lanes and connects them even when they are occluded by objects such as cars.
It uses the bounded inverse class weighting because the two classes (lane/background) are highly unbalanced.

The instance segmentation branch separates these lane pixels into separate lanes. It does this by using a one-shot method based on distance metric learning.
This branch is trained to produce an embedding for each lane pixel. In the following stage, those embedding will be clustered together in such a way as to facilitate 
the process of forming the lanes using polynomial fitting. The lanes are converted into clusters using both pull and push forces.

## Binary segmentation branch

Mean-shift clustering

## Instance segmentation branch

## Transforming the lane pixels into curves
The prediction of the lanes should be insensitive to the view projection which might greatly change based on the sloping of the road and of course the camera being used.

The least-squares algorithm is used to fit an n-degree polynomial.

## Install dependencies

```bash
pip3 install -r requirements.txt
```

## Crate a virtual environment

```bash
python -m venv lane_detection
```

## Activate the virtual environment

```bash
source ./venv/bin/activate
```

## Run unit tests

```bash
python3 -m unittest discover
```

## Packaging the project for distribution

```bash
python3 -m pip install --upgrade build
python3 -m build
```

Start Jupyter notebook

```bash
jupyter notebook
```

## Uploading the distribution archives

```bash
python3 -m pip install --upgrade twine
python3 -m twine upload --repository testpypi dist/*
```

## Howto

### Download the Cityscape dataset

You should already have an account. If you don't, you can request one for free from the Cityscape maintainers.

```bash
wget -v --keep-session-cookies --save-cookies=cookies.txt --post-data 'username={username}&password={password}&submit=Login' https://www.cityscapes-dataset.com/login/
```

For downloading packages:

```bash
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=2
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=3
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=4
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=6
```

The mapping of package IDs to files:

| Id | Filename                             |
|----|--------------------------------------|
| 1  | gtFine_trainvaltest (241MB)          |
| 2  | gtCoarse.zip (1.3GB)                 |
| 3  | leftImg8bit_trainvaltest.zip (11GB)  |
| 4  | leftImg8bit_trainextra.zip (44GB)    |
| 5  | rightImg8bit_trainvaltest.zip (11GB) |
| 6  | rightImg8bit_trainextra.zip (44GB)   |
| 8  | camera_trainvaltest.zip (2MB)        |
| 9  | camera_trainextra.zip (8MB)          |
| 10 | vehicle_trainvaltest.zip (2MB)       |
| 11 | vehicle_trainextra.zip (7MB)         |

### Create the necessary directories

```bash
mkdir pretrained_model
mkdir -p datasets/cityscapes/data_unzipped/
```

### Unzip the Cityscape dataset

```bash
unzip -o datasets/cityscapes/data/gtCoarse.zip -d datasets/cityscapes/data_unzipped/
unzip -o datasets/cityscapes/data/leftImg8bit_trainvaltest.zip -d datasets/cityscapes/data_unzipped/
unzip -o datasets/cityscapes/data/leftImg8bit_trainextra.zip -d datasets/cityscapes/data_unzipped/
unzip -o datasets/cityscapes/data/rightImg8bit_trainvaltest.zip -d datasets/cityscapes/data_unzipped/
unzip -o datasets/cityscapes/data/rightImg8bit_trainextra.zip -d datasets/cityscapes/data_unzipped/
unzip -o datasets/cityscapes/data/gtFine_trainvaltest.zip -d datasets/cityscapes/data_unzipped/
```