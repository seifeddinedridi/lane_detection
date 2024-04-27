# Lane Detection

An implementation of [ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation](https://arxiv.org/pdf/1606.02147.pdf).

Tasks to do:
* <s>Understand what would be the input to Enet loss function (a set of label-polygon pair?)</s>
* <s>Implement a function for detecting if a 2D point is within the boundaries of a polygon</s>
* <s>Implement a reader for the cityscapes dataset</s>
* Check the implementation of the five stages
* ✅ Learn how max-pooling and max-unpooling internally work
* ✅ Implement stages four and five

# The ENet model
The model consists of 7 stages:
- Stage 0: Downsamples the input image
- Stages 1, 2, 3: encoder stages
- Stages 4, 5: decoder stages
- Transpose CNN: upsamples the output of stage 5 into the desired target size 

## What would be the output of the model?
It should output an image where the pixels are labeled with a number ranging from [0..C) where C is the number of classes.

## What would be the input to the loss function?
TODO

## Install dependencies

```bash
pip3 install -r requirements.txt
```

## Crate virtual environment
```bash
python -m venv lane_detection
```

## Activate virtual environment
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

### Unzip the Cityscape dataset

```bash
tar -xf datasets/cityscapes/data/gtCoarse.zip -C datasets/cityscapes/data_unzipped/
tar -xf datasets/cityscapes/data/gtFine_trainvaltest.zip -C datasets/cityscapes/data_unzipped/
tar -xf datasets/cityscapes/data/leftImg8bit_trainvaltest.zip -C datasets/cityscapes/data_unzipped/
tar -xf datasets/cityscapes/data/rightImg8bit_trainvaltest.zip -C datasets/cityscapes/data_unzipped/
```