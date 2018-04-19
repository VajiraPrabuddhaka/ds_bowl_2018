# Spot Neuclei in cell Images using Mask-RCNN (Instance segmentation)

This project has used Mask-RCNN architecture to detect and segment instances of neuclei in cell images. This project based on Mask-RCNN [implementation](https://github.com/matterport/Mask_RCNN) using `keras`. Further information on Mask-RCNN can be found [here](https://github.com/facebookresearch/Detectron) by the Facebook AI research called `Detectron`.

## Getting Started..

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

```angular2html
Linux pc with python3 installed
CUDA capable GPU 
keras with Tensorflow backend
Other python libraries like numpy, matplotlib(for visualizations), pandas, skimage etc..
``` 
Please refer to the documentations of above tools and install latest versions that they supported. I recommand to install them in a seperate python virtual environment for convenience.

### Installing..

First we need to install `mrcnn` into our python environment. If you are using virtual environment, activate it and then,
```angular2html
$ git clone https://github.com/matterport/Mask_RCNN.git
$ cd Mask_RCNN
$ python setup.py install
```  
### Start training..

For this project I used kaggle dataset can be found [here](https://www.kaggle.com/c/data-science-bowl-2018), you can just log into this and download train dataset under the 'data' section. There are 670 images with their corresponding masks in the dataset.

Please make sure to set correct path of your train dataset to `TRAIN_PATH` variable in the `train.py` script. 

After setting all, just run the `train.py` file by using
```angular2html
python train.py
``` 
Please note that you can change training parameters like no, of epochs, validation steps per epoch, etc.. Further information can be found [here](https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/config.py). All the configurations must be go inside of `train.py` file.

### Do simple Inference using trained model..

For the demonstration of inference I used kaggle's dataset as above training dataset. You may find it easily using the link that I posted under training section.

For test some inference,
```angular2html
python test.py
```

Please make sure that you have set correct paths and configurations you wish to use, before starting. For more information just refer to the code or create an issue under this repo..



### Hope you enjoy well....  
