# kg-dog-breed
Code for kaggle's [Dog Breed Identification problem](https://www.kaggle.com/c/dog-breed-identification).

## Setup
1. Clone the repository and navigate to its root.
```
git clone https://github.com/hjwk/kg-dog-breed
cd kg-dog-project
```

2. Create a virtual environment from the requirements files (I used the same environment as for Udacity's [dog project](https://github.com/udacity/dog-project).
```
conda env create -f requirements/dog-windows.yml
activate kg-dog-breed
```

3. Download the [training](https://www.kaggle.com/c/dog-breed-identification/download/train.zip) and [testing](https://www.kaggle.com/c/dog-breed-identification/download/test.zip) data and put them in `data/train` and `data/test`. 

4. Execute the dataCleaning.py script in order to re-organize the training data into a training and a validation datasets. After running this script you should have a new folder named data_gen and withing it a test and train folder in which the photos are organized into folders named after their classes.

## Generating the bottleneck features and training the CNN
You can now use kg-dog-breed.py to generate the bottleneck features and train your model. If you do not want to rebuild the bottleneck features each tme you run the script simply comment the appropriate lines.
