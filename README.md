<p align="center">
  <img width='650' src='https://github.com/chivington/Reckon/blob/master/imgs/reckon.jpg' alt='Reckon Logo'/>
</p>

# Reckon
Reckon is a multi-domain data analytics platform for curating datasets, performing inference tasks, and enabling users to gain valuable insights into their individual sets of structured and unstructured data.

![Build Status](https://img.shields.io/badge/build-Stable-green.svg)
![License](https://img.shields.io/badge/license-NONE-lime.svg)
<br/><br/><br/>

## Contents
* [Prerequisites](https://github.com/chivington/Reckon/tree/master#prerequisites)
* [Installation](https://github.com/chivington/Reckon/tree/master#installation)
* [Usage](https://github.com/chivington/Reckon/tree/master#usage)
* [Authors](https://github.com/chivington/Reckon/tree/master#authors)
* [Contributing](https://github.com/chivington/Reckon/tree/master#contributing)
* [Acknowledgments](https://github.com/chivington/Reckon/tree/master#acknowledgments)
* [License](https://github.com/chivington/Reckon/tree/master#license)
<br/>

## Prerequisites
  * Python
  * Numpy
  * Matplotlib
<br/><br/>


## Installation
```bash
  git clone https://github.com/chivington/Reckon.git
```
<br/>


## Usage
Install Reckon by cloning this repository. Once cloned, change into the root directory and run reckon with:
```bash
  python reckon.py
```

Be sure to use the correct version of Python. Some users have multiple versions of Python installed and may not be aware that certain modules, such as Numpy, are only installed in one version or the other. Reckon should run in any stable version of Python but be sure you run the version that has the correct dependencies if you have multiple versions installed.

You may pass additional parameters at the start but none are required. View the help menu for a full list of available runtime parameters with one of the following commands:
```bash
  python reckon.py -h
  python reckon.py --help
```

Reckon's operation is streamlined to enable users to design and run experiments flexibly, on a wide range of data. To use Reckon:
1. Select a dataset.
2. Select a task to perform on the data.
3. Select one of the models available for that task.
4. Then supply the required hyperparameters for the selected model.
5. To run multiple experiments on the data with the same model, simply supply a list of lists of the hyperparameters required for that model.


Once a dataset, task, model and required hyperparameters are selected, Reckon will:
1. Load the dataset.
2. Split the dataset according to the model and supplied hyperparameters.
3. Display a random example from the training set.
4. Begin an optimization algorithm, displaying specified the metrics at specified intervals.
5. Display a random example from the test set, along with it's classification and label.
6. Print the final training and test set errors to the terminal.
7. Display plots of recorded metrics during optimization.
8. Save the recorded metrics and plots to
9. End.



Feel free to ask me questions on [GitHub](https://github.com/chivington)

<br/>
<p align="center">
  <img width='600' src='https://github.com/chivington/Reckon/blob/master/imgs/random-img.jpg' alt='Random Digit'/>
</p><br/>

<p align="center">
  <img width='600' src='https://github.com/chivington/Reckon/blob/master/imgs/errors-and-times.jpg' alt='Training & Validation Errors'/>
</p><br/>

<p align="center">
  <img width='600' src='https://github.com/chivington/Reckon/blob/master/imgs/classification.jpg' alt='Classification Test'/>
</p>
<br/><br/>


## Authors
* **Johnathan Chivington:** [Web](https://chivington.net) or [GitHub](https://github.com/chivington)

## Contributing
Not currently accepting outside contributors, but feel free to use as you wish.

## License
There is currently no license associated with this content.
<br/><br/>
