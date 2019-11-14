# Predicting Star Ratings from User Reviews

*Group : Anders Poirel, Oasys Okubo and Anish Parasrampuria*

Final project for CSE 142: Machine Learning and Data Mining, taught at UC Santa Cruz in Fall 2019.

This project has for goal to predict the star rating given a Yelp review.

## Installation

### Setting up the environment
Clone this repository and run

```
conda env create -f environment.yml
```
Switch to the newly created environement:
```
conda activate cse142-finalproject
```

### Obtaining data

Download the data from [https://ucsc-courses.github.io/CSE142-Fall2019/hw/SampleFilesForProject.zip](https://ucsc-courses.github.io/CSE142-Fall2019/hw/SampleFilesForProject.zip). Extract the files and save `data_train.json` and `data_test.json` to `data/raw`.

## Project Structure

* `data`
    * `data/raw`: raw json files
    * `data/processed`: serialized data after processign through the pipeline
* `notebooks`: explorary jupyter notebooks - naming convention `<ucsc id>-<purpose>-<number>.ipynb`
* `output`: saved serialized tensorflow models and model predictions
* `predicting_ratings`: source code for the project
    * `predicting_ratings/features`: code for data pre-processing
    * `predicting_ratings/models`: code for building models and generating predictions
* `reports`: final project reports

## Generating Predictions

