# Depression and Anxiety Prediction of Students using Electronic Health Records

In this project we are trying to capture the patterns that will lead to depression and anxiety in students pursing bachelor degree.
### DATASET
The dataset can be found [here](https://datadryad.org/stash/dataset/doi:10.5061/dryad.54qt7).

### RUN THE PROJECT
To run the project:
- cd into root of the project and create a virtual environment with Python 3.9
- run `pip install -r requirements.txt`
- run `python main.py`
- You will see the results for 2 predictions, first, predicting anxiety and second, predicting depression. With getting to each step, the program prints the step following the list of feature selection along with its scores and modeling evaluations.
<br>In modeling.py file, the data has been fitted to our most well performed model, Ensemble model, which consists of Random Forest as the base estimator and AdaBoost as boosting technique.
<br> 
<br>All .ipynb files are used for experiments and prototyping of choosing the best method in each step.

<br>This project is done as part of the DATA 240 course at SJSU, Spring 2023.
