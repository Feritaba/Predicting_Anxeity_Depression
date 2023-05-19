# Depression and Anxiety Prediction of Students using Electronic Health Records

Link to [access the project](https://github.com/Feritaba/Predicting_Anxeity_Depression) online
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

### Experimental Notebooks
All .ipynb files in "ExperimentationFiles_FeatureSelection_Imputation_Modelings" folder are used for experiments and prototyping of choosing the best method in each step such as missing value imputations, feature selection and modeling.

### Contributions:
This project is done as part of the DATA 240 course at SJSU, Spring 2023. The contributions to this project are as follow:
<br>Problem Definition and Data Collection : Pallavi, Foroozan, Deepak
<br>Exploratory Data Analysis : Pallavi, Deepak
<br>Data Pre-processing : Deepak, Foroozan
<br>Feature Selection : Pallavi, Deepak
<br>Modeling : Foroozan, Pallavi
<br>Experimental Analysis : Foroozan, Deepak
<br>Evaluation : Pallavi, Foroozan
<br>Conclusion & Documentation : Pallavi, Foroozan, Deepak



