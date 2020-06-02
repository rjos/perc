from src.data.build_dataset import build, normalize
from src.model.classification.perturbation import PerC_Mean, PerC_Covariance, PerC

from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.metrics import accuracy_score

import numpy as np

if __name__ == "__main__":
    
    # Set Folds number
    folds = 10

    # Set repeats times
    repeats = 100

    # Set dataset into data/raw directory
    dataset = 'balance'

    # Load dataset
    X, y = build(dataset)

    # Normalize attribute into [0, 1]
    X = normalize(X)

    # Split dataset in train and test sets 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    # Step: Train classifiers
    ## PerC(Mean)
    perc_mean = PerC_Mean()
    perc_mean.fit(X_train, y_train)

    ## PerC(Cov)
    perc_cov = PerC_Covariance()
    perc_cov.fit(X_train, y_train)

    ## PerC
    perc = PerC()
    perc.fit(X_train, y_train)

    # Step: Test classifiers
    ## PerC(Mean)
    y_perc_mean = perc_mean.predict(X_test)
    
    ## PerC(Cov)
    y_perc_cov = perc_cov.predict(X_test)

    ## PerC
    y_perc = perc.predict(X_test)

    # Step: Evaluation classifiers output
    print('Accuracy')
    print(f'PerC(Mean): {accuracy_score(y_test, y_perc_mean)}')
    print(f'PerC(Cov): {accuracy_score(y_test, y_perc_cov)}')
    print(f'PerC: {accuracy_score(y_test, y_perc)}')




    



            


