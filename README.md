# FantasyHockeyMachineLearning

This is a very simple python script designed to predict the 2016/2017 fantasy hockey point score of every player who played a game in 2015/2016. The model of choice is [sklearn/RandomForestRegressor](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html). The formula for fantasy points is: 
``` 
Fantasy Points = Goals * 4 + Assists * 2.5 - Powerplay Goals - Powerplay Assists * 0.5 + Plus/Minus * 0.6 + Shots * 0.1 + Hits * 0.06 + Blocked Shots * 0.03 
```
## Data

Data in the csv files es1 to es4 and also in pp1 was downloaded from [Corsica](corsica.hockey). Every possible data category was downloaded for the seasons between 2011/2012 and 2015/2016. Some data processing occurs in the script, but before that, duplicate columns were removed, NAs were filled with 0s and relevant (non percentile or rate) columns were adjusted for the 2012/2013 lockout shortened season.

The data in es5 was downloaded from [Hockey Reference](Hockey-Reference.com).

## Dependencies
Python 3.5.2
sklearn 0.18
scipy 0.18.1
numpy 1.11.1

## Running the Script

Execute the script with `python script_pandas.py`.

The script will export data to feature_weights.csv and predictions.csv.
