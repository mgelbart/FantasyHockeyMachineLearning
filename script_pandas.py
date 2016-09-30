import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt


# Calculates the 2015/2016 season fantasy points for each player, according to the following formula:
# Fantasy Points = Goals * 4 + Assists * 2.5 - Powerplay Goals - Powerplay Assists * 0.5 + Plus/Minus * 0.6 +
# Shots * 0.1 + Hits * 0.06 + Blocked Shots * 0.03
def compute_score(master):
    Y = ((master['G|20152016'] + master['PP_G|20152016']) * 4) + \
        ((master['A|20152016'] + master['PP_A|20152016']) * 2.5) + \
        ((master['GF|20152016'] - master['GA|20152016']) * 0.6) + \
        ((master['PP_G|20152016']) * -1) + \
        ((master['PP_A|20152016']) * -0.5) + \
        ((master['iSF|20152016'] + master['PP_iSF|20152016']) * 0.1) + \
        ((master['iHF|20152016']) * 0.06) + \
        ((master['iBLK|20152016']) * 0.03)
    return Y


# Reads raw data from es1-es5 (even strength statistics) and pp1 (powerplay statistics). Each row in these files
# represents a player-year object, which is a single season of play for a single player. The data has been pre-scaled
# for the 2012/2013 NHL lockout and percentage fields (such as shooting percentage) with div0 errors have been replaced
# with 0. Year data ranges from 2011/2012 to 2015/2016 but only players who played in 2015/2016 are present.
#
# This method flattens the player-year objects into player objects with one set of fields for each season.
# any players have valid player-year objects for certain years, but played 0 minutes of powerplay during those years,
# so those players are artificially added to pp1 and filled with 0s for all features. All tables are merged into one
# master table. es[5] contained only ages for the 2015/2016 season, so ages are reverse calculated for prior years.
# Missing years are back filled.
#
# The training data is the subset of the master table for years 2011/2012 to 2014/15.
# The answer vector is the fantasy hockey points for each player in the 2015/2016 season as computed by computer_score.
# The test data is the subset of the master table for years 2012/2013 to 2015/2016.
def pre_process():
    # load CSVs
    es = dict()
    for i in range(1, 6):
        if i == 5:
            es[i] = pd.read_csv("es%s.csv" % i, encoding="ISO-8859-1")
        else:
            es[i] = pd.read_csv("es%s.csv" % i)
    pp1 = pd.read_csv("pp1.csv")

    # add players not in pp1 and fill them with 0
    not_in_pp1 = es[1][~es[1]["Player"].isin(pp1["Player"])].dropna()
    not_in_pp1 = not_in_pp1[["Player", "Season"]]
    not_in_pp1["Season"] = 20152016
    not_in_pp1 = not_in_pp1.drop_duplicates()
    pp1 = pp1.merge(not_in_pp1, on=["Player", "Season"], how="outer")
    pp1 = pp1.fillna(value=0)
    pp1.Season = pp1.Season.astype(int)

    # flatten to remove dates.
    for k, v in es.items():
        es[k] = v.pivot(index="Player", columns="Season")
    pp1 = pp1.pivot(index="Player", columns="Season")

    # merge tables
    master = pd.concat([es[1], es[2], es[3], es[4], es[5], pp1], axis=1)

    # remove players who didn't play in 20152016.
    master = master[master["GP"][20152016] > 0]

    # set ages
    master["Age", 20142015] = np.where(pd.notnull(master["GP"][20142015]), master["Age"][20152016] - 1, master["Age"][20152016])
    master["Age", 20132014] = np.where(pd.notnull(master["GP"][20132014]), master["Age"][20142015] - 1, master["Age"][20142015])
    master["Age", 20122013] = np.where(pd.notnull(master["GP"][20122013]), master["Age"][20132014] - 1, master["Age"][20132014])
    master["Age", 20112012] = np.where(pd.notnull(master["GP"][20112012]), master["Age"][20122013] - 1, master["Age"][20122013])

    # fill NAs
    master = master.fillna(method="bfill", axis=1)

    # flatten hierarchy
    master.columns = ['%s%s' % (a, '|%s' % b if b else '') for a, b in master.columns]

    # compute score
    Y = compute_score(master)

    # create test and training data
    train_cols = [c for c in master.columns if "20152016" not in c]
    X = master[train_cols]

    test_cols = [c for c in master.columns if "20112012" not in c]
    test_data = master[test_cols]

    return X, Y, test_data


# Exports to two CSV files.
# 1. Exports a list of the features and their weights to feature_weights.csv.
# 2. Exports a list of player predictions for the 2015/2016 data based on the fitted model to predictions.csv.
def export_data(rf, X, Y, test_data):
    rf.fit(X, Y)
    importances = pd.DataFrame(rf.feature_importances_)
    importances.index = X.columns
    print("saved feature weights to feature_weights.csv")
    importances.to_csv("feature_weights.csv")

    result = pd.DataFrame(rf.predict(test_data))
    result.index = Y.index
    print("saved predictions to predictions.csv")
    result.to_csv("predictions.csv")


# Graphs the OOB_Error for estimators ranging from min_estimators to max_estimators.
def print_oob_vs_estimators(min_estimators, max_estimators, step, rf, X, Y):
    error_rate = []
    step = step
    for i in range(min_estimators, max_estimators, step):
        rf.set_params(n_estimators=i, oob_error=True)
        rf.fit(X, Y)
        oob_error = 1 - rf.oob_score_
        error_rate.append(oob_error)

    # Generate the "OOB error rate" vs. "n_estimators" plot.
    xs = range(min_estimators, max_estimators, step)
    ys = error_rate
    plt.plot(xs, ys)

    plt.xlim(min_estimators, max_estimators)
    plt.xlabel("n_estimators")
    plt.ylabel("OOB error rate")
    plt.show()


# Processes the data and creates the model object and then either makes a prediction and exports the data or tests a
# variety of potential tree values. Always comment all but one of the lines marked with an asterisk. Fitting is done in
# an asterisked method.
def main():
    X, Y, test_data = pre_process()
    rf = RandomForestRegressor(n_estimators=10, n_jobs=-1)     # *
    #print_oob_vs_estimators(100, 550, 50, rf, X, Y)            # *
    export_data(rf, X, Y, test_data)


if __name__ == "__main__":
    main()