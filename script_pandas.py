import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


def preprocess():
    # load CSVs
    es1 = pd.read_csv("es1.csv")
    es2 = pd.read_csv("es2.csv")
    es3 = pd.read_csv("es3.csv")
    es4 = pd.read_csv("es4.csv")
    es5 = pd.read_csv("es5.csv", encoding="ISO-8859-1")
    pp1 = pd.read_csv("pp1.csv")

    # add players not in pp1 and fill them with 0
    not_in_pp1 = es1[~es1["Player"].isin(pp1["Player"])].dropna()
    not_in_pp1 = not_in_pp1[["Player", "Season"]]
    not_in_pp1["Season"] = 20152016
    not_in_pp1 = not_in_pp1.drop_duplicates()
    pp1 = pp1.merge(not_in_pp1, on=["Player", "Season"], how="outer")
    pp1 = pp1.fillna(value=0)
    pp1.Season = pp1.Season.astype(int)

    # flatten to remove dates.
    es1 = es1.pivot(index="Player", columns="Season")
    es2 = es2.pivot(index="Player", columns="Season")
    es3 = es3.pivot(index="Player", columns="Season")
    es4 = es4.pivot(index="Player", columns="Season")
    es5 = es5.pivot(index="Player", columns="Season")
    pp1 = pp1.pivot(index="Player", columns="Season")

    # merge tables
    master = pd.concat([es1, es2, es3, es4, es5, pp1], axis=1)

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
    Y = ((master['G|20152016'] + master['PP_G|20152016']) * 4) + \
        ((master['A|20152016'] + master['PP_A|20152016']) * 2.5) + \
        ((master['GF|20152016'] - master['GA|20152016']) * 0.6) + \
        ((master['PP_G|20152016']) * -1) + \
        ((master['PP_A|20152016']) * -0.5) + \
        ((master['iSF|20152016'] + master['PP_iSF|20152016']) * 0.1) + \
        ((master['iHF|20152016']) * 0.06) + \
        ((master['iBLK|20152016']) * 0.03)

    # create test and training data
    train_cols = [c for c in master.columns if "20152016" not in c]
    X = master[train_cols]

    test_cols = [c for c in master.columns if "20112012" not in c]
    test_data = master[test_cols]

    return X, Y, test_data


def print_text(rf, X, Y, test_data):
    rf.fit(X, Y)
    print("FEATURE IMPORTANCES")
    importances = rf.feature_importances_
    for f in importances:
        print(f)

    result = pd.DataFrame(rf.predict(test_data))
    result.index = Y.index
    print("saved predictions to predictions.csv")
    result.to_csv("predictions.csv")


def print_OOB_vs_estimators(min, max, step, rf, X, Y):
    error_rate = []
    min_estimators = min
    max_estimators = max
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


def main():
    X, Y, test_data = preprocess()
    rf = RandomForestRegressor(n_estimators=10, n_jobs=-1)
    #print_OOB_vs_estimators(100, 550, 50, rf, X, Y)
    print_text(rf, X, Y, test_data)


if __name__ == "__main__":
    main()