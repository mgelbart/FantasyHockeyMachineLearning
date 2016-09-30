import numpy as np
from sklearn.ensemble import RandomForestClassifier


def remove_2015(numpy_array, num_cols_per_year):
    for i in range(num_cols_per_year * 5 - 1, num_cols_per_year * 4 - 1, -1):
        numpy_array = np.delete(numpy_array, i, 1)
    return numpy_array


def remove_2011(numpy_array, num_cols_per_year):
    for i in range(num_cols_per_year - 1, -1, -1):
        numpy_array = np.delete(numpy_array, i, 1)
    return numpy_array


def get_score(numpy_array, num_cols_per_year, frames):
    Y = []
    base = num_cols_per_year * 4
    for row in numpy_array:
        score = (float(row[base + frames[0].shape[1] + frames[1].shape[1] - 4]) + float(row[base + frames[0].shape[1] + frames[1].shape[1] + frames[2].shape[1] + frames[3].shape[1] - 8 + 1])) * 4 # goals
        score += (float(row[base + frames[0].shape[1] + frames[1].shape[1] - 4 + 3]) + float(row[base + frames[0].shape[1] + frames[1].shape[1] + frames[2].shape[1] + frames[3].shape[1] - 8 + 4])) * 2.5  # assists
        score += (float(row[base + frames[0].shape[1] - 2 - 7]) - float(row[base + frames[0].shape[1] - 2 - 6])) * 0.6 # p/m
        score += float(row[base + frames[0].shape[1] + frames[1].shape[1] + frames[2].shape[1] + frames[3].shape[1] - 8 + 1]) * -1  # pp goals
        score += float(row[base + frames[0].shape[1] + frames[1].shape[1] + frames[2].shape[1] + frames[3].shape[1] - 8 + 4]) * -1  # pp assists
        score += (float(row[base + frames[0].shape[1] + frames[1].shape[1] - 4 + 18]) + float(row[base + frames[0].shape[1] + frames[1].shape[1] + frames[2].shape[1] + frames[3].shape[1] - 8 + 19])) * 0.1 # shots
        score += float(row[base + frames[0].shape[1] + frames[1].shape[1] - 4 + 26]) * 0.06 #hits
        score += float(row[base + frames[0].shape[1] + frames[1].shape[1] - 4 + 30]) * 0.03 #blocked shots
        Y.append(score)
    return Y


def get_correct_col(frame, column, year, num_cols_per_year, frames):
    col = 0
    if year == b'20122013':
        col = num_cols_per_year
    if year == b'20132014':
        col = num_cols_per_year * 2
    if year == b'20142015':
        col = num_cols_per_year * 3
    if year == b'20152016':
        col = num_cols_per_year * 4
    for i in range(frame):
        col += frames[i].shape[1] - 2
    col += column
    return col


def curate_data():
    d1 = np.genfromtxt('Corsica_Skater.Stats_03h23.csv', delimiter=',', unpack=True, dtype=None).transpose()
    d2 = np.genfromtxt('Corsica_Skater.Stats_03h23 (1).csv', delimiter=',', unpack=True, dtype=None).transpose()
    d3 = np.genfromtxt('Corsica_Skater.Stats_03h23 (2).csv', delimiter=',', unpack=True, dtype=None).transpose()
    d4 = np.genfromtxt('Corsica_Skater.Stats_03h23 (3).csv', delimiter=',', unpack=True, dtype=None).transpose()
    pp1 = np.genfromtxt('Corsica_Skater.Stats_07h20.csv', delimiter=',', unpack=True, dtype=None).transpose()
    d1 = np.delete(d1, 0, 0)
    d2 = np.delete(d2, 0, 0)
    d3 = np.delete(d3, 0, 0)
    d4 = np.delete(d4, 0, 0)
    pp1 = np.delete(pp1, 0, 0)
    frames = [d1, d2, d3, d4, pp1]
    num_cols_per_year = int(d1.shape[1] + d2.shape[1] + d3.shape[1] + d4.shape[1] + pp1.shape[1]) - 10
    master = dict()
    unique_players = set()
    for row in d1:
        if row[1] == b'20152016' and row[0] not in unique_players:
            unique_players.add(row[0])
            master[row[0]] = [None for x in range(num_cols_per_year * 5)]
    for i in range(len(frames)):
        for row in frames[i]:
            if row[0] in unique_players:
                for j in range(2, row.size):
                    try:
                        float(row[j])
                        master[row[0]][get_correct_col(i, j - 2, row[1], num_cols_per_year, frames)] = float(row[j])
                    except ValueError:
                        if row[j] == b'NA':
                            master[row[0]][get_correct_col(i, j - 2, row[1], num_cols_per_year, frames)] = 0
                        else:
                            master[row[0]][get_correct_col(i, j - 2, row[1], num_cols_per_year, frames)] = row[j]

    for key, player in master.items():
        for i in range(len(player)):
            if player[i] is None:
                further_indexes = []
                j = i + num_cols_per_year
                while j < len(player):
                    further_indexes.append(j)
                    j += num_cols_per_year
                nones = 0
                for x in further_indexes:
                    if type(player[x]) is not float and type(player[x]) is not int and player[x] is not None:
                        player[i] = player[x]
                        break
                    if player[x] is None:
                        nones += 1
                if nones == len(further_indexes):
                    player[i] = 0
                if player[i] is None:
                    temp_array = np.array([player[x] if player[x] is not None else np.NaN for x in further_indexes])
                    player[i] = np.nanmean(temp_array)

    master_numpy = np.array(list(master.values()))
    names = list(master.keys())
    return names, remove_2015(master_numpy, num_cols_per_year), remove_2011(master_numpy, num_cols_per_year), \
                              get_score(master_numpy, num_cols_per_year, frames)


def print_Y(names, Y):
    strings = []
    for i in range(len(names)):
        strings.append(str(names[i]) + str(Y[i]))
    strings.sort()
    for i in strings:
        print(i)


def main():
    names, X_end_2014, X_end_2015, Y = curate_data()
    forest = RandomForestClassifier()
    forest.fit(X=X_end_2014, y=np.asarray(Y, dtype="|S6"))
    Ypredict = forest.predict(X_end_2015)
    print_Y(names, Ypredict)

if __name__ == "__main__":
    main()