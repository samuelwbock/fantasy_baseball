import numpy as np
from sklearn import preprocessing


def get_file_contents(path, date):
    f = open(path)
    first = f.readline()
    contents = np.loadtxt(f, dtype=str, delimiter='\t')
    normalized = clean_data(contents, date)
    return normalized


def floaten(data):
    floater = lambda x: float(x)
    vector = np.vectorize(floater)
    return vector(data)


def clean_data(data: np.ndarray, date:str):
    labels = data[:, :2]
    raw_data = np.append(data[:, 3:15], data[:, 17:], 1)
    normalized_data = preprocessing.normalize(raw_data)

    dates = normalized_data[:, 0:1]
    normalize = lambda x: date
    vector = np.vectorize(normalize)
    dates = vector(dates)

    labels = np.append(dates, labels, 1)
    normalized = np.append(labels, normalized_data, 1)
    return normalized


def check_columns():
    master = 'Player,Team,G,AB,R,H,2B,3B,HR,RBI,SB,CS,BB,SO,HBP,AVG,OBP,SLG,OPS'
    first = '"Player","Team","G","PA","AB","H","2B","3B","HR","R","RBI","BB","SO","HBP","SB","CS","-1","AVG","OBP","SLG","OPS","wOBA","-1","wRC+","BsR","Fld","-1","Off","Def","WAR","-1","ADP","playerid"'

    second = 'Player	Team	Pos	G	AB	R	H	2B	3B	HR	RBI	SB	CS	BB	SO	SH	SF	HBP	AVG	OBP	SLG	OPS'

    first = first.replace('"','').split(",")
    first = first[:3] + first[4:5] + first[9:10] + first[5:9] + first[10:11] + first[14:16] + first[11:14] + first[17:21]
    first = str.join(',', first)

    second = second.split(('\t'))
    second = second[:2] + second[3:15] + second[17:]
    second = str.join(',', second)


    assert second == first


def get_predict_data(path):
    f = open(path)
    first = f.readline()
    contents = np.loadtxt(f, dtype=str, delimiter=',')
    normalize = lambda x: x.replace('"', '')
    vector = np.vectorize(normalize)
    normalized_contents = vector(contents)
    labels = normalized_contents[:, [0,1]]
    raw_data = normalized_contents[:, [2,4,9,5,6,7,8,10,14,15,11,12,13,17,18,19,20]]
    normalized_data = preprocessing.normalize(raw_data)
    normalized = np.append(labels, normalized_data, 1)
    return normalized


def get_yearly_stats():
    team_wins = np.loadtxt(open("../raw/predict/teamWins.csv"), dtype=str, delimiter=',')
    team_wins = team_wins[0:,[0, 3, 5]]
    return team_wins


def get_score(players, wins):
    win_map = {}
    for win in wins:
        year = win[0]
        team = win[1]
        rank = win[2]
        if team == 'MIA':
            win_map[year + "FLA"] = float(rank)
        win_map[year + team] = float(rank)

    ranks = []
    misses = []
    scores = players[:, [0,2]]
    for score in scores:
        year = score[0]
        team = score[1]
        key = year + team
        if key not in win_map:
            misses.append(key)
        else:
            ranks.append(win_map[key])

    return np.array(ranks)


def get_data():
    training_data = None
    for i in range(0,9):
        file_contents = get_file_contents("../raw/predict/201{}.csv".format(i), '201{}'.format(i))
        if training_data is None:
            training_data = file_contents
        else:
            training_data = np.append(training_data, file_contents, 0)

    predict_data = get_predict_data("../raw/test/predict.csv")

    yearly_stats = get_yearly_stats()

    scores = get_score(training_data, yearly_stats)

    labels = predict_data[:, 0:2]
    predict_data = floaten(predict_data[:, 2:])
    training_data = floaten(training_data[:, 3:])

    return training_data, labels, predict_data, scores


check_columns()
get_data()
