from sklearn.neural_network import MLPRegressor
import numpy as np

from predict.clean import get_data, get_yearly_stats


def sort_predictions(labeled_preds):
    results = []
    for labeled_pred in labeled_preds:
        key = '{}__{}'.format(str(labeled_pred[2]), labeled_pred[0])
        results.append(key)
    results.sort()
    return results


def main():
    training_data, labels, predict_data, scores = get_data()

    clr = MLPRegressor(solver='lbfgs', alpha=1e-5,
                       hidden_layer_sizes=(5, 2), random_state=1)
    fit = clr.fit(training_data, scores)

    pred = fit.predict(predict_data)

    labeled_preds = np.append(labels, np.reshape(pred, [4112,1]), 1)

    sorted = sort_predictions(labeled_preds)

    f = open('pitchers.txt', 'w')
    for rec in sorted:
        f.write(rec + '\n')
    f.close()

if __name__ == "__main__":
    main()