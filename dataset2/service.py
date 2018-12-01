import pickle
from sklearn.model_selection import GridSearchCV
class service:
    def __init__(self):
        self.data = ""

    def read_csv_data(self, _url):
        with open(_url, 'r') as file:
            _fileResponse = [line.split(',') for line in file.read().split('\n')]
            _fileResponse.pop(_fileResponse.__len__() - 1)
            train_response = [[int(element) for element in row] for row in _fileResponse]
            train_features = [d[:-1] for d in train_response]
            train_labels = [d[-1] for d in train_response]
            return train_features, train_labels

    def read_csv_data_without_labels(url):
        with open(url, 'r') as file:
            data = [line.split(',') for line in file.read().split('\n')]
            data.pop(data.__len__() - 1)
            test_features = [[int(element) for element in row] for row in data]
            return test_features

    def save_csv_result(self, _predicted, file_name, _path):
        with open(_path + '/' + file_name + '.csv', 'w') as _file:
            for i in range(len(_predicted)):
                _file.write('%d,%d\n' % (i + 1, _predicted[i]))

    def save_model_to_pkl(self, classifier, file_name, _path):
        with open(_path +'/'+ file_name + '.pkl', 'wb') as _file:
            pickle.dump(classifier, _file)

    def hyper_parameter_tuning(self, _classifier, param_dist, train_features, train_labels):
        clf = GridSearchCV(estimator=_classifier, param_grid=param_dist, cv=10)
        clf.fit(train_features, train_labels)
        print('Best hyerparameters:\n', clf.best_params_)
        print('Best CV accuracy'.format(clf.best_score_))
