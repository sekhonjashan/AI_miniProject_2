from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.extend(["../"])
sys.path.extend(["."])
from dataset1.service import service


def load_kneighbors():
        _service = service()
        train_features, train_labels = _service.read_csv_data('dataset1/ds1/ds1Train.csv')
        validation_features, validation_labels = _service.read_csv_data('dataset1/ds1/ds1Val.csv')
        # classifier = KNeighborsClassifier(n_neighbors=1)
        classifier = KNeighborsClassifier(n_neighbors=1)
        classifier = classifier.fit(train_features, train_labels)
        validation_predicted = classifier.predict(validation_features)
        print(classification_report(validation_labels, validation_predicted))
        print(confusion_matrix(validation_labels, validation_predicted))
        print("Accuracy Score:", accuracy_score(validation_labels, validation_predicted))
        _service.save_csv_result(validation_predicted, 'k_neighbors_results', 'dataset1/results/')
        _service.save_model_to_pkl(classifier, 'k_neighbors_results', 'dataset1/models/')
        # _service.hyper_parameter_tuning(KNeighborsClassifier(), {'n_neighbors': range(1, 50)}, train_features,
        #                                 train_labels)


def parameter_iteration_tunning():
        _service = service()
        train_features, train_labels = _service.read_csv_data('dataset1/ds1/ds1Train.csv')
        validation_features, validation_labels = _service.read_csv_data('dataset1/ds1/ds1Val.csv')
        _range = range(1, 30)
        for index in _range:
                _clf = KNeighborsClassifier(n_neighbors=index)
                _clf.fit(train_features, train_labels )
                pred = _clf.predict(validation_features)
                print('NeighborsClassifier accuracy ' + str(index) + ' is', accuracy_score(validation_labels, pred))


load_kneighbors()
# parameter_iteration_tunning()
