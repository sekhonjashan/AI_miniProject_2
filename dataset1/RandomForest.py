from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.extend(["../"])
sys.path.extend(["."])
from dataset1.service import service


def load_random_forest():
        _service = service()
        train_features, train_labels = _service.read_csv_data('dataset1/ds1/ds1Train.csv')
        validation_features, validation_labels = _service.read_csv_data('dataset1/ds1/ds1Val.csv')
        # classifier = RandomForestClassifier()
        classifier = RandomForestClassifier(n_estimators=10000)
        classifier = classifier.fit(train_features, train_labels)
        validation_predicted = classifier.predict(validation_features)
        print(classification_report(validation_labels, validation_predicted))
        print(confusion_matrix(validation_labels, validation_predicted))
        print("Accuracy Score:", accuracy_score(validation_labels, validation_predicted))
        _service.save_csv_result(validation_predicted, 'ds1Val-3', 'output_csvs')
        _service.save_model_to_pkl(classifier, 'ds1Classifier-3', 'dataset1/models/')
        # _service.hyper_parameter_tuning(RandomForestClassifier(), {'n_estimators': [10,100,200,300,400,500,1000]}, train_features,
        #                                 train_labels)


def parameter_iteration_tunning():
        _service = service()
        train_features, train_labels = _service.read_csv_data('dataset1/ds1/ds1Train.csv')
        validation_features, validation_labels = _service.read_csv_data('dataset1/ds1/ds1Val.csv')
        _range = [10,100,200,300,400,500,1000]
        for index in _range:
                _clf = RandomForestClassifier(n_estimators=index)
                _clf.fit(train_features, train_labels )
                pred = _clf.predict(validation_features)
                print('RandomForest accuracy ' + str(index) + ' is', accuracy_score(validation_labels, pred))


load_random_forest()
# parameter_iteration_tunning()
