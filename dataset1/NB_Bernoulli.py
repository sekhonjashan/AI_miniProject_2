from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.extend(["../"])
sys.path.extend(["."])
from dataset1.service import service


def load_naive_bayes_bernoulli():
        _service = service()
        train_features, train_labels = _service.read_csv_data('dataset1/ds1/ds1Train.csv')
        validation_features, validation_labels = _service.read_csv_data('dataset1/ds1/ds1Val.csv')
        # classifier = BernoulliNB()
        classifier = BernoulliNB(alpha = 0.001)
        classifier = classifier.fit(train_features, train_labels)
        validation_predicted = classifier.predict(validation_features)
        print(classification_report(validation_labels, validation_predicted))
        print(confusion_matrix(validation_labels, validation_predicted))
        print("Accuracy Score:", accuracy_score(validation_labels, validation_predicted))
        _service.save_csv_result(validation_predicted, 'ds1Val-nb', 'output_csvs')
        _service.save_model_to_pkl(classifier, 'ds1Classifier-nb', 'dataset1/models/')
        # _service.hyper_parameter_tuning(BernoulliNB(), {'alpha': [0.01, 0.001, 0.1, 1, 10]}, train_features,
        #                                 train_labels)


def parameter_iteration_tunning():
        _service = service()
        train_features, train_labels = _service.read_csv_data('dataset1/ds1/ds1Train.csv')
        validation_features, validation_labels = _service.read_csv_data('dataset1/ds1/ds1Val.csv')
        _range = [0.01, 0.001, 0.1, 1, 10]
        for index in _range:
                _clf = BernoulliNB(alpha=index)
                _clf.fit(train_features, train_labels )
                pred = _clf.predict(validation_features)
                print('BernoulliNB accuracy ' + str(index) + ' is', accuracy_score(validation_labels, pred))


load_naive_bayes_bernoulli()
# parameter_iteration_tunning()
