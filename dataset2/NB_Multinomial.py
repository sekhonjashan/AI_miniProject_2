from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.extend(["../"])
sys.path.extend(["."])
from dataset1.service import service

def load_naive_bayes_multinomial():
        _service = service()
        train_features, train_labels = _service.read_csv_data('dataset2/ds2/ds2Train.csv')
        validation_features, validation_labels = _service.read_csv_data('dataset2/ds2/ds2Val.csv')
        # classifier = MultinomialNB()
        classifier = MultinomialNB(alpha=1)
        classifier = classifier.fit(train_features, train_labels)
        validation_predicted = classifier.predict(validation_features)
        print(classification_report(validation_labels, validation_predicted))
        print(confusion_matrix(validation_labels, validation_predicted))
        print("Accuracy Score:", accuracy_score(validation_labels, validation_predicted))
        _service.save_csv_result(validation_predicted, 'nb_multinomial_results', 'dataset2/results/')
        _service.save_model_to_pkl(classifier, 'nb_multinomial_results', 'dataset2/models/')
        # _service.hyper_parameter_tuning(MultinomialNB(), {'alpha': [0.01, 0.001, 0.1, 1, 10, 100, 1000]}, train_features,
        #                                 train_labels)


def parameter_iteration_tunning():
        _service = service()
        train_features, train_labels = _service.read_csv_data('dataset2/ds2/ds2Train.csv')
        validation_features, validation_labels = _service.read_csv_data('dataset2/ds2/ds2Val.csv')
        _range = [0.01, 0.001, 0.1, 1, 10, 100, 1000]
        for index in _range:
                _clf = MultinomialNB(alpha=index)
                _clf.fit(train_features, train_labels )
                pred = _clf.predict(validation_features)
                print('BernoulliNB accuracy ' + str(index) + ' is', accuracy_score(validation_labels, pred))


load_naive_bayes_multinomial()
# load_naive_bayes_multinomial()