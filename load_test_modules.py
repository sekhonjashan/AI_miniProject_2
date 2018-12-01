import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.extend(["../"])
sys.path.extend(["."])
from dataset1.service import service
import pickle

def init_classifier_models(classifier_data_url , test_data_url, fileName):
    with open(classifier_data_url, 'rb') as file:
        classifier = pickle.load(file)
    _service = service()
    _test_data = _service.read_csv_data_without_labels(test_data_url)
    _val_pred = classifier.predict(_test_data)
    _service.save_csv_result(_val_pred, fileName, 'output_csvs')


#   classifier with daa sets 1 && 2
init_classifier_models("dataset1/models/ds1Classifier-dt.pkl", "dataset1/ds1/ds1Test.csv", 'ds1Test-dt.csv')
init_classifier_models("dataset2/models/ds2Classifier-dt.pkl", "dataset2/ds2/ds2Test.csv", 'ds2Test-dt.csv')

init_classifier_models("dataset1/models/ds1Classifier-nb.pkl", "dataset1/ds1/ds1Test.csv", 'ds1Test-nb.csv')
init_classifier_models("dataset2/models/ds2Classifier-nb.pkl", "dataset2/ds2/ds2Test.csv", 'ds2Test-nb.csv')

init_classifier_models("dataset1/models/ds1Classifier-3.pkl", "dataset1/ds1/ds1Test.csv", 'ds1Test-3.csv')
init_classifier_models("dataset2/models/ds2Classifier-3.pkl", "dataset2/ds2/ds2Test.csv", 'ds2Test-3.csv')