from evaluation_metrics.metrics import EVALUATION_METRICS
from os import path
import pytest
file_path = path.dirname(path.abspath(__file__))
data_path = file_path + "/data/output_data.pickle"


class IMG_PREDICTIONS(object):

    def __init__(self, name, coord_predictions, coord_ground_truth, ground_truth, confidence):
        self.name = name
        if not (any(coord_predictions)):
            self.predictions = None
        else:
            self.predictions = coord_predictions

        if not (any(coord_ground_truth)):
            self.ground_truth = None
        else:
            self.ground_truth = coord_ground_truth
        self.confidence = confidence
        self.gt = ground_truth

    def getName(self):
        return self.name

    def getPredictions(self):
        return self.predictions

    def getCGT(self):
        return self.ground_truth

    def getConfidence(self):
        return self.confidence

    def getGT(self):
        return self.gt


@pytest.mark.parametrize("data_path, conf", [
    (data_path, 0.5), (data_path, 0.6)
])
def test_evaluation_metrics(data_path, conf ):
    EV = EVALUATION_METRICS(data_path)
    EV.IoU()
    EV.precision(conf)
    EV.recall(conf)
    EV.make_dataframe(conf)

if (__name__ == "__main__"):
    EV = EVALUATION_METRICS(data_path)
    conf = 0.5
    EV.IoU()
    EV.precision(conf)
    EV.recall(conf)
    EV.make_dataframe(conf)