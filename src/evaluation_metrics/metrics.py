from pandas import DataFrame
from numpy import divide, zeros_like
from .cython_modules.eval_metrics import IoU
from sqlite3 import connect
from collections import defaultdict
import json
import pickle
import sys


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


class METRICS:

    # ###########################################################################################
    # ############################ calculate confusion matrix ###################################
    # ###########################################################################################
    @staticmethod
    def _truePositive(matrix, iou_threshold):

        '''
        calculates the true positive value from a confusion matrix
        matrix: dataframe
        '''
        tp = 0
        tps = list()
        gt = matrix["Ground Truth"].tolist()
        pred = matrix["IOU>{}".format(iou_threshold)].tolist()
        for i in range(len(pred)):
            if (pred[i] == gt[i] == 1):
                tp += 1
                tps.append(tp)
            else:
                tps.append(tp)
        return tps

    @staticmethod
    def _falsePositive(matrix, iou_threshold):
        '''
        calculates the false positive value from a confusion matrix
        matrix: dataframe
        '''
        fp = 0
        fps = list()
        gt = matrix["Ground Truth"].tolist()
        pred = matrix["IOU>{}".format(iou_threshold)].tolist()
        for i in range(len(pred)):
            if (pred[i] == 1) and (pred[i] != gt[i]):
                fp += 1
                # print("False N ->", fp)
                fps.append(fp)
            else:
                fps.append(fp)
        return fps

    @staticmethod
    def _falseNegative(matrix, iou_threshold):
        """
        calculates the false negative value from a confusion matrix
        matrix: dataframe
        """
        fn = 0
        fns = list()
        gt = matrix["Ground Truth"].tolist()
        pred = matrix["IOU>{}".format(iou_threshold)].tolist()
        for i in range(len(pred)):
            if (pred[i] == 0) and (pred[i] != gt[i]):
                fn += 1
                fns.append(fn)
            else:
                fns.append(fn)
        return fns

    # ##########################################################################################
    # ################################# calculate IoU  #########################################
    # ##########################################################################################

    @staticmethod
    def _bb_intersection_over_union(boxA, boxB, name=None):
        '''
        takes two bounding boxes (ground truth and predicted values) and calculates a IoU
        '''
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min((boxA[2] + boxA[0]), (boxB[2] + boxB[0]))
        yB = min((boxA[3] + boxA[1]), (boxB[3] + boxB[1]))

        # compute the area of intersection rectangle
        interArea = max(abs(xB - xA), 0) * max(abs(yB - yA), 0)
        if (interArea == 0):
            return 0.0

        # compute the area of both the prediction and ground-truth rectangles
        boxAArea = abs(((boxA[2] + boxA[0]) - boxA[0]) * ((boxA[3] + boxA[1]) - boxA[1]))
        boxBArea = abs(((boxB[2] + boxB[0]) - boxB[0]) * ((boxB[3] + boxB[1]) - boxB[1]))

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth areas - the intersection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        if (iou < 0 or iou > 1):
            return 0.0
        return iou


class EVALUATION_METRICS(METRICS):

    def __init__(self, file_path):
        self._data = defaultdict(list)
        super().__init__()
        path = file_path.strip().split(".")[-1]
        if (path == "pickle"):
            self._prepare_pickel_data(file_path)

        elif (path == "json"):
            self._prepare_json_data(file_path)

    def _load_data(self, file_path):
        ext = file_path.strip().split(".")[-1]
        if (ext == "pickle"):
            with open(file_path, "rb") as file:
                data = pickle.load(file)
        elif (ext == "json"):
            with open(file_path, "r") as file:
                data = json.load(file)
        else:
            print("das Format ist kein gültiges Format!")
            sys.exit()
        return data

    def _prepare_pickel_data(self, file_path):
        _raw_data = self._load_data(file_path)
        keys = list(_raw_data.keys())
        self._predictions = []
        self._ground_truth = []
        self._names = []
        self._gt = []
        self.conf_level = _raw_data[keys[0]].confidence

        for key in keys:
            self._predictions.append(_raw_data[key].predictions)
            self._ground_truth.append(_raw_data[key].ground_truth)
            self._gt.append(_raw_data[key].gt)
            self._names.append(_raw_data[key].name)

    def _prepare_json_data(self, file_path):
        _raw_data = self._load_data(file_path)
        keys = _raw_data.keys()
        self._predictions = []
        self._ground_truth = []
        self._names = []
        self._gt = []
        self.conf_level = _raw_data[keys[0]][4]

        for key in keys:
            self._predictions.append(_raw_data[key][1])
            self._ground_truth.append(_raw_data[key][2])
            self._gt.append(_raw_data[keys[0]][3])
            self._names.append(_raw_data[key][0])

    # ##########################################################################################
    # ############################# calculate IoU ########################################
    # ##########################################################################################
    @staticmethod
    def _select_IOU_value(row, threshold=0.5):
        """
        classifies the values of a row according to threshold value
        row: header of the column to classify
        threshold: floating value to use as a reference
        """
        if (row["IOU"] > threshold):
            return 1  # "YES"
        else:
            return 0  # 'NO'

    def _calculate_iou(self):
        assert (len(self._predictions) > 0), "Predictions ist leer"
        assert (len(self._predictions) == len(self._ground_truth) == len(self._names)), "irgendwas lief mit der Formatumwandlung schief!"
        _predictions = [pred.tolist() if pred is not None else None for pred in self._predictions]
        _ground_truth = [gt.tolist() if gt is not None else None for gt in self._ground_truth]
        iou = IoU(_predictions, _ground_truth, list(b.encode("utf-8") for b in self._names))
        _gt, _data = iou.pymetrics()

        self.df = DataFrame.from_dict(_data, orient="index", columns=['IOU'])
        self.df["Ground Truth"] = list(_gt.values())

    def IoU(self,):
        self._calculate_iou()
        print(self.df.tail(60))

    # ##########################################################################################
    # ############################# calculate precision ########################################
    # ##########################################################################################

    @staticmethod
    def _calculate_precision(matrix):
        """
        takes a confusion matrix as dataframe and calculates the precision or recall metric
        matrix: dataframe
        """
        TP = matrix["TP"].to_numpy()
        FP = matrix["FP"].to_numpy()
        b = TP + FP
        return divide(TP.astype("float64"), b.astype("float64"), out=zeros_like(TP.astype("float64")), where=b != 0)

    def precision(self, iou_threshold):
        self._calculate_iou()
        self.df["IOU>{}".format(iou_threshold)] = self.df.apply(self._select_IOU_value, args=[iou_threshold], axis=1)
        self.df["TP"] = self._truePositive(self.df, iou_threshold)
        self.df["FP"] = self._falsePositive(self.df, iou_threshold)
        self.df["FN"] = self._falseNegative(self.df, iou_threshold)
        self.df["Precision"] = self._calculate_precision(self.df)

        print(self.df.tail(60))

    # ##########################################################################################
    # ################################ calculate recall ########################################
    # ##########################################################################################

    @staticmethod
    def _calculate_recall(matrix):
        """
                takes a confusion matrix as dataframe and calculates the precision or recall metric
                matrix: dataframe
        """
        TP = matrix["TP"].to_numpy()
        FN = matrix["FN"].to_numpy()
        b = TP + FN
        return divide(TP.astype("float64"), b.astype("float64"), out=zeros_like(TP.astype("float64")), where=b != 0)

    def recall(self, iou_threshold):
        self._calculate_iou()
        self.df["IOU>{}".format(iou_threshold)] = self.df.apply(self._select_IOU_value, args=[iou_threshold], axis=1)
        self.df["TP"] = self._truePositive(self.df, iou_threshold)
        self.df["FP"] = self._falsePositive(self.df, iou_threshold)
        self.df["FN"] = self._falseNegative(self.df, iou_threshold)
        self.df["Recall"] = self._calculate_recall(self.df)
        print(self.df.tail(60))

    # ##########################################################################################
    # ############################# Create a data frame ########################################
    # ##########################################################################################

    def make_dataframe(self, confidence):
        '''
        create a dataframe with the predicted and ground truth coordinates
        '''
        assert (len(self._predictions) > 0), "Predictions ist leer"
        assert (len(self._predictions) == len(self._ground_truth) == len(
            self._names)), "irgendwas lief mit der Formatumwandlung schief!"

        _predictions = [pred.tolist() if pred is not None else None for pred in self._predictions]
        _ground_truth = [pred.tolist() if pred is not None else None for pred in self._ground_truth]
        iou = IoU(_predictions, _ground_truth, list(b.encode("utf-8") for b in self._names))
        gt, data = iou.pymetrics()

        self.df = DataFrame.from_dict(data, orient="index", columns=['IOU'])
        self.df["Ground Truth"] = list(gt.values())
        self.df["IOU>{}".format(confidence)] = self.df.apply(self._select_IOU_value, axis=1)
        self.df["TP"] = self._truePositive(self.df, confidence)
        self.df["FP"] = self._falsePositive(self.df, confidence)
        self.df["FN"] = self._falseNegative(self.df, confidence)
        self.df["Precision"] = self._calculate_precision(self.df)
        self.df["Recall"] = self._calculate_recall(self.df)
        print(self.df.tail(60))

    def saveCSV(self, filename):

        self.df.to_csv(filename, index=True)

    def create_databanK(self, filename, confidence):
        '''
                saves a dataframe on a data bank from outside the class
                filename: name of the file as string
        '''

        cn = connect(filename)
        cursor = cn.cursor()
        cols = list(self.df.columns)
        conf = "IOU>{}".format(confidence)
        query = "CREATE TABLE IF NOT EXISTS computation_table" + \
                "(File TEXT, {} DECIMAL, {} INTEGER, {} INTEGER,{} INTEGER," \
                "{} INTEGER, {} INTEGER, {} DECIMAL, {} DECIMAL)".format(
                    cols[0], cols[1].replace("Ground Truth", "GT"), cols[2].replace(conf, "Lt"), cols[3],
                    cols[4], cols[5], cols[6], cols[7]
                )
        cursor.execute(query)
        cn.commit()
        self.df.to_sql(filename, cn, index=False)

    def get_IoU(self):
        try:
            return self.df["IOU"].tolist()
        except KeyError as e:
            print("[Infor]: ", e)
            print(" Zeile wurde noch nicht erstellt. Führen Sie bitte 'EVALUATION_METRICS().IoU()' zuerst aus ")
            return None

    def get_precision(self):
        try:
            return self.df["Precision"].tolist()
        except KeyError as e:
            print("[Infor]: ", e)
            print(" Zeile wurde noch nicht erstellt. Führen Sie bitte 'EVALUATION_METRICS().precision()' zuerst aus ")
            return None

    def get_recall(self):
        try:
            return self.df["Recall"].tolist()
        except KeyError as e:
            print("[Infor]: ", e)
            print(" Zeile wurde noch nicht erstellt. Führen Sie bitte 'EVALUATION_METRICS().recall()' zuerst aus ")
            return None

    def get_index_values(self):
        try:
            return self.df.index.values.tolist()
        except KeyError as e:
            print("[Infor]: ", e)
            print(" Zeile wurde noch nicht erstellt. Führen Sie bitte 'EVALUATION_METRICS().IoU()' zuerst aus ")
            return None


if (__name__ == "__main__"):
    ev = EVALUATION_METRICS("../../tests/data/output_data.pickle")
    ev.IoU()
    ev.precision(0.6)
    ev.recall(0.7)
    ev.make_dataframe(0.8)
    ev.create_databanK("./data.db", 0.8)
