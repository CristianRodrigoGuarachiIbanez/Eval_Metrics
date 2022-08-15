
from libcpp.map cimport map
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.utility cimport pair
from libcpp.cast cimport static_cast
from cython.operator cimport dereference, postincrement
from cython import boundscheck, wraparound, cdivision
from numpy cimport ndarray, int32_t, int64_t
from numpy import amax, asarray, int32, int64


cdef extern from "<utility>" namespace "std" nogil:
    pair[T,U] make_pair[T,U](T&,U&)

cdef class IoU:
    cdef:
        map[string, vector[float]] output
        map[string, vector[int]] gt_output
        # vector[pair[string, float]] output
    def __cinit__(self, list bboxes,  list gt_bboxes, list names):

        self.IoU(bboxes, gt_bboxes, names)

    @boundscheck(False)
    @wraparound(False)
    @cdivision(True)
    cdef inline float _bb_intersection_over_union(self, int[:] boxA, int[:] boxB):
        '''
        takes two bounding boxes (ground truth and predicted values) and calculates a IoU
        '''
        # determine the (x, y)-coordinates of the intersection rectangle
        cdef:
            int xA, yA, xB, yB, interArea, boxAArea, boxBArea
            float iou, bbAAreaFloat, bbBAreaFloat, interAreaFloat
            int * _iou
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min((boxA[2] + boxA[0]), (boxB[2] + boxB[0]))
        yB = min((boxA[3] + boxA[1]), (boxB[3] + boxB[1]))

        # compute the area of intersection rectangle
        interArea = max(abs(xB - xA),0) * max(abs(yB - yA),0) # max(0, xB - xA + 1) * max(0, yB - yA + 1)  #
        if (interArea == 0):
            return 0.0

        # compute the area of both the prediction and ground-truth rectangles
        boxAArea = abs(((boxA[2] + boxA[0]) - boxA[0]) * ((boxA[3] + boxA[1]) - boxA[1]))
        boxBArea = abs(((boxB[2] + boxB[0]) - boxB[0]) * ((boxB[3] + boxB[1]) - boxB[1]))

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth areas - the intersection area
        bbAAreaFloat = <float>(boxAArea)
        bbBAreaFloat = <float>(boxBArea)
        interAreaFloat = <float>(interArea)

        iou = ((interArea) /((bbAAreaFloat + bbBAreaFloat) - interAreaFloat))

        # return the intersection over union value
        if (iou < 0.0) or (iou > 1.0):
            return 0.0
        return iou

    @boundscheck(False)
    @wraparound(False)
    @cdivision(True)
    cdef inline void _calculate_iou(self, list bboxes, list gt_bboxes, string name):

        cdef:
            int i, j
            int length_gt = len(gt_bboxes) if gt_bboxes is not None else 0
            int length_bb = len(bboxes) if bboxes is not None else 0
            float iou = 0.0
            vector[float] value
            vector[int] gt_value
        if (gt_bboxes is None) or (length_gt == 0):
            value.push_back(0.0)
            self.output.insert([name, value])
            value.clear()

            gt_value.push_back(0)
            self.gt_output.insert([name, gt_value])
            gt_value.clear()
        else:
            # print("len -> predictions {}, Ground Truth {}".format(length_bb, length_gt))
            if (length_gt > 1):
                for i in range(length_gt):
                    for j in range(length_bb):
                        try:
                            iou = self._bb_intersection_over_union(asarray(gt_bboxes[i], dtype=int32), asarray(bboxes[j], dtype=int32))
                            # print("iou result  ->", iou)
                        except IndexError as e:
                            print("Info[INDEX ERROR] : ", e)
                            iou = 0.0
                        if (iou >= 0.0 and iou <= 1.0):
                            value.push_back(iou)
                            self.output[name] = value

                            gt_value.push_back(1)
                            self.gt_output[name] = gt_value

                value.clear()
                gt_value.clear()
            else:
                if (len(bboxes)!=0 or bboxes is None):
                    iou = self._bb_intersection_over_union(asarray(gt_bboxes[0], dtype=int32), asarray(bboxes[0], dtype=int32))
                    value.push_back(iou)
                    self.output[name] = value
                    value.clear()

                    gt_value.push_back(1)
                    self.gt_output[name] = gt_value
                    gt_value.clear()

    @boundscheck(False)
    @wraparound(False)
    @cdivision(True)
    cdef void IoU(self, list bboxes,  list gt_bboxes, list names):
        assert (len(bboxes) > 0), "Predictions ist leer"
        assert (len(bboxes) == len(gt_bboxes) == len(names)), "irgendwas lief mit der Formatumwandlung schief!"
        cdef int i
        for i in range(len(bboxes)):
            self._calculate_iou(bboxes[i], gt_bboxes[i], names[i])


    @boundscheck(False)
    @wraparound(False)
    @cdivision(True)
    cdef dict iou(self):
        cdef:
            int j
            vector[float] prediction
            string name
            map[string, vector[float]].iterator iter = self.output.begin()

        data = {}
        while (iter != self.output.end()):
            prediction = dereference(iter).second
            name = dereference(iter).first
            for j in range(prediction.size()):
                new_key = name + b"_" + bytes(str(j), "utf-8")
                data[new_key] = prediction[j]
            postincrement(iter)

        return data

    @boundscheck(False)
    @wraparound(False)
    @cdivision(True)
    cdef dict ground_truth(self):
        cdef:
            int j
            vector[int] gt
            string name
            map[string, vector[int]].iterator iter = self.gt_output.begin()

        grTr = {}
        while (iter != self.gt_output.end()):
            gt = dereference(iter).second
            name = dereference(iter).first
            for j in range(gt.size()):
                new_key = name + b"_" + bytes(str(j), "utf-8")
                grTr[new_key] = gt[j]
            postincrement(iter)
        return  grTr

    def pymetrics(self):
        return self.ground_truth(), self.iou()