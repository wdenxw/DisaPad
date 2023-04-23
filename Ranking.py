import numpy as np
import os
import axfunction as S
import pandas as pd
import datetime as dt
import constants as C

from settings import settings

st = settings()
args, unknown = st.parser.parse_known_args()


class ranking:
    """
    Preprocess data for ranking and return data after ranking.
        """

    def Ypredeal(self, type):
        """Load label information.

            Args:
                type: dataset name

            Returns:
                npfile: label information of the dataset

            Raises:
                None
            """
        npfile = np.load(C.file_dir + type + str(C.seq) + '-t' + str(C.window) + '.npy')
        npfile = S.totaldy(npfile)
        return npfile

    def Xpredeal(self, typee, changedim, istimese):
        """Load data, get entropy and rank sensor readings according to the entropy.

            Args:
                typee: dataset name
                changedim: whether to rank sensor readings according to the indicator.
                istimese: is the current input time series data?

            Returns:
                npfile: Ranked sensor readings
                rawx: sensor readings, motion information and displacement information of the input dataset

            Raises:
                None
            """
        Columx = C.ColumSuSmx
        npfile = np.load(C.file_dir + typee + str(C.seq) + '-t' + str(C.window) + '.npy')
        rankindicator = args.rankindicator

        rawx = npfile
        if changedim:
            totalname, totalindicator = self.genindicator(typee, npfile, Columx, rankindicator)
            if typee == 'SuSmX_train':
                print(npfile[0, 0, :])
                S.getindex(typee, npfile, totalindicator, totalname)
            npfile = self.adjustpos(typee, npfile, totalindicator, totalname)
        npfile = S.totaldx(npfile, typee, istimese)
        return npfile, rawx

    def predealtrain(self, changedim, istimese):
        """Return ranked sensor readings and its labels.

            Args:
                changedim: whether to rank sensor readings according to the indicator.
                istimese: is the current input time series data?

            Returns:
                SuSmX_train: ranked sensor readings for training the SuSm dataset
                SuSmY_train: sensor labels for training the SuSm dataset

            Raises:
                None
            """
        SuSmX_train, _ = self.Xpredeal('SuSmX_train', changedim, istimese)
        SuSmY_train = self.Ypredeal('SuSmY_train')
        return SuSmX_train, SuSmY_train

    def predealteva(self, changedim, istimese):
        """Return ranked sensor readings and its labels.

            Args:
                changedim: whether to rank sensor readings according to the indicator.
                istimese: is the current input time series data?

            Returns:
                SuSmX_test: ranked sensor readings for testing
                SuSmY_test: sensor labels for testing
                SuSmX_validate: ranked sensor readings for validation
                SuSmY_validate: sensor labels for validation

            Raises:
                None
            """
        SuSmX_test, rawxtest = self.Xpredeal('SuSmX_test', changedim, istimese)
        SuSmX_validate, _ = self.Xpredeal('SuSmX_valid', changedim, istimese)
        SuSmY_test = self.Ypredeal('SuSmY_test')
        SuSmY_validate = self.Ypredeal('SuSmY_valid')
        return SuSmX_test, SuSmY_test, SuSmX_validate, SuSmY_validate

    def rankusmo(self, dtype, rankindicator, npfile, Colum):
        """Return ranked sensor readings and its labels.

            Args:
                dtype: dataset name
                rankindicator: which indicator is considered for ranking sensor readings
                npfile: the target domain data before ranking
                Colum: the colum name of data

            Returns:
                npfile: ranked data

            Raises:
                ValueError: If the input does not match any dataset.
            """
        starttime = dt.datetime.now()
        totalname, totaljitter = self.genindicator(dtype, npfile, Colum, rankindicator)
        if dtype == 'MuMm' or dtype == 'SuMm':
            if not (os.path.exists(C.file_dir + dtype + 'totalname.npy')):
                np.save(C.file_dir + dtype + 'totalname.npy', totalname)
                np.save(C.file_dir + dtype + 'totalentropy.npy', totaljitter)
            else:
                nname = np.load(C.file_dir + dtype + 'totalname.npy')
                eentropy = np.load(C.file_dir + dtype + 'totalentropy.npy')
                nname = np.concatenate((nname, totalname), axis=0)
                eentropy = np.concatenate((eentropy, totaljitter), axis=0)
                np.save(C.file_dir + dtype + 'totalname.npy', nname)
                np.save(C.file_dir + dtype + 'totalentropy.npy', eentropy)
        npfile = self.adjustpos(dtype, npfile, totaljitter, totalname)
        endtime = dt.datetime.now()
        print("ranking time", endtime - starttime, (endtime - starttime) / npfile.shape[0])
        return npfile

    def adjustpos(self, dtype, xdata, totalindicator, totalname):
        """Arrange different sensor readings according to indicators (entropy, jitter, std).

            Args:
                dtype: dataset name
                xdata: sensor readings, user information, motion information and
                displacement information of the current dataset
                totalindicator: the indicator value (entropy, jitter, std) of all displacements of
                the current dataset
                totalname: the displacement name of all displacements of the current dataset
            Returns:
                xdata: sensor readings after ranked

            Raises:
                ValueError: If the input does not match any dataset.
            """
        # If it is the multi-user multi-motion dataset
        if 'SuSm' in dtype:
            columname = C.ColumSuSmx
            for i in range(len(totalname)):
                currentrank = [0] + list(totalindicator[i]) + [7, 8, 9]  # new order of sensor readings
                currentindex = np.argwhere(xdata[:, 0, columname.index('rawname')] == totalname[i])[:, 0]
                for j in currentindex:
                    uuu = xdata[j]
                    xdata[j] = uuu[:, currentrank]
        # If it is the multi-user multi-motion dataset
        elif dtype == 'MuMm':
            columname = C.ColumMuMmx
            for i in range(len(totalname)):
                currentrank = np.array([0] + list(totalindicator[i]) + [7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
                currentindex = np.argwhere(xdata[:, 0, int(columname.index('rawname'))] == totalname[i])[:, 0]
                for j in currentindex:
                    uuu = xdata[j]
                    xdata[j] = uuu[:, currentrank]
        # If it is the single-user multi-motion dataset
        elif dtype == 'SuMm':
            columname = C.ColumSuMmx
            for i in range(len(totalname)):
                currentrank = np.array([0] + list(totalindicator[i]) + [7, 8, 9, 10])
                currentindex = np.argwhere(xdata[:, 0, columname.index('rawname')] == totalname[i])[:, 0]
                for j in currentindex:
                    uuu = xdata[j]
                    xdata[j] = uuu[:, currentrank]
        else:
            raise ValueError("The input does not match any dataset. The current input type:", dtype)
        return xdata

    def genindicator(self, dtype, xdata, Columx, rankindicator):
        """Compute indicator value (entropy, jitter, std) of all displacements of the current dataset.

            Args:
                dtype: dataset name
                xdata: sensor readings, user information, motion information and
                displacement information of the current dataset
                Columx: column name of the current dataset
                rankindicator: which indicator is considered for ranking
                sensor readings
            Returns:
                totalname: the displacement name of all displacements of the current dataset
                totalindicator: the indicator value (entropy, jitter, std) of all displacements of
                the current dataset

            Raises:
                ValueError: If the input does not match any dataset.
            """

        print("current type:", dtype)
        totalname = []
        totalindicator = []
        if 'SuSm' in dtype:
            for i in range(0, 70, 1):
                for name in ['CZP_' + str(i) + '_down4', 'CZP_' + str(i) + '_down3', 'CZP_' + str(i) + '_down2'
                    , 'CZP_' + str(i) + '_down1', 'CZP_' + str(i) + '_0', 'CZP_' + str(i) + '_up1'
                    , 'CZP_' + str(i) + '_up2', 'CZP_' + str(i) + '_up3', 'CZP_' + str(i) + '_up4']:
                    npp = np.argwhere(xdata[:, 0, Columx.index('rawname')] == name)[:, 0]
                    if npp.shape[0] != 0:
                        totalname.append(name)
                        pauseindicator = self.doindicomp(xdata[npp, :, :], 'SuSm', rankindicator)
                        totalindicator.append(pauseindicator)
        elif dtype == 'MuMm':
            totalname = list(set(list(xdata[:, 0, Columx.index('rawname')])))
            totalname = list(set(totalname))
            for name in range(len(totalname)):
                npp = np.argwhere(xdata[:, 0, Columx.index('rawname')] == totalname[name])[:, 0]
                pauseindicator = self.doindicomp(xdata[npp, :, :], 'MuMm', rankindicator)
                totalindicator.append(pauseindicator)
        elif dtype == 'SuMm':
            totalname = list(set(list(xdata[:, 0, Columx.index('rawname')])))
            totalname = list(set(totalname))
            for name in range(len(totalname)):
                npp = np.argwhere(xdata[:, 0, Columx.index('rawname')] == totalname[name])[:, 0]
                pauseindicator = self.doindicomp(xdata[npp, :, :], 'SuMm', rankindicator)
                totalindicator.append(pauseindicator)
        else:
            raise ValueError("The input does not match any dataset. The current input type:", dtype)
        return totalname, totalindicator

    def doindicomp(self, npa, dtype, rankindicator):
        """Return computing results of indicators.

        Return computing results of indicators (entropy, jitter, std) of one displacement.

            Args:
                npa: sensor readings of current data
                dtype: dataset name
                rankindicator: which indicator is considered for ranking sensor readings

            Returns:
                sortedid: The ranking result of the six sensor readings of one displacement

            Raises:
                None
            """
        if dtype == 'MuMm':
            Columx = ['time', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'username', 'circular',
                      'lateral', 'motion', 'rawname', 'upperc', 'lowerc', 'height', 'weight', 'age']
        elif dtype == 'SuSm':
            Columx = ['time', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'upperc', 'lowerc', 'height', 'weight', 'age']
        elif dtype == 'SuMm':
            Columx = ['time', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'circular', 'lateral', 'motion', 'rawname']
        elif dtype == 'dealed':
            Columx = ['r1', 'r2', 'r3', 'r4', 'r5', 'r6']
        else:
            raise ValueError("Type error. The input does not match any dataset. Current input:", dtype)
        npa = npa[:, 0, :]

        y1 = self.dealfunc(npa[:, Columx.index('r1')].astype(float), rankindicator)
        y2 = self.dealfunc(npa[:, Columx.index('r2')].astype(float), rankindicator)
        y3 = self.dealfunc(npa[:, Columx.index('r3')].astype(float), rankindicator)
        y4 = self.dealfunc(npa[:, Columx.index('r4')].astype(float), rankindicator)
        y5 = self.dealfunc(npa[:, Columx.index('r5')].astype(float), rankindicator)
        y6 = self.dealfunc(npa[:, Columx.index('r6')].astype(float), rankindicator)
        listall = [y1, y2, y3, y4, y5, y6]
        sortedid = sorted(range(len(listall)), key=lambda k: listall[k], reverse=False)
        sortedid = list(map(lambda x: x + 1, sortedid))
        return sortedid

    def dealfunc(self, input, rankindicator):
        """Return indicator values of input sensor readings.

            Args:
                input: sensor readings
                rankindicator: which indicator (entropy, jitter, std) is considered for ranking
                sensor readings

            Returns:
                output: indicator values of input sensor readings.

            Raises:
                None
            """
        input1 = self.deleoutlier(input)
        if len(input1) == 0:
            input = input
        else:
            input = input1
        input = list(S.ToNormalize(np.array(input)))
        if rankindicator == 'entropy':
            output = self.Fuzzy_Entropy(input, args.m, args.r)
        elif rankindicator == 'std':
            output = self.stdstd(input)
        elif rankindicator == 'jitter':
            output = self.caljitter(input)
        else:
            raise ValueError("The input does not match any existing inidcator. The current input indicator:",
                             rankindicator)
        return output

    def Fuzzy_Entropy(self, x, m, r):
        """Only keep label information.

            Args:
                x: the whole dataset including user information, displacement
                information and labels.
                m: window size
                r: standard deviation of the original time series. The value range is 0.1-0.25

            Returns:
                entropy: the computed entropy value

            Raises:
                ValueError: if the dimension of the input data is not 1d
            """
        x = np.array(x)
        # Check if x is one-dimensional data
        if x.ndim != 1:
            raise ValueError("The dimension of x is not one dimension")
        # Calculate whether the number of rows in x is less than m plus 1
        if len(x) < m + 1:
            raise ValueError("len(x) is less than m+1")
        # Divide x with m as the window
        entropy = 0
        for temp in range(0, 2):
            X = []
            for i in range(0, len(x) - m + 1 - temp, 3):
                X.append(x[i:i + m + temp])
            X = np.array(X)
            # Calculate the maximum value of the absolute difference of data of
            # corresponding index to any row of X and other rows of X
            D_value = []  # save the differences
            for index1, i in enumerate(X):
                sub = []
                for index2, j in enumerate(X):
                    if index1 != index2:
                        sub.append(max(np.abs(i - j)))
                D_value.append(sub)
            # Calculate fuzzy membership.
            D = np.exp(-np.power(D_value, 2) / r)
            # Calculate the average of all membership degrees.
            Lm = np.average(D.ravel())
            entropy = abs(entropy) - Lm
        return entropy

    def compute_updownlimit(self, nums):
        """compute the uplimit and the down limit of boxplot.

        Calculate the upper and lower limits based on the quartile

        Args:
            nums: input total array list.

        Returns:
            maxvalue: the upper limits of outliers.
            minvalue: the lower limits of outliers.

        Raises:
            None
        """
        nums = nums.astype(float)
        df_nums = pd.Series(nums)
        dfa = df_nums.describe()
        quartile_three = dfa.iloc[6]
        quartile_one = dfa.iloc[4]
        iqr = quartile_three - quartile_one
        upper_limit = quartile_three + 1.5 * iqr
        down_limit = quartile_one - 1.5 * iqr
        upper = []
        lower = []
        for i in range(len(nums)):
            if nums[i] < upper_limit:
                upper.append(nums[i])
            if nums[i] > down_limit:
                lower.append(nums[i])
        if len(upper) == 0:
            maxvalue = np.max(nums)
        else:
            maxvalue = np.max(upper)
        if len(lower) == 0:
            minvalue = np.min(nums)
        else:
            minvalue = np.min(lower)
        return maxvalue, minvalue

    def stdstd(self, y1):
        """compute standard deviation of input data.

        Calculate standard deviation of input data.

        Args:
            y1:

        Returns:
            output: standard deviation of input data

        Raises:
            None
        """
        output = np.std(y1)
        return output

    def caljitter(self, sensorr):
        """Calculate the jitter value of input sensor readings.


        Args:
            sensorr: input sensor readings

        Returns:
            output: the average jitter value of current sensor readings

        Raises:
            None
        """
        yy1 = []
        # the first derivative
        for i in range(len(sensorr) - 1):
            yy1.append(abs(sensorr[i + 1] - sensorr[i]))
        yyy1 = []
        # the second derivative
        for i in range(len(yy1) - 1):
            yyy1.append(abs(yy1[i + 1] - yy1[i]))
        yyyy1 = []
        # the third derivative
        for i in range(len(yyy1) - 1):
            yyyy1.append(abs(yyy1[i + 1] - yyy1[i]))
        output = np.average(yyyy1)
        return output

    def is_noise(self, sensorr, upper_limit, down_limit):
        """Judge whether the current sensor reading is outlier.

        Args:
            sensorr: one sensor reading
            upper_limit: upper limit of the boxplot of current sensor readings
            down_limit: down limit of the boxplot of current sensor readings

        Returns:
            True: this sensor reading is noise.
            False: this sensor reading is not noise.

        Raises:
            None
        """
        if sensorr >= upper_limit or sensorr <= down_limit:
            return True
        return False

    def deleoutlier(self, y1):
        """Removed the numeric outliers with boxplot.

            Args:
                y1: single sensor readings of one displacement

            Returns:
                output: single sensor readings of one displacement removing outliers

            Raises:
                None
            """
        upper_limit, down_limit = self.compute_updownlimit(y1)
        NoiseGT = [self.is_noise(numa, upper_limit, down_limit) for numa in y1]
        delete = []
        for i in range(len(NoiseGT)):
            if NoiseGT[i] == True:
                delete.append(i)
        output = np.delete(y1, delete)
        return output
