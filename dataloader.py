"""
Load and preprocess data.
"""
import os
import numpy as np
import random
import axfunction as S
import pandas as pd
from Ranking import ranking
import constants as C
from constants import usernamel, motionnamel
from settings import settings

st = settings()
us = usernamel()
mo = motionnamel()
R = ranking()
args, unknown = st.parser.parse_known_args()


class dataloader:
    """
    Load, preprocess data and return data for training, testing and validation.
    """

    def loadtrain(self, userfortest, motionfortest, changedim, testTarget, shuffle,
                  istimese):
        """Assign data for training and validation.

            Args:
                userfortest: user's name of the target domain (when transfer to the multi-user
                multi-motion dataset)
                motionfortest: motion's name of the target domain (when transfer to the single-user
                multi-motion dataset)
                changedim: whether to rank sensor readings according to the indicator.
                testTarget: whether to transfer the model trained on the single-user single-motion
                dataset to the single-user multi-motion dataset or multi-user multi-motion dataset
                shuffle: whether to shuffle the source domain (single-user single-motion training set).
                istimese: is the current input time series data?

            Returns:
                SuSmX_train: sensor readings of single-user single-motion training set
                SuSmY_train: labels of single-user single-motion training set
                Y_tatrain: sensor readings of target training set
                X_tatrain: labels of target training set
                targettrainname: displacement name of target training set

            Raises:
                None
            """
        print("load train data:")
        lSuSm = SuSmloader()
        lMuMm = MuMmloader()
        lSuMm = SuMmloader()
        rankindicator = args.rankindicator
        SuSmX_train, SuSmY_train = lSuSm.loadSuSmTrain(changedim, istimese)
        targettrainname = 'None'
        if testTarget:
            if motionfortest == None:  # The target domain is multi-user multi-motion dataset
                if os.path.exists(C.file_dir + rankindicator + 'tousx.npy'):
                    tousx = np.load(C.file_dir + rankindicator + 'tousx.npy')
                    tousy = np.load(C.file_dir + rankindicator + 'tousy.npy')
                else:
                    tousx, tousy = self.loadcousmt(changedim, 'MuMm')
                    if rankindicator == 'entropy' and changedim:
                        np.save(C.file_dir + rankindicator + 'tousx.npy', tousx)
                        np.save(C.file_dir + rankindicator + 'tousy.npy', tousy)
                        print("save successs!!!!!")
                usrtestdataX, usrtestdataY, targettrainname = lMuMm.gettargetusrtestdata(tousx, tousy, userfortest,
                                                                                         )
                usrtrainX, usrtrainY = lMuMm.getusrtrain(tousx, tousy, userfortest, istimese)
                Y_tatrain = usrtrainY
                X_tatrain = usrtrainX
            elif userfortest == None:  # The target domain is single-user multi-motion dataset.
                if os.path.exists(C.file_dir + rankindicator + 'tomtx.npy') and changedim:
                    print(C.file_dir + rankindicator + 'tomtx.npy', "SuMm dataset exist!")
                    tomtx = np.load(C.file_dir + rankindicator + 'tomtx.npy')
                    tomty = np.load(C.file_dir + rankindicator + 'tomty.npy')
                else:
                    tomtx, tomty = self.loadcousmt(changedim, 'SuMm')
                    if rankindicator == 'entropy' and changedim:
                        np.save(C.file_dir + rankindicator + 'tomtx.npy', tomtx)
                        np.save(C.file_dir + rankindicator + 'tomty.npy', tomty)
                mttrainX, mttrainY, targettrainname = \
                    lSuMm.getmttr(tomtx, tomty, C.ColumSuMmx, motionfortest, istimese)
                Y_tatrain = mttrainY
                X_tatrain = mttrainX
        else:
            Y_tatrain = np.array([0])
            X_tatrain = np.array([0])
        if shuffle:  # Shuffle the training set.
            cc = list(zip(SuSmX_train, SuSmY_train))
            random.seed(10000000)
            random.shuffle(cc)
            SuSmX_train, SuSmY_train = zip(*cc)
        SuSmX_train = np.array(self.DoNormalize(np.array(SuSmX_train), '3d'))
        if testTarget:
            X_tatrain = self.DoNormalize(np.array(X_tatrain), '3d')
        SuSmY_train = np.array(SuSmY_train)
        return SuSmX_train, SuSmY_train, Y_tatrain, X_tatrain, targettrainname

    def loadteva(self, userfortest, motionfortest, changedim, testTarget, rankindicator,
                 istimese):
        """Assign data for training and validation.

            Args:
                userfortest: user's name of the target domain (when transfer to the multi-user
                multi-motion dataset)
                motionfortest: motion's name of the target domain (when transfer to the single-user
                multi-motion dataset)
                changedim: whether to rank sensor readings according to the indicator.
                testTarget: whether to transfer the model trained on the single-user single-motion
                dataset to the single-user multi-motion dataset or multi-user multi-motion dataset
                rankindicator: which indicator is considered for ranking sensor readings
                (option: entropy, std, jitter)
                istimese: is the current input time series data?

            Returns:
                SuSmX_test, SuSmY_test, SuSmX_validate, \
            SuSmY_validate, X_tatest, Y_tatest, rawxtest, X_tavalidate, Y_tavalidate
                SuSmX_test: sensor readings of single-user single-motion testing set
                SuSmY_test: labels of single-user single-motion testing set
                SuSmX_validate: sensor readings of single-user single-motion validation set
                SuSmY_validate: labels of single-user single-motion validation set
                X_tatest: sensor readings of target testing set
                Y_tatest: labels of target testing set
                rawxtest:sensor readings, user information, motion information and
                 displacement information of training set of Multi-user Multi-motion dataset
                X_tavalidate: sensor readings of target validation set
                Y_tavalidate: labels of target validation set

            Raises:
                None
            """
        lSuSm = SuSmloader()
        lMuMm = MuMmloader()
        lSuMm = SuMmloader()
        print("load test and validate data")
        SuSmX_test, SuSmY_test, SuSmX_validate, SuSmY_validate, rawxtest = lSuSm.loadSuSmTest(changedim, rankindicator,
                                                                                              istimese)
        if testTarget:
            print("testtarget")
            if motionfortest == None and userfortest != None:  # The target domain is multi-user multi-motion dataset
                print("enter user")
                if os.path.exists(C.file_dir + rankindicator + 'tousx.npy'):
                    print("user dataset exist")
                    tousx = np.load(C.file_dir + rankindicator + 'tousx.npy')
                    tousy = np.load(C.file_dir + rankindicator + 'tousy.npy')

                else:
                    tousx, tousy = self.loadcousmt(changedim, 'MuMm')
                    if rankindicator == 'entropy' and changedim:
                        np.save(C.file_dir + rankindicator + 'tousx.npy', tousx)
                        np.save(C.file_dir + rankindicator + 'tousy.npy', tousy)
                usrtestX, usrtestY, usrvaliX, usrvaliY = \
                    lMuMm.getusrteva(tousx, tousy, userfortest, istimese)
                X_tatest = usrtestX
                Y_tatest = usrtestY
                X_tavalidate = usrvaliX
                Y_tavalidate = usrvaliY
            elif userfortest == None and motionfortest != None:  # The target domain is single-user multi-motion dataset.
                print("enter motion", C.file_dir + rankindicator + 'tomtx.npy', changedim)
                if os.path.exists(C.file_dir + rankindicator + 'tomtx.npy') and changedim:
                    print("dataset exist!")
                    tomtx = np.load(C.file_dir + rankindicator + 'tomtx.npy')
                    tomty = np.load(C.file_dir + rankindicator + 'tomty.npy')
                else:
                    tomtx, tomty = self.loadcousmt(changedim, 'SuMm')  # motion的完全集
                    if rankindicator == 'entropy' and changedim:
                        np.save(C.file_dir + rankindicator + 'tomtx.npy', tomtx)
                        np.save(C.file_dir + rankindicator + 'tomty.npy', tomty)
                mttestX, mttestY, mtvaliX, mtvaliY = \
                    lSuMm.getmtteva(tomtx, tomty, C.ColumSuMmx, motionfortest, istimese)
                X_tatest = mttestX
                Y_tatest = mttestY
                X_tavalidate = mtvaliX
                Y_tavalidate = mtvaliY
            elif userfortest != None and motionfortest != None:
                raise ValueError("The model cannot transfer to both SuMm and MuMm together.")
        else:
            X_tatest = np.array([0])
            Y_tatest = np.array([0])
            X_tavalidate = np.array([0])
            Y_tavalidate = np.array([0])
        SuSmX_test = self.DoNormalize(np.array(SuSmX_test), '3d')
        SuSmX_validate = self.DoNormalize(np.array(SuSmX_validate), '3d')
        if testTarget:
            X_tatest = self.DoNormalize(np.array(X_tatest), '3d')
            X_tavalidate = self.DoNormalize(np.array(X_tavalidate), '3d')
        return SuSmX_test, SuSmY_test, SuSmX_validate, \
               SuSmY_validate, X_tatest, Y_tatest, rawxtest, X_tavalidate, Y_tavalidate

    def Xusmodeal(self, dtype, featurename, changedim):
        """Return ranked sensor readings and its labels.

            Args:
                dtype: dataset name
                featurename: If predicting SuMm, it's motion name. If predicting MuMm,
                it's user name.
                changedim: Sensor readings to be normalized

            Returns:
                npfile: ranked sensor readings of target domain data

            Raises:
                ValueError: If the input does not match any dataset.
            """
        rankindicator = args.rankindicator
        if dtype == 'MuMm':
            Colum = C.ColumMuMmx
            npfile = np.load(C.file_dir + 'X_' + featurename + str(C.seq) + '-t' + str(C.window) + '.npy')
        elif dtype == 'SuMm':
            Colum = C.ColumSuMmx
            npfile = np.load(C.file_dir + dtype + 'X_' + featurename + str(C.seq) + '-t' + str(C.window) + '.npy')
        else:
            raise ValueError("Type error. The input does not match any dataset. Current input:", dtype)

        if changedim:
            npfile = R.rankusmo(dtype, rankindicator, npfile, Colum)
        return npfile

    def Yusermodeal(self, dtype, featurename):
        """Load label information.

            Args:
                dtype: Predict SuMm (single-user multi-motion) dataset or MuMm
                (multi-user multi-motion) dataset.
                featurename: If predict SuMm, return motion name. If predict MuMm,
                return user name.

            Returns:
                npfile: label information of the target domain dataset

            Raises:
                None
            """
        if dtype == 'MuMm':
            npfile = np.load(C.file_dir + 'Y_' + featurename + str(C.seq) + '-t' + str(C.window) + '.npy')
        else:
            npfile = np.load(C.file_dir + dtype + 'Y_' + featurename + str(C.seq) + '-t' + str(C.window) + '.npy')
        npfile = S.totaldy(npfile)
        return npfile

    def DoNormalize(self, inputnpy, type):
        """Normalize sensor readings (the demension of sensor readings is more than 1).

            Args:
                inputArr: Sensor readings to be normalized
                type: It can be chosen from 2d or more than 2d (for example: 3d)

            Returns:
                outputnpy: Normalized sensor readings

            Raises:
                None
            """
        if type == '2d':
            for i in range(inputnpy.shape[1]):
                inputnpy[:, i] = S.ToNormalize(inputnpy[:, i])
        else:
            for i in range(inputnpy.shape[1]):
                inputnpy[:, i, :] = S.ToNormalize(inputnpy[:, i, :])
        outputnpy = inputnpy
        return outputnpy

    def loadcousmt(self, changedim, dtype):
        """Load and connect dataset of different users/motions.

            Args:
                changedim: whether to rank sensor readings according to the indicator.
                dtype: dataset name
                (multi-user multi-motion) dataset.

            Returns:
                touftx: sensor readings, ulser information, motion information and
                 displacement information of training set of target domain
                 (connected by different motion/users)
                toufty: label of target domain (connected by different motion/users)

            Raises:
                ValueError: If the input does not match any dataset.
            """
        if dtype == 'MuMm':  # It's MuMm dataset
            df = pd.read_csv(C.file_dir + 'multisj0417.csv')
            ndf = df.values
            featurename = list(set(ndf[:, list(df.columns).index('username')]))
        elif dtype == 'SuMm':  # It's SuMm dataset
            df = pd.read_csv(C.file_dir + 'multimt0417.csv')
            ndf = df.values
            featurename = list(set(ndf[:, list(df.columns).index('motion')]))
        else:
            raise ValueError("The input does not match any dataset.")

        for i in range(len(featurename)):  # Concatenate data of different users.
            print(featurename[i])
            Xft = self.Xusmodeal(dtype, featurename[i], changedim)
            Yft = self.Yusermodeal(dtype, featurename[i])
            if i == 0:
                touftx = Xft
                toufty = Yft
            else:
                touftx = np.concatenate((touftx, Xft), axis=0)
                toufty = np.concatenate((toufty, Yft), axis=0)
        return touftx, toufty


class MuMmloader:
    """
    Devide MuMm data for training, testing and validation.
    """

    def gettargetusrtestdata(self, tousx, tousy, userfortest):
        """Fetches single user data of three displacements for training, testing and validation.

            Fetches data belong to single user and assign train, test and validate datasets
             with data at three displacements.

            Args:
                tousx: Sensor readings, user information and motion information of
                multi-user multi-motion dataset
                tousy: Label (optical ground truth) of multi-user multi-motion dataset
                userfortest: User to be predicted


            Returns:
                netrainX: Sensor readings of training set. The training set only include
                single user and single displacement.
                netrainY: label of trainingset
                trainname: displacement name of training set

            Raises:
                None
            """
        Columuser = C.ColumMuMmx
        nutrindex, trainname = self.filterusdTrain(userfortest, tousx, Columuser)
        netrainX = tousx[nutrindex, :, :]
        netrainY = tousy[nutrindex, :]
        return netrainX, netrainY, trainname

    def filterusdTrain(self, username, tousx, Columuser):
        """Assign dataset for training.

            Args:
                username: User to be predicted
                tousx: Sensor readings, user information and motion information of
                multi-user multi-motion dataset
                Columuser: Column name of multi-user multi-motion dataset

            Returns:
                nutrindex: Index of training set. The training set only include
                single user and single displacement.
                name: displacement name of training set

            Raises:
                None
            """

        if username == 'night':
            name = us.night[0]
            nutrindex = list(np.argwhere(tousx[:, 0, Columuser.index('rawname')] == name)[:, 0])
        elif username == 'two':
            name = us.two[0]
            nutrindex = list(np.argwhere(tousx[:, 0, Columuser.index('rawname')] == name)[:, 0])
        elif username == 'ten':
            name = us.ten[0]
            nutrindex = list(np.argwhere(tousx[:, 0, Columuser.index('rawname')] == name)[:, 0])
        elif username == 'five':
            name = us.five[0]
            nutrindex = list(np.argwhere(tousx[:, 0, Columuser.index('rawname')] == name)[:, 0])
        elif username == 'seven':
            name = us.seven[0]
            nutrindex = list(np.argwhere(tousx[:, 0, Columuser.index('rawname')] == name)[:, 0])
        elif username == 'six':
            name = us.six[0]
            nutrindex = list(np.argwhere(tousx[:, 0, Columuser.index('rawname')] == name)[:, 0])
        elif username == 'four':
            name = us.four[0]
            nutrindex = list(np.argwhere(tousx[:, 0, Columuser.index('rawname')] == name)[:, 0])
        elif username == 'eleven':
            name = us.eleven[0]
            nutrindex = list(np.argwhere(tousx[:, 0, Columuser.index('rawname')] == name)[:, 0])
        elif username == 'three':
            name = us.three[0]
            nutrindex = list(np.argwhere(tousx[:, 0, Columuser.index('rawname')] == name)[:, 0])
        elif username == 'twelve':
            name = us.twelve[0]
            nutrindex = list(np.argwhere(tousx[:, 0, Columuser.index('rawname')] == name)[:, 0])
        elif username == 'eight':
            name = us.eight[0]
            nutrindex = list(np.argwhere(tousx[:, 0, Columuser.index('rawname')] == name)[:, 0])
        return nutrindex, name

    def filterusteva(self, userfortest, tousx, Columuser):
        """Assign dataset for testing.

            Args:
                userfortest: User to be predicted
                tousx: Sensor readings, user information and motion information of
                multi-user multi-motion dataset
                Columuser: Column name of multi-user multi-motion dataset

            Returns:
                teindex: Index of testing set. The testing set only include
                single user and single displacement.
                vaindex: Index of validation set. The validation set only include
                single user and single displacement.

            Raises:
                None
            """
        if userfortest == 'night':
            teindex = list(np.argwhere(tousx[:, 0, Columuser.index('rawname')] == us.night[1])[:, 0])
            vaindex = list(np.argwhere(tousx[:, 0, Columuser.index('rawname')] == us.night[2])[:, 0])
        elif userfortest == 'ten':
            teindex = list(np.argwhere(tousx[:, 0, Columuser.index('rawname')] == us.ten[1])[:, 0])
            vaindex = list(np.argwhere(tousx[:, 0, Columuser.index('rawname')] == us.ten[2])[:, 0])
        elif userfortest == 'two':
            teindex = list(np.argwhere(tousx[:, 0, Columuser.index('rawname')] == us.two[1])[:, 0])
            vaindex = list(np.argwhere(tousx[:, 0, Columuser.index('rawname')] == us.two[2])[:, 0])
        elif userfortest == 'five':
            teindex = list(np.argwhere(tousx[:, 0, Columuser.index('rawname')] == us.five[1])[:, 0])
            vaindex = list(np.argwhere(tousx[:, 0, Columuser.index('rawname')] == us.five[2])[:, 0])
        elif userfortest == 'seven':
            teindex = list(np.argwhere(tousx[:, 0, Columuser.index('rawname')] == us.seven[1])[:, 0])
            vaindex = list(np.argwhere(tousx[:, 0, Columuser.index('rawname')] == us.seven[2])[:, 0])
        elif userfortest == 'six':
            teindex = list(np.argwhere(tousx[:, 0, Columuser.index('rawname')] == us.six[1])[:, 0])
            vaindex = list(np.argwhere(tousx[:, 0, Columuser.index('rawname')] == us.six[2])[:, 0])
        elif userfortest == 'four':
            teindex = list(np.argwhere(tousx[:, 0, Columuser.index('rawname')] == us.four[1])[:, 0])
            vaindex = list(np.argwhere(tousx[:, 0, Columuser.index('rawname')] == us.four[2])[:, 0])
        elif userfortest == 'eleven':
            teindex = list(np.argwhere(tousx[:, 0, Columuser.index('rawname')] == us.eleven[1])[:, 0])
            vaindex = list(np.argwhere(tousx[:, 0, Columuser.index('rawname')] == us.eleven[2])[:, 0])
        elif userfortest == 'three':
            teindex = list(np.argwhere(tousx[:, 0, Columuser.index('rawname')] == us.three[1])[:, 0])
            vaindex = list(np.argwhere(tousx[:, 0, Columuser.index('rawname')] == us.three[2])[:, 0])
        elif userfortest == 'twelve':
            teindex = list(np.argwhere(tousx[:, 0, Columuser.index('rawname')] == us.twelve[1])[:, 0])
            vaindex = list(np.argwhere(tousx[:, 0, Columuser.index('rawname')] == us.twelve[2])[:, 0])
        elif userfortest == 'eight':
            teindex = list(np.argwhere(tousx[:, 0, Columuser.index('rawname')] == us.eight[1])[:, 0])
            vaindex = list(np.argwhere(tousx[:, 0, Columuser.index('rawname')] == us.eight[2])[:, 0])
        return teindex, vaindex

    def getusrtrain(self, tousx, tousy, userfortest, istimese):
        """Fetch data for training, testing, validation on multi-user multi-motion dataset.

            Args:
                tousx: Sensor readings, user information and motion information of
                multi-user multi-motion dataset
                tousy: Label (optical ground truth) of multi-user multi-motion dataset
                userfortest: User to be predicted
                istimese: is the current input time series data?

            Returns:
                usrtrainX: Sensor readings of training set. The training set only include
                single user and single displacement.
                usrtrainY: Label of training set. The training set only include
                single user and single displacement.

            Raises:
                None
            """
        Columuser = C.ColumMuMmx
        netrainX, netrainY = self.gentrainset(tousx, tousy, userfortest, Columuser)
        usrtrainX = np.array(S.totaldx(netrainX, 'MuMm', istimese))
        usrtrainY = np.array(S.totaldy(netrainY))
        return usrtrainX, usrtrainY

    def getusrteva(self, tousx, tousy, userfortest, istimese):
        """Fetch data for training, testing, validating on multi-user multi-motion dataset.

            Args:
                tousx: Sensor readings, user information and motion information of
                multi-user multi-motion dataset
                tousy: Label (optical ground truth) of multi-user multi-motion dataset
                userfortest: User to be predicted
                istimese: is the current input time series data?

            Returns:
                usrtestX: Sensor readings of testing set. The testing set only include
                single user and single displacement.
                usrtestY: Label of testing set. The testing set only include
                single user and single displacement.
                usrvaliX: Sensor readings of validation set. The validation set only include
                single user and single displacement.
                usrvaliY: Label of validation set. The validation set only include
                single user and single displacement.

            Raises:
                None
            """
        Columuser = C.ColumMuMmx
        usrtestX, usrtestY, usrvaliX, usrvaliY = self.gentevaset(tousx, tousy, userfortest, Columuser,
                                                                 istimese)
        return usrtestX, usrtestY, usrvaliX, usrvaliY

    def gentrainset(self, tousx, tousy, targetuser, Columuser):
        """Assign data for training and validation.

            Args:
                tousx: Sensor readings, user information and motion information of
                multi-user multi-motion dataset
                tousy: Label (optical ground truth) of multi-user multi-motion dataset
                targetuser: user name to be predicted
                Columuser: Column name of multi-user multi-motion dataset

            Returns:
                usrtrainX: sensor readings of training set of Multi-user Multi-motion
                dataset
                usrtrainY: labels of training set of Multi-user Multi-motion dataset
            Raises:
                None
            """
        # Assign data for training
        trindex, _ = self.filterusdTrain(targetuser, tousx, Columuser)
        # Assign data of different displacements for validation
        usrtrainX = tousx[trindex, :, :]
        usrtrainY = tousy[trindex, :]
        return usrtrainX, usrtrainY

    def gentevaset(self, tousx, tousy, targetuser, Columuser, istimese):
        """Assign data for training and validation.

            Args:
                tousx: Sensor readings, user information and motion information of
                multi-user multi-motion dataset
                tousy: Label (optical ground truth) of multi-user multi-motion dataset
                targetuser: user name to be predicted
                Columuser: Column name of multi-user multi-motion dataset
                istimese: is the current input time series data?

            Returns:
                usrteX: sensor readings of testing set of Multi-user Multi-motion
                dataset
                usrteY: labels of testing set of Multi-user Multi-motion dataset
                usrvaliX: sensor readings of validation set of Multi-user Multi-motion
                dataset
                usrvaliY: labels of validation set of Multi-user Multi-motion dataset

            Raises:
                None
            """
        # Assign data of different displacements for validation
        teindex, vaindex = self.filterusteva(targetuser, tousx, Columuser)
        usrteX = S.totaldx(tousx[teindex, :, :], 'MuMm', istimese)
        usrteY = S.totaldy(tousy[teindex, :])
        usrvaliX = S.totaldx(tousx[vaindex, :, :], 'MuMm', istimese)
        usrvaliY = S.totaldy(tousy[vaindex, :])
        return usrteX, usrteY, usrvaliX, usrvaliY


class SuMmloader:
    """
    Devide SuMm data for training, testing and validation.
    """

    def getmtteva(self, tomtx, tomty, Colummt, motionfortest, istimese):
        """Fetches single motion data of three displacements for testing and validation.

            Fetches data belong to single motion and assign test and validate datasets
             with data at three displacements.

            Args:
                tomtx: Sensor readings, motion information and user information of single-user multi-motion dataset
                tomty: Label (optical ground truth) of single-user multi-motion dataset
                Colummt: Column name of single-user multi-motion dataset
                motionfortest: Motion to be predicted
                istimese: is the current input time series data?

            Returns:
                mttestX: sensor readings of testing set
                mttestY: label of testing set
                mtvaliX: sensor readings of validation set
                mtvaliY: label of validation set


            Raises:
                None
            """
        if motionfortest == 'clap':
            vaindex = list(np.argwhere(tomtx[:, 0, Colummt.index('rawname')] == mo.clap[1])[:, 0])
            teindex = list(np.argwhere(tomtx[:, 0, Colummt.index('rawname')] == mo.clap[2])[:, 0])
        if motionfortest == 'walk':
            vaindex = list(np.argwhere(tomtx[:, 0, Colummt.index('rawname')] == mo.walk[1])[:, 0])
            teindex = list(np.argwhere(tomtx[:, 0, Colummt.index('rawname')] == mo.walk[2])[:, 0])
        if motionfortest == 'run':
            vaindex = list(np.argwhere(tomtx[:, 0, Colummt.index('rawname')] == mo.run[1])[:, 0])
            teindex = list(np.argwhere(tomtx[:, 0, Colummt.index('rawname')] == mo.run[2])[:, 0])
        if motionfortest == 'jump':
            vaindex = list(np.argwhere(tomtx[:, 0, Colummt.index('rawname')] == mo.jump[1])[:, 0])
            teindex = list(np.argwhere(tomtx[:, 0, Colummt.index('rawname')] == mo.jump[2])[:, 0])
        mttestX = S.totaldx(tomtx[teindex, :, :], 'SuMm', istimese)
        mtvaliX = S.totaldx(tomtx[vaindex, :, :], 'SuMm', istimese)
        mttestY = S.totaldy(tomty[teindex, :])
        mtvaliY = S.totaldy(tomty[vaindex, :])
        return mttestX, mttestY, mtvaliX, mtvaliY

    def getmttr(self, tomtx, tomty, Colummt, motionfortest, istimese):
        """Fetches single motion data of three displacements for training.

            Fetches data belong to single motion and assign training datasets
             with data at three displacements.

            Args:
                tomtx: Sensor readings, motion information and user information of single-user multi-motion dataset
                tomty: Label (optical ground truth) of single-user multi-motion dataset
                Colummt: Column name of single-user multi-motion dataset
                motionfortest: Motion to be predicted
                istimese: is the current input time series data?

            Returns:
                mttrainX: sensor readings of training set
                mttrainY: label of trainingset
                motiontrainname: displacement name of training set


            Raises:
                None
            """
        if motionfortest == 'clap':
            motiontrainname = mo.clap[0]
            trindex = list(np.argwhere(tomtx[:, 0, Colummt.index('rawname')] == motiontrainname)[:, 0])
        if motionfortest == 'walk':
            motiontrainname = mo.walk[0]
            trindex = list(np.argwhere(tomtx[:, 0, Colummt.index('rawname')] == motiontrainname)[:, 0])
        if motionfortest == 'run':
            motiontrainname = mo.run[0]
            trindex = list(np.argwhere(tomtx[:, 0, Colummt.index('rawname')] == motiontrainname)[:, 0])
        if motionfortest == 'jump':
            motiontrainname = mo.jump[0]
            trindex = list(np.argwhere(tomtx[:, 0, Colummt.index('rawname')] == motiontrainname)[:, 0])
        mttrainX = np.array(S.totaldx(tomtx[trindex, :, :], 'SuMm', istimese))
        mttrainY = np.array(S.totaldy(tomty[trindex, :]))

        return mttrainX, mttrainY, motiontrainname


class SuSmloader:
    """
    Load and preprocess SuSm data for training, testing and validation.
    """

    def loadSuSmTrain(self, changedim, istimese):
        """Assign data for training and validation.

            Args:
                changedim: whether to rank sensor readings according to the indicator.
                istimese: is the current input time series data?

            Returns:
                SuSmX_train: sensor readings of single-user single-motion training set
                SuSmY_train: labelss of single-user single-motion training set

            Raises:
                None
            """
        rankindicator = args.rankindicator
        type = 'SuSm'
        if changedim:
            if os.path.exists(C.file_dir + rankindicator + type + 'X_train30-t1.npy'):
                SuSmX_train = np.load(C.file_dir + rankindicator + type + 'X_train30-t1.npy')
                SuSmY_train = np.load(C.file_dir + rankindicator + type + 'Y_train30-t1.npy')
            else:
                SuSmX_train, SuSmY_train = R.predealtrain(changedim, istimese)
                SuSmX_test, SuSmY_test, SuSmX_validate, SuSmY_validate = R.predealteva(
                    changedim, istimese)
                np.save(C.file_dir + rankindicator + type + 'X_train30-t1.npy', SuSmX_train)
                np.save(C.file_dir + rankindicator + type + 'X_test30-t1.npy', SuSmX_test)
                np.save(C.file_dir + rankindicator + type + 'Y_train30-t1.npy', SuSmY_train)
                np.save(C.file_dir + rankindicator + type + 'Y_test30-t1.npy', SuSmY_test)
                np.save(C.file_dir + rankindicator + type + 'X_validate30-t1.npy', SuSmX_validate)
                np.save(C.file_dir + rankindicator + type + 'Y_validate30-t1.npy', SuSmY_validate)
        else:
            SuSmX_train, SuSmY_train = R.predealtrain(changedim, istimese)
        return SuSmX_train, SuSmY_train

    def loadSuSmTest(self, changedim, rankindicator, istimese):
        """Assign data for training and validation.

            Args:
                changedim: whether to rank sensor readings according to the indicator.
                rankindicator: which indicator is considered for ranking sensor readings
                (option: entropy, std, jitter)
                istimese: is the current input time series data?

            Returns:
                SuSmX_test: sensor readings of single-user single-motion testing set
                SuSmY_test: labelss of single-user single-motion testing set
                SuSmX_validate: sensor readings of single-user single-motion validation set
                SuSmY_validate: labelss of single-user single-motion validation set
                rawxtest: sensor readings, user information, motion information and
                 displacement information of training set of Multi-user Multi-motion dataset

            Raises:
                None
            """
        type = 'SuSm'
        if changedim:
            if os.path.exists(C.file_dir + rankindicator + type + 'X_train30-t1.npy'):
                SuSmX_test = np.load(C.file_dir + rankindicator + type + 'X_test30-t1.npy')
                SuSmY_test = np.load(C.file_dir + rankindicator + type + 'Y_test30-t1.npy')
                SuSmX_validate = np.load(C.file_dir + rankindicator + type + 'X_validate30-t1.npy')
                SuSmY_validate = np.load(C.file_dir + rankindicator + type + 'Y_validate30-t1.npy')
            else:
                SuSmX_test, SuSmY_test, SuSmX_validate, SuSmY_validate = R.predealteva(
                    changedim, istimese)
                np.save(C.file_dir + rankindicator + type + 'X_test30-t1.npy', SuSmX_test)
                np.save(C.file_dir + rankindicator + type + 'Y_test30-t1.npy', SuSmY_test)
                np.save(C.file_dir + rankindicator + type + 'X_validate30-t1.npy', SuSmX_validate)
                np.save(C.file_dir + rankindicator + type + 'Y_validate30-t1.npy', SuSmY_validate)
        else:
            SuSmX_test, SuSmY_test, SuSmX_validate, SuSmY_validate = R.predealteva(changedim,
                                                                                   istimese)
        rawxtest = 0
        return SuSmX_test, SuSmY_test, SuSmX_validate, SuSmY_validate, rawxtest
