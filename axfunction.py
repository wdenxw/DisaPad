"""
Some auxiliary functions
"""
import constants as C
import numpy as np
import torch.nn as nn
import torch.nn.init as init


def initvar(shouldtest):
    """Extract row index of the SuSm dataset with the lowest indicator (entropy, jitter, std)
     value of each sensor

        Args:
            shouldtest: whether to train and test on SuSm dataset

        Returns:
            Transfer: whether to transfer to another application scenario: e.g. MuMm or SuMm
            shuffle: whether to shuffle the training dataset
            Changedim: whether to rank sensor readings
            testTarget: whether to test current model on MuMm or SuMm

        Raises:
            None
        """
    if shouldtest:
        Transfer = False
        shuffle = True
        Changedim = False
        testTarget = False
    else:
        Transfer = True
        shuffle = False
        Changedim = True
        testTarget = True
    return Transfer, shuffle, Changedim, testTarget


def initiali(model):
    targetlist = []
    valierror = []
    tarerrlist = []
    model.apply(weight_init)
    epoch = 1
    return valierror, epoch, targetlist, tarerrlist, model, valierror


def getindex(dtype, xdata, totalindicator, totalname):
    """Extract row index of the SuSm dataset with the lowest indicator (entropy, jitter, std)
     value of each sensor

        Args:
            dtype: dataset name
            xdata: sensor readings, motion information and displacement information of
            the current dataset
            totalindicator: the indicator value (entropy, jitter, std) of all displacements of
            the current dataset
            totalname: the displacement name of all displacements of the current dataset
        Returns:
            None

        Raises:
            None
        """
    all = []
    for j in range(1, 7, 1):
        relatedind = []
        for i in range(len(totalname)):
            currentindex = np.argwhere(xdata[:, 0, C.ColumSuSmx.index('rawname')] == totalname[i])[:, 0]
            if (totalindicator[i][0] == j) or (totalindicator[i][1] == j):
                relatedind += list(currentindex)
        all.append(relatedind)
    all = np.array(all)
    np.save(C.file_dir + dtype + 'all.npy', all)


def reshappp(input, istimese):
    """Change the 3d sensor readings to 2d.

    Change the 3d sensor readings to 2d when the adopted model is fully connected neural work.

        Args:
            input: sensor readings of the current dataset
            istimese: is the current input time series data?
        Returns:
            output: 2d sensor readings when the input is not time series data
            or 3d sensor readings when the input should be time series data

        Raises:
            None
        """
    if istimese:
        output = input

    else:
        output = input[:, -1, :]
    return output


def totaldx(input, dtype, istimese):
    """Only return needed sensor readings.

        Args:
            input: input sensor readings, motion information and displacement information of
            the current dataset
            dtype: dataset name
            istimese: is the current input time series data?
        Returns:
            output: needed sensor readings

        Raises:
            ValueError: If the input does not match any dataset.
        """
    if dtype == 'MuMm':
        Columx = C.ColumMuMmx
        inputcolum = [Columx.index('r1'), Columx.index('r2'), Columx.index('r3'),
                      Columx.index('r4'), Columx.index('r5'), Columx.index('r6')]
        input = input[:, :, inputcolum]
    elif 'SuMm' in dtype:
        Columx = C.ColumSuMmx
        input = input[:, :, Columx.index('r1'):Columx.index('circular')]
        input = np.array(input)
    elif 'SuSm' in dtype:
        Columx = C.ColumSuSmx
        input = input[:, :, Columx.index('r1'):Columx.index('circular')]
        input = np.array(input)
    else:
        raise ValueError("Type error. The input does not match any dataset. Current input:", dtype)
    input = input.astype(float)
    output = reshappp(input, istimese)
    return output


def totaldy(input):
    """Change 3d label data to 2d.

        Args:
            input: labels of the current dataset.
            endpose:

        Returns:
            output: label of current dataset (after changing the sequence length)

        Raises:
            None
        """
    # output=input
    output = input[:, -1]
    return output


def ToNormalize(inputArr):
    """Normalize 1D sensor readings.

        Args:
            inputArr: Sensor readings to be normalized

        Returns:
            outputnpy: Normalized sensor readings

        Raises:
            None
        """
    inputArr = inputArr.astype(float)
    min = np.min(inputArr)
    max = np.max(inputArr)
    yy = max - min
    lastlist = []
    for i in range(len(inputArr)):
        inputArr[i] = (inputArr[i] - min)
        inputArr[i] = inputArr[i]
        lastlist.append(inputArr[i] / yy)
    outputnpy = np.array(lastlist)
    return outputnpy


def weight_init(m):
    """Init weights of neural network.

        Args:
            m: input model

        Returns:
            None

        Raises:
            None
        """
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
