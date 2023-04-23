"""
    Evaluate the tracking result.
"""
import torch
import numpy as np
from eskf import Reeskf


def evaluate(tarerrlist, valierrorlist, model, SuSmX_test, SuSmY_test,
             SuSmX_validate, SuSmY_validate, X_tatest, Y_tatest, X_tavalidate, Y_tavalidate, username, motionname):
    model.eval()
    with torch.no_grad():
        teerr, outputte, teerrorD = computeErr(SuSmX_test, SuSmY_test, 'test', model)
        vaerror, outputva, vaerrorD = computeErr(SuSmX_validate, SuSmY_validate, 'validate', model)
        if username == None and motionname == None:
            valierrorlist.append(vaerror)
            tarerrlist.append(teerr)
            # validate train result
        elif username != None or motionname != None:
            tavaer, outputto, _ = computeErr(X_tavalidate, Y_tavalidate, 'target validate', model)
            # validate target result
            targeter, outputta, tateerror = computeErr(X_tatest, Y_tatest, 'target test', model)
        else:
            raise ValueError("the motionname and username cannot exist at the same time")


def computeErr(Xdata, Ydata, name, model):
    global xtestraw, testTarget
    outputTorch = model(torch.from_numpy(Xdata).float().cuda())
    output = outputTorch.cpu().numpy()
    rawoutput = output
    if len(Ydata.shape) == 2:
        Ydata = Ydata[:, 0]
    if len(output.shape) == 2:
        output = output[:, 0]
    error = abs(output - Ydata)
    print("The average tracking error in " + name + "dataset is", np.mean(error))
    if name == 'target test' or name == 'target validate':
        # Kalman filter
        ee = Reeskf()
        outputkalman = ee.StartEskf(rawoutput, 9, 1.5)
        errorkalman = abs(outputkalman[:, 0] - Ydata)
        print("The average tracking error after ESKF in " + name + "dataset is", np.average(errorkalman))
        return np.mean(errorkalman), outputkalman, errorkalman
    else:
        return np.mean(error), output, error
