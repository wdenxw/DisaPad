import utils as ul
import math
import torch
from lstmw import LstmwT
import numpy as np
import axfunction as S
import evaluate as eva
import torch.optim as optim
import torch.nn.functional as F
import dataloader as loader
from settings import settings
import constants as C
from Ranking import ranking
from dataloader import dataloader

loader = dataloader()
R = ranking()
st = settings()


class run:
    """
    Load data, train and evaluate model.
    """

    def __init__(self):
        self.ITERATION = 1000000000  # Number of times to enter the batch

    def loaddata(self, username, motionname):
        args, unknown = st.parser.parse_known_args()
        if motionname == None and username == None:
            Transfer, shuffle, Changedim, testTarget = S.initvar(True)
        else:
            Transfer, shuffle, Changedim, testTarget = S.initvar(False)

        X_train, Y_train, Y_tatrain, X_tatrain, targettrainname \
            = loader.loadtrain(username, motionname, Changedim, testTarget, shuffle,
                               True)
        space = int(X_train.shape[0] / args.batch_size)
        # Filter the data whose first or second entropy value is equal to the target domain training data for transferring.
        if motionname == None and username != None:
            dtype = 'MuMm'
            nppname = np.load(C.file_dir + dtype + 'totalname.npy')
            rankedindicator = np.load(C.file_dir + dtype + 'total' + args.rankindicator + '.npy')
        elif motionname != None and username == None:
            dtype = 'SuMm'
            nppname = np.load(C.file_dir + dtype + 'totalname.npy')
            rankedindicator = np.load(C.file_dir + dtype + 'total' + args.rankindicator + '.npy')
        elif motionname == None and username == None:
            return X_train, Y_train, space, Y_tatrain, X_tatrain, targettrainname, Transfer
        else:
            raise ValueError("The model cannot transfer to both SuMm and MuMm dataset together.")
        trainindall = np.load(C.file_dir + 'SuSmX_trainall.npy', allow_pickle=True)
        iindex = (list(nppname)).index(targettrainname)
        firstusrtrainentropy = rankedindicator[iindex][0]
        secondusrtrainentropy = rankedindicator[iindex][1]
        index1 = trainindall[firstusrtrainentropy - 1]
        index2 = trainindall[secondusrtrainentropy - 1]
        lastindex = list(set(list(index1) + list(index2)))
        X_train = X_train[lastindex, :, :]
        Y_train = Y_train[lastindex, :]
        space = int(len(lastindex) / args.batch_size)
        return X_train, Y_train, space, Y_tatrain, X_tatrain, targettrainname, Transfer

    def train(self):

        """train the model

        Args:
            None
        Returns:
            None

        Raises:
            ValueError: If the motion name and user name is not equal to None at the same time.
        """
        args, unknown = st.parser.parse_known_args()
        if args.motionname != None and args.username != None:
            raise ValueError("The model cannot transfer to both SuMm and MuMm dataset together.")
        X_train, Y_train, space, Y_target, X_target, targettrainname, Transfer = self.loaddata(args.username,
                                                                                               args.motionname)
        SuSmX_test, SuSmY_test, SuSmX_validate, SuSmY_validate, X_tatest, \
        Y_tatest, rawxtest, X_tavalidate, Y_tavalidate = self.evaloaddata(args.username, args.motionname)  #
        args, unknown = st.parser.parse_known_args()
        lrchange = args.lrchange
        lr = args.lr
        BATCHSIZE = args.batch_size
        lrdecay = args.lrdecay
        lambdParam = args.lambdParam
        MMDWEIGHT = args.MMDWEIGHT
        model = LstmwT(1, C.seq)
        model.cuda()
        print("model success cuda")
        valierror, epoch, targetlist, tarerrlist, model, valierror = S.initiali(
            model)
        optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)
        model.train()
        total_loss = 0
        clstotal = 0
        mmdtotal = 0
        mmdrr = 0
        train_losses = []
        loader_source = ul.transform_np(X_train, Y_train, BATCHSIZE)
        iter_source = iter(loader_source)
        if Transfer:
            loader_target = ul.transform_np(X_target, Y_target, BATCHSIZE)
            iter_target = iter(loader_target)
        for i in range(1, self.ITERATION + 1):
            try:
                Xbatch_source, Ybatch_source = next(iter_source)
            except Exception as _:
                iter_source = iter(loader_source)
                epoch += 1
                Xbatch_source, Ybatch_source = next(iter_source)
            if Transfer:
                try:
                    Xbatch_target, Ybatch_target = next(iter_target)
                except Exception as _:
                    iter_target = iter(loader_target)
                    Xbatch_target, Ybatch_target = next(iter_target)
            optimizer.zero_grad()
            output = model(Xbatch_source.cuda())
            cls_loss = F.mse_loss(output, Ybatch_source.cuda())
            loss = cls_loss
            if Transfer:
                outputtarget = model(Xbatch_target.cuda())
                mmdloss = ul.cal_mmd_loss(output, outputtarget)
                if i < 970:
                    lambd = (2 / (1 + math.exp(-10 * i / lambdParam)) - 1)
                if i >= 970:
                    lambd = (2 / (1 + math.exp(-10 * i / (lambdParam / 10))) - 1)
                lastmmdloss = lambd * MMDWEIGHT * mmdloss
                loss = cls_loss + lastmmdloss
            loss.backward()
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            epochLimit = args.epochs
            optimizer.step()
            clstotal += cls_loss.item()
            if Transfer:
                mmdrr += mmdloss.item()
                mmdtotal += lastmmdloss.item()
            total_loss += loss.item()
            train_losses.append(loss.item())
            if i % space == 0:
                cur_loss = total_loss / args.log_interval
                clslo = clstotal / args.log_interval
                if Transfer:
                    mmdraw = mmdrr / args.log_interval
                    print('Train Epoch: {:2d}\tLearning rate: {:.4f}\tLoss: {:.6f}\t cls loss:{:.6f}'
                          '\tMMD Loss: {:.6f}\tlambda: {:.6f}\tmmdraw loss: {:.6f}\titeration: {:.6f}'.format(
                        epoch, lr, cur_loss, clslo, lastmmdloss, lambd, mmdraw, i))

                else:
                    print('Train Epoch: {:2d} '
                          '\tLearning rate: {:.4f}\tLoss: {:.6f}\t cls loss:{:.6f}'.format(
                        epoch, lr, cur_loss, clslo))

                total_loss = 0
                clstotal = 0
                mmdtotal = 0
                mmdrr = 0
            if i % (space * lrchange) == 0:
                if lrdecay and lr > 0.0066:
                    lr = lr * 0.9
                    print("learning decay!!!!!!!!!")
                else:
                    lr = lr
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            if i % (space * 2) == 0:
                eva.evaluate(tarerrlist, valierror, model,
                             SuSmX_test, SuSmY_test, SuSmX_validate, SuSmY_validate, X_tatest, \
                             Y_tatest, X_tavalidate, Y_tavalidate, args.username, args.motionname)
                model.train()
            if epoch >= epochLimit + 1:
                print("End")
                break

    def evaloaddata(self, username, motionname):
        """Load data for evaluate
        """
        args, unknown = st.parser.parse_known_args()
        if username == None and motionname == None:
            Transfer, shuffle, Changedim, testTarget = S.initvar(True)
        else:
            Transfer, shuffle, Changedim, testTarget = S.initvar(False)
        SuSmX_test, SuSmY_test, SuSmX_validate, \
        SuSmY_validate, X_tatest, Y_tatest, rawxtest, X_tavalidate, Y_tavalidate \
            = loader.loadteva(username, motionname, Changedim, testTarget, args.rankindicator,
                              True)

        return SuSmX_test, SuSmY_test, SuSmX_validate, SuSmY_validate, X_tatest, Y_tatest, rawxtest, X_tavalidate, Y_tavalidate
