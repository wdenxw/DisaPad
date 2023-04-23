"""
Use the error state kalman filter to filter noises
"""
import numpy as np


class Reeskf(object):
    """use the error state kalman filter to filter noises
    use the error state kalman filter to filter noises

    Args:
        adjust P in maineskf function. That's the predicted error
        adjust self.V in maineskf function. That's the mesuring error
    Returns:
        None

    Raises:
        None
    """

    def __init__(self):
        self.bvhn = None
        self.deltax = np.zeros((2, 2))
        self.yhat = None  # output
        self.DeltaT = 0.05
        self.x = np.array([[0, 0], [0, 0]], dtype=float)

    def StartEskf(self, RawArray, Pparam, Vparam):
        """use the error state kalman filter to filter noises
        do the manipulation to the whole 3-demension array

        Args:
            nparray: the 3-demension input array
        Returns:
            the filtered 3-demension array

        Raises:
            None
        """
        RawArray = RawArray.transpose()
        NewArray = np.ones((RawArray.shape[0], RawArray.shape[1]))
        for i in range(RawArray.shape[0]):
            ToInput = RawArray[i]
            NewSeq = self.Doeskf(ToInput, Pparam, Vparam)
            NewArray[i] = NewSeq
        NewArray = NewArray.transpose()
        return NewArray

    def Doeskf(self, nparray, Pparam, Vparam):
        """use the error state kalman filter to filter noises
        do the manipulation to the whole 1*c array (a line that belongs to the same sequence)

        Args:
            nparray: the input sequence
        Returns:
            the filtered sequence

        Raises:
            None
        """
        newnparray = np.ones(nparray.shape[0])

        for i in range(nparray.shape[0]):
            if i == 0:
                update, P = self.maineskf(nparray[0], nparray[0], Pparam, Vparam)
            else:
                update, P = self.maineskf(nparray[i], update, Pparam, Vparam)
            newnparray[i] = update
        return newnparray

    def maineskf(self, measure1, measure2, Pparam, Vparam):
        """use the error state kalman filter to filter noises
        the core algorithm of the error state kalman filter

        Args:
            adjust P in maineskf function. That's the predicted error
            adjust self.V in maineskf function. That's the mesuring error
            measure1 is the first measuring value
            measure2 is the second measuring value
        Returns:
            None

        Raises:
            None
        """
        P = np.array([[Pparam, 0], [0, Pparam]], dtype=float)
        self.V = Vparam
        self.x[0][0] = measure1
        self.x[1][0] = self.x[1][0]
        F = np.array([[1, self.DeltaT], [0, 1]])  # error state Jacobian matrix
        FP = F.dot(P)
        P = FP.dot(F.T)
        H = np.array([[1, 0]])
        PHt = P.dot(H.T)
        S = H.dot(PHt)
        K = P.dot(H.T) * (np.linalg.inv(S + self.V))
        y = measure2
        self.deltax = K * (y - (self.x[0][0] + self.deltax[0][0]))  # Correct the error state.
        P = (np.eye(2) - K * H).dot(P)  # Update estimation error.
        self.x = self.x + self.deltax
        self.deltax = np.zeros((2, 2))  # reset the error state
        return self.x[0][0], P
