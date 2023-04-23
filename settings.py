import argparse


class settings:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='DisPad Project')
        self.parser.add_argument('--lr', type=float, default=0.01,
                                 help='initial learning rate (default: 1e-2)')
        self.parser.add_argument('--optim', type=str, default='Adam',
                                 help='optimizer to use (default: Adam)')
        self.parser.add_argument('--epochs', type=int, default=24,
                                 help='upper epoch limit (default: 25)')
        self.parser.add_argument('--epochlog_interval', type=int, default=1, metavar='N',
                                 help='lrchange interval (default: 1')
        self.parser.add_argument('--clip', type=float, default=1,
                                 help='gradient clip, -1 means no clip (default: 1)')
        self.parser.add_argument('--lrchange', type=int, default=2, metavar='N',
                                 help='lrchange interval (default: 2')
        self.parser.add_argument('--batch_size', type=int, default=512, metavar='N',
                                 help='batch size (default: 512)')
        self.parser.add_argument('--lambdParam', type=int, default=1010000000, metavar='N',
                                 help='parameter of controlling the mmd loss during training (default: 1010000000)')
        self.parser.add_argument('--MMDWEIGHT', type=int, default=5000000, metavar='N',
                                 help='weight of MMD loss (default: 5000000)')
        self.parser.add_argument('--lrdecay', type=bool, default=True, metavar='N',
                                 help='whether to decay learnning rate during training (default: True)')
        self.parser.add_argument('--rankindicator', type=str, default='entropy', metavar='N',
                                 help="which indicator to use (option: 'entropy', 'jitter', 'std')")
        self.parser.add_argument('--username', type=str, default=None, metavar='N',
                                 help="which user to transfer (option: ['two','three','four','five','six','seven','eight','night',"
                                      "'ten','eleven','twelve',None])")
        self.parser.add_argument('--motionname', type=str, default=None, metavar='N',
                                 help="which motion to transfer (option: ['run','clap','jump','walk',None])")
        self.parser.add_argument('--m', type=list, default=2, metavar='N',
                                 help="window size of fuzzy entropy function (default: 2)")
        self.parser.add_argument('--log-interval', type=int, default=194, metavar='N',
                                 help='report interval (default: 194')
        self.parser.add_argument('--r', type=list, default=0.25, metavar='N',
                                 help="Standard deviation of the original time series in fuzzy entropy function. The value range is 0.1-0.25. (default: 0.25)")
