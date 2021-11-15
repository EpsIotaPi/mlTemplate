import argparse



class HyperParameter():
    epochs = 100
    batch_size = 64
    lr = 1e5
    momentum = 0.5
    trainset_ratio = 0.8

    cuda_available = False
    seed = 1

    log_interval = 20
    log_dir = '../logger/'

    dataset_dir = './dataset/'

    save_model = True
    checkpoint_dir = './checkpoint/'

    train_name = "first_train"

    def __init__(self):
        parser = argparse.ArgumentParser(description='')
        parser.add_argument('--batch-size', help='input batch size for training (default: 64)',
                            type=int, default=64, metavar='N')

        args = parser.parse_args()

        self.batch_size = args.batch_size
        self.test_batch_size = args.test_batch_size
        self.epochs = args.epochs
        self.lr = args.lr
        self.momentum = args.momentum
        self.no_cuda = args.no_cuda
        self.seed = args.seed
        self.log_interval = args.log_interval
        self.save_model = args.save_model

    def parameter_log(self, logger):
        logger.info("EPOCHS=".format(self.epochs))


