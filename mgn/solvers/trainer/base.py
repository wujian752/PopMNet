import logging
import coloredlogs
import os

import math
import tensorboardX

from torch.optim import Adam, Adadelta, Adagrad, SGD, RMSprop
from utils.checkpoint import Checkpoint
from torchnet.meter import AverageValueMeter
from mgn.solvers.inferencer import get_inferencer
from mgn.module.encoder_decoder import encoder_decoder_factory
from mgn.model import PlanModel


class BaseTrainer(object):
    def __init__(self, model, criterion, args, inferencer=None):
        self.model = model
        self.criterion = criterion
        self.dataset = None

        self.args = args
        self.iteration = 0

        self.recorder = {}
        for recorder in self.define_record_infos():
            self.recorder[recorder] = AverageValueMeter()

        self.optimizer = self.get_optimizer(self.model.parameters(), self.args.optimizer)
        if inferencer is None:
            self.inferencer = get_inferencer(self.args.trainer, self.model, self.dataset, args)
        else:
            self.inferencer = inferencer

        self.train_iterations = 0

        self.logger = None
        self.logging_setting(name=os.path.join(args.save_dir, 'train.log'))
        self.train_writer = tensorboardX.SummaryWriter(log_dir=os.path.join(args.save_dir, 'train'))
        self.test_writer = tensorboardX.SummaryWriter(log_dir=os.path.join(args.save_dir, 'test'))

        self.logger.info(self.args)

    def define_record_infos(self):
        raise NotImplementedError

    def set_dataset(self, dataset):
        self.dataset = dataset
        if hasattr(self.inferencer, 'set_dataset'):
            self.inferencer.set_dataset(dataset)

    def get_optimizer(self, params, optimizer):
        if optimizer == 'Adam':
            return Adam(params, betas=(0.9, 0.999), lr=self.args.lr, weight_decay=self.args.weight_decay)
        elif optimizer == 'Adadelta':
            return Adadelta(params, lr=self.args.lr, rho=0.9, eps=1e-6, weight_decay=self.args.weight_decay)
        elif optimizer == 'Adagrad':
            return Adagrad(params, lr=self.args.lr, lr_decay=0, weight_decay=self.args.weight_decay)
        elif optimizer == 'SGD':
            return SGD(params, lr=self.args.lr, momentum=0, dampening=0, weight_decay=self.args.weight_decay,
                       nesterov=False)
        elif optimizer == 'RMSprop':
            return RMSprop(params, lr=self.args.lr, alpha=0.99, eps=1e-8, weight_decay=self.args.weight_decay,
                           momentum=0, centered=False)

    def cuda(self):
        self.model.cuda()
        self.criterion.cuda()
        self._cuda = True

    @classmethod
    def resume(cls, ckpt, criterion, parser):
        if os.path.isfile(ckpt):
            checkpoint = Checkpoint.load(path=ckpt)
        else:
            checkpoint = Checkpoint.load(Checkpoint.get_latest_checkpoint(ckpt))

        iteration = checkpoint.step
        args = checkpoint.args
        args.start_iteration = iteration + 1

        encoder_decoder = encoder_decoder_factory(arch=args.encoder_decoder, condition_on_chord=args.chords,
                                                  condition_on_plan=args.plan)

        model = PlanModel(input_size=encoder_decoder.input_size, encoder_decoder=encoder_decoder,
                          num_classes=encoder_decoder.num_classes, args=args)

        model.load_state_dict(checkpoint.model)

        args = parser.parse_args(namespace=args)
        trainer = cls(model, criterion=criterion, args=args)
        return trainer, args

    def logging_setting(self, name):
        # Define format
        fmt = '%(asctime)s, %(filename)s:%(lineno)d %(levelname)s %(message)s'
        # Add Notice Level
        logging.addLevelName(25, 'Notice')

        def notice(self, message, *args, **kwargs):
            if self.isEnabledFor(25):
                self._log(25, message, args, **kwargs)
        logging.Logger.notice = notice

        # Install Coloredlogs
        coloredlogs.install(level='INFO', fmt=fmt)

        # Get logger
        self.logger = logging.getLogger(__name__)
        # build and configure file handler
        file_handler = logging.FileHandler(name)
        file_handler.setLevel('INFO')
        file_handler.setFormatter(logging.Formatter(fmt))
        self.logger.addHandler(file_handler)

    def __str__(self):
        output = self.model.__str__()
        output += self.dataset.__str__()
        return output

    def train_steps(self):
        if self.dataset is None:
            raise ValueError('Please set dataset firstly.')

        self.model.train()

        best_loss = 10e10
        for epoch in range(self.args.start_epoch, self.args.max_epoch):
            self.dataset.train()
            self.model.train()

            self.train_single_epoch(epoch, loader=self.dataset.build_loader(batch_size=self.args.batch_size,
                                                                            num_workers=self.args.workers))
            self.train_iterations = epoch * math.ceil(len(self.dataset) / self.args.batch_size)

            self.dataset.eval()
            self.model.eval()
            loss = self.eval_single_epoch(epoch, loader=self.dataset.build_loader(batch_size=self.args.batch_size,
                                                                                  num_workers=self.args.workers))

            if epoch % self.args.save_interval == 0:
                checkpoint = Checkpoint(model=self.model, optimizer=self.optimizer, epoch=epoch, step=0,
                                        args=self.args)
                save_as_best = (epoch == 0 or loss < best_loss)

                if save_as_best:
                    self.logger.notice('Best loss: {:.3f} loss: {:.3f}'.format(best_loss, loss))
                    self.logger.notice('Save as best model: {}'.format(save_as_best))
                else:
                    self.logger.info('Best loss: {:.3f} loss: {:.3f}'.format(best_loss, loss))
                    self.logger.info('Save as best model: {}'.format(save_as_best))

                checkpoint.save(experiment_dir=self.args.save_dir, epoch=epoch,
                                save_as_best=save_as_best)
                best_loss = min(best_loss, loss)

                self.model.eval()
                self.inference(save_name='{}'.format(epoch))
                self.model.train()

    def record_infos(self, epoch, iteration, time, writer, phase='Train'):
        if phase == 'Train':
            log_msg = '{} [{epoch}][{iter}/{max_iter}] Time: {time:.4f}'.format(
                phase, epoch=epoch, iter=iteration, max_iter=math.ceil(len(self.dataset) / self.args.batch_size), time=time)
            global_step = epoch * math.ceil(len(self.dataset) / self.args.batch_size) + iteration
        elif phase == 'Test':
            log_msg = '{} [{epoch}]'.format(phase, epoch=epoch)
            global_step = self.train_iterations
        else:
            raise ValueError('Unknown phase.')

        for key in self.recorder:
            writer.add_scalar(tag=key, scalar_value=self.recorder[key].value()[0],
                              global_step=global_step)
            log_msg += ' {}: {:.3f}'.format(key, self.recorder[key].value()[0])
            self.recorder[key].reset()
        if phase == 'Train':
            self.logger.info(log_msg)
        else:
            self.logger.notice(log_msg)

    def train_single_epoch(self, epoch, loader):
        raise NotImplementedError

    def eval_single_epoch(self, epoch, loader, steps_of_epoch=0):
        """
        Evaluate the model on the test dataset.
        :param epoch:
        :param loader:
        :param steps_of_epoch
        :return:
            the loss on the test dataset.
        """
        raise NotImplementedError

    def inference(self, *args, **kwargs):
        self.inferencer.inference(*args, **kwargs)
