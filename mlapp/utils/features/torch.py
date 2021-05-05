from mlapp.utils.general import calc_number_of_batches

import torch
import torch.nn.functional as F
import numpy as np
from torch import nn, optim
from torchtext.legacy.data import TabularDataset, BucketIterator, Iterator
from tqdm import tqdm
from sklearn.metrics import f1_score

# constants
TORCH = "torch"
MODEL = "model"
UNDERSCORE = "_"
CHECKPOINT = "checkpoint"
SHUFFLE = "shuffle"
SORT = "sort"


def convert_to_bucket_iterator(data: TabularDataset, batch_size: int, **kwargs):
    """
    This method convert TorchText dataset to bucket iterator.
    * kwargs:
    --> shuffle - bool shuffle the data in the iterator
    --> sort - sort data in the iterator
    :param data: Dataset
    :param batch_size: int
    :param kwargs: dict
    :return: BucketIterator
    """
    device = kwargs.get('device', "cpu")
    sort = kwargs.get('sort', False)
    sort_within_batch = kwargs.get('sort_within_batch', False)
    shuffle = kwargs.get('shuffle', False)
    return BucketIterator.splits(
        (data,)
        , batch_size=batch_size, device=device
        , shuffle=shuffle, sort_within_batch=sort_within_batch
        , sort=sort, sort_key=lambda x: x.order)[0]


def order_minus_one(order_list):
    return list(map(lambda x: x - 1, order_list))


def _batch_data(X_text, y=None, **kwargs):
    """
    This method is responsible for batching the data
    :param X_text:
    :param y:
    :param kwargs:
    :return:
    """
    y_batch = None
    X_text_batch = next(X_text)
    if y is not None:
        y_batch = y[X_text_batch.order.cpu().numpy()]

    return X_text_batch, y_batch


def train(model: nn.Module, data, **kwargs):
    """
    Train - this method will be used to train neural networks on a set of data.
    :param model: torch nn model object
    :param data: text data iterator
    :param kwargs:
        @option epochs: (int) number of epochs, default: 1.
        @option optimizer: (torch optim object) optimzer to use, default: Adam.
        @option loss: (torch nn.functional object) loss function to use, default: cross_entropy.
        @option learning_rate: (float) learning rate to use, default: 1e-3.
        @option batch_size: (int) batch size, default: 1.
        @option to_shuffle: (boolean) shuffle the data each epoch, default: False.
        @option verbose_epoch: (boolean) print any information after every epoch, default: True.
        @option verbose_batch: (boolean) print any information after every batch, default: True.
    :return:
    """
    # set parameters
    epochs = kwargs.get('epochs', 1)
    optimizer_func = kwargs.get('optimizer', optim.Adam)
    loss_func = kwargs.get('loss', F.cross_entropy)
    lr = kwargs.get('learning_rate', 1e-3)
    run_on_cpu = kwargs.get('run_on_cpu', False)
    batch_size = kwargs.get('batch_size', 1)
    to_shuffle = kwargs.get('to_shuffle', False)
    to_sort = kwargs.get('to_shuffle', False)
    verbose_epoch = kwargs.get('verbose_epoch', True)
    verbose_batch = kwargs.get('verbose_batch', True)
    seed = kwargs.get('seed', 0)
    on_cuda = False

    # set accuracies dict
    accuracies = {}
    accuracies['epochs'] = {}

    if seed > 0:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # initial optimizer
    optimizer = optimizer_func(model.parameters(), lr=lr)

    # check for running on 'cuda' or 'cpu'
    device = "cuda" if torch.cuda.is_available() and not run_on_cpu else "cpu"
    if device == 'cuda':
        print("> Running on CUDA")
        model.cuda()
        on_cuda = True
    else:
        print("> Running on CPU")

    # get X, y of train data, X -> tuple of text data
    X_train_text, y_train = data['train']

    # calc number of batches
    train_data_len = y_train.shape[0]
    n_batches = calc_number_of_batches(train_data_len, batch_size)
    print("> Total of %s batches" % str(n_batches))

    for epoch in tqdm(range(epochs), desc="Running Epochs", total=epochs):
        # init loss an acc counters
        total_epoch_loss = 0
        correct = 0
        acc_denominator = 0
        total_epoch_acc = 0
        y_pred = []

        # set model mode to 'train'
        model.train()

        # initialize bucket_iterator on text each epoch and check for shuffling and sorting
        tabular_dataset_to_bucket_iterator_kwargs = {}
        tabular_dataset_to_bucket_iterator_kwargs[SHUFFLE] = to_shuffle
        tabular_dataset_to_bucket_iterator_kwargs[SORT] = to_sort
        X_train_text_bucket_iterator = convert_to_bucket_iterator(X_train_text, batch_size,
                                                                  **tabular_dataset_to_bucket_iterator_kwargs)

        # create the iter object on train text data
        X_train_text_iter = iter(X_train_text_bucket_iterator)

        for batch in range(n_batches):
            # creates batch for training
            batch_data_kwargs = {}
            batch_data_kwargs['batch'] = batch
            batch_data_kwargs['batch_size'] = batch_size
            X_text_batch, y_batch = _batch_data(X_train_text_iter, y_train, **batch_data_kwargs)

            # check if cuda mode is on
            X_text_batch.text = X_text_batch.text.cuda() if on_cuda else X_text_batch.text
            y_batch = y_batch.cuda() if on_cuda else y_batch

            # perform forward propagation
            optimizer.zero_grad()
            output = model(X_text_batch, **kwargs)

            # calculates loss function and perform back propagation
            loss = loss_func(output, y_batch)
            loss.backward()
            optimizer.step()

            # calc train accuracy
            pred = output.data.max(1, keepdim=True)[1]
            y_pred.extend(list(pred.cpu().numpy().reshape(1, -1)[0]))
            acc_denominator += y_batch.shape[0]
            correct += pred.eq(y_batch.data.view_as(pred)).cpu().sum()
            total_epoch_acc = np.true_divide(100.0 * correct, acc_denominator)
            total_epoch_f1_score = f1_score(y_true=list(y_train.cpu().numpy()[:len(y_pred)]), y_pred=y_pred, average='weighted')
            total_epoch_loss += loss.item()
            if verbose_batch and (batch % 1000 == 0) and batch > 0:
                print("Epoch {}, batch {}, train accuracy {:.2f}, train f1-score {:.2f}, {}".format(epoch + 1, batch + 1, total_epoch_acc.item(), total_epoch_f1_score))

        total_epoch_loss = np.true_divide(total_epoch_loss, n_batches).item()
        train_f1_score = f1_score(y_true=list(y_train.cpu().numpy()), y_pred=y_pred, average='weighted')
        dev_result = evaluate(model, data['test'], **kwargs)
        dev_accuracy = dev_result.get('accuracy', -1)
        dev_f1_score = dev_result.get('f1_score', -1)

        # set epoch accuracies
        accuracies['epochs'][epoch + 1] = {
            'train_accuracy': total_epoch_acc.item(),
            'dev_accuracy': dev_accuracy,
            'loss': total_epoch_loss,
            'train_f1_score_weighted': train_f1_score
        }

        if verbose_epoch:
            print(
                "Epoch {}: loss {:.2f}, train accuracy {:.2f}, train f1 score {:.2f}, dev accuracy {:.2f}, dev f1 score {:.2f}".format(
                    epoch + 1, total_epoch_loss, total_epoch_acc.item(), train_f1_score, dev_accuracy, dev_f1_score))

    # set final accuracies
    accuracies['final_train_accuracy'] = total_epoch_acc.item()
    accuracies['final_dev_accuracy'] = dev_accuracy
    accuracies['final_loss'] = total_epoch_loss
    accuracies['f1_score_weighted'] = f1_score(y_true=list(y_train.cpu().numpy()), y_pred=y_pred, average='weighted')

    # set result dict
    result = {}
    result["model"] = model.state_dict()
    result["model_params"] = model.get_params()
    result["optimizer"] = optimizer.state_dict()
    result["loss"] = loss
    result["epoch"] = epoch + 1
    result["metrics"] = accuracies
    result["batch_size"] = batch_size

    return result


def evaluate(model: nn.Module, data, **kwargs):
    # set parameters
    run_on_cpu = kwargs.get('run_on_cpu', False)
    batch_size = kwargs.get('batch_size', 1)
    verbose = kwargs.get('verbose', True)
    on_cuda = False

    # set model mode to 'eval'
    model.eval()

    # check for running on 'cuda' or 'cpu'
    device = "cuda" if torch.cuda.is_available() and not run_on_cpu else "cpu"
    if device == 'cuda':
        model.cuda()
        on_cuda = True

    # get X, y of eval data, X -> tuple of text data
    X_text, y = data

    # calc number of batches
    data_len = y.shape[0]
    n_batches = calc_number_of_batches(data_len, batch_size)
    correct = 0
    y_pred = []

    # initialize iterator on text each epoch
    X_text_bucket_iterator = convert_to_bucket_iterator(X_text, batch_size)
    X_text_iter = iter(X_text_bucket_iterator)

    for batch in range(n_batches):
        # creates batch for training
        batch_data_kwargs = {}
        batch_data_kwargs['batch'] = batch
        batch_data_kwargs['batch_size'] = batch_size
        X_text_batch, y_batch = _batch_data(X_text_iter, y, **batch_data_kwargs)

        # check if cuda mode is on
        X_text_batch.text = X_text_batch.text.cuda() if on_cuda else X_text_batch.text
        y_batch = y_batch.cuda() if on_cuda else y_batch

        # perform forward propagation
        output = model(X_text_batch, **kwargs)

        # calc evaluation accuracy
        pred = output.data.max(1, keepdim=True)[1]
        y_pred.extend(list(pred.cpu().numpy().reshape(1, -1)[0]))
        correct += pred.eq(y_batch.data.view_as(pred)).cpu().sum()

    result = {}
    result['predictions '] = y_pred
    result['accuracy'] = np.true_divide(correct * 100, data_len).item()
    result['f1_score'] = f1_score(y_true=list(y.cpu().numpy()), y_pred=y_pred, average='weighted')

    if verbose:
        print(
            "Total evaluation accuracy: {:.2f}, Total f1 score: {:.2f}".format(result['accuracy'], result['f1_score']))
    return result


def predict(model: nn.Module, data, **kwargs):
    # set parameters
    run_on_cpu = kwargs.get('run_on_cpu', False)
    batch_size = kwargs.get('batch_size', 1)
    data_len = kwargs.get('data_len')
    on_cuda = False

    # set model mode to 'eval'
    model.eval()

    # check for running on 'cuda' or 'cpu'
    device = "cuda" if torch.cuda.is_available() and not run_on_cpu else "cpu"
    if device == 'cuda':
        model.cuda()
        on_cuda = True

    # get X of eval data, X -> tuple of text data
    X_text = data

    # calc number of batches
    n_batches = calc_number_of_batches(data_len, batch_size)

    y_pred = []
    X_text_iter = iter(X_text)

    for batch in tqdm(range(n_batches), total=n_batches, desc="Predicting"):
        # creates batch for training
        batch_data_kwargs = {}
        batch_data_kwargs['batch'] = batch
        batch_data_kwargs['batch_size'] = batch_size
        X_text_batch, y_batch = _batch_data(X_text_iter, **batch_data_kwargs)

        # check if cuda mode is on
        X_text_batch.text = X_text_batch.text.cuda() if on_cuda else X_text_batch.text

        # perform forward propagation
        output = model(X_text_batch, **kwargs)

        # calc evaluation accuracy
        pred = output.data.max(1, keepdim=True)[1]
        y_pred.extend(list(pred.cpu().numpy().reshape(1, -1)[0]))

    result = {}
    result['y_pred'] = y_pred

    return result
