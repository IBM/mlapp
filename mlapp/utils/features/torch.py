from torchtext.legacy.data import TabularDataset, BucketIterator, Iterator


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