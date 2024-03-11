import numpy as np
from torch import Tensor


def dirichlet_unbalance_split(num_clients, num_samples, alpha):
    """Assign different sample number for each client using Dirichlet distribution.

    Sample numbers for clients are drawn from Dirichlet distribution.

    Args:
        num_clients (int): Number of clients for partition.
        num_samples (int): Total number of samples.
        alpha (float): Dirichlet concentration parameter

    Returns:
        numpy.ndarray: A numpy array consisting ``num_clients`` integer elements, each represents sample number of corresponding clients.

    """
    min_size = 0
    while min_size < 10:
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = proportions / proportions.sum()
        min_size = np.min(proportions * num_samples)

    client_sample_nums = (proportions * num_samples).astype(int)
    return client_sample_nums


def lognormal_unbalance_split(num_clients, num_samples, unbalance_sgm):
    """Assign different sample number for each client using Log-Normal distribution.

    Sample numbers for clients are drawn from Log-Normal distribution.

    Args:
        num_clients (int): Number of clients for partition.
        num_samples (int): Total number of samples.
        unbalance_sgm (float): Log-normal variance. When equals to ``0``, the partition is equal to :func:`balance_partition`.

    Returns:
        numpy.ndarray: A numpy array consisting ``num_clients`` integer elements, each represents sample number of corresponding clients.

    """
    num_samples_per_client = int(num_samples / num_clients)
    if unbalance_sgm != 0:
        client_sample_nums = np.random.lognormal(mean=np.log(num_samples_per_client),
                                                 sigma=unbalance_sgm,
                                                 size=num_clients)
        client_sample_nums = (
                client_sample_nums / np.sum(client_sample_nums) * num_samples).astype(int)
        diff = np.sum(client_sample_nums) - num_samples  # diff <= 0

        # Add/Subtract the excess number starting from first client
        if diff != 0:
            for cid in range(num_clients):
                if client_sample_nums[cid] > diff:
                    client_sample_nums[cid] -= diff
                    break
    else:
        client_sample_nums = (np.ones(num_clients) * num_samples_per_client).astype(int)

    return client_sample_nums


def random_slicing(dataset, num_clients, sgm=0):
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    if num_clients > 0:
        user_samples = lognormal_unbalance_split(num_clients, len(dataset), sgm)
    for i in range(num_clients):
        dict_users[i] = list(
            np.random.choice(all_idxs, user_samples[i], replace=False))
        all_idxs = list(set(all_idxs) - set(dict_users[i]))
    return dict_users


def tensor_bytes(tensor: Tensor):
    if tensor is None:
        return 0
    """Return the number of bytes of a tensor."""
    return tensor.numel() * tensor.element_size()


def size_str(k):
    units = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB']
    size = k
    unit_index = 0
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024.0
        unit_index += 1
    return "{:.2f} {}".format(size, units[unit_index])
