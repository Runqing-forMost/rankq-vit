import numpy as np
import torch.nn.functional as F
import torch
from copy import deepcopy

def label_smoothing(labels, num_classes=1000, epsilon=0.1):
    """
    Applies label smoothing to the given labels.

    Args:
        labels: numpy array of shape (batch_size, num_classes) containing one-hot labels
        epsilon: float, smoothing parameter (default: 0.1)

    Returns:
        smoothed_labels: numpy array of shape (batch_size, num_classes) containing smoothed labels
    """
    # num_classes = labels.shape[1]
    label_cp = deepcopy(labels)
    label_cp = F.one_hot(label_cp, num_classes)
    smoothed_labels = (1 - epsilon) * label_cp + epsilon / num_classes
    return smoothed_labels


def cross_entropy_loss(predictions, labels):
    """
    Computes the cross-entropy loss between the predictions and smoothed labels.

    Args:
        predictions: numpy array of shape (batch_size, num_classes) containing model predictions
        labels: numpy array of shape (batch_size, num_classes) containing one-hot labels
        epsilon: float, smoothing parameter (default: 0.1)

    Returns:
        loss: float, cross-entropy loss
    """
    # smoothed_labels = label_smoothing(labels, epsilon)
    loss = -torch.sum(labels * torch.log_softmax(predictions, dim=1), axis=1)
    return torch.mean(loss)

