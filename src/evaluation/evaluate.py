import torch
import torch.nn as nn
from src.data_processing.load_data import load_chameleon
from torch.optim import Adam
from src.models.deformable_gcn import DeformableGCN


def train(model, data, optimizer, loss_fn, alpha=0.01, beta=0.001):
    model.train()
    optimizer.zero_grad()

    outputs = model(data['x'], data['edge_index'], data['y'])
    raw_score = outputs[0]
    loss_sep = outputs[1]
    loss_focus = outputs[2]

    train_x = raw_score[data['train_mask']]
    train_y = data['y'][data['train_mask']]
    cross_entropy_loss = loss_fn(train_x, train_y)

    total_loss = cross_entropy_loss + (alpha * loss_sep) + (beta * loss_focus)

    total_loss.backward()
    optimizer.step()

    prediction = raw_score.argmax(dim=1)
    correct = (prediction[data['train_mask']] == data['y'][data['train_mask']]).sum().item()
    total = data['train_mask'].sum().item()
    accuracy = correct / total

    return total_loss.item(), accuracy


def evaluate(model, data, mask):
    model.eval()
    with torch.no_grad():
        output = model(data['x'], data['edge_index'], data['y'])
        raw_score = output[0]

        prediction = raw_score.argmax(dim=1)
        label = data['y']

        predict = prediction[mask]
        label_masked = label[mask]

        correct = (predict == label_masked).sum().item()
        total = mask.sum().item()
        accuracy = correct / total

        print(f"Correct: {correct}, Total: {total}")
        return accuracy
