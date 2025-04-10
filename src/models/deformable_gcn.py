import torch
import torch.nn.functional as F
import torch.nn as nn


class graphSmoothing(nn.Module):
    def __init__(self, smoothing_steps=3):
        super().__init__()
        self.steps = smoothing_steps

    def forward(self, x, edge_index):
        h = x
        x_list = [h]
        for _ in range(self.steps):
            h = self.aggregate_mean(h, edge_index)
            x_list.append(h)
        return torch.stack(x_list, dim=1)

    def aggregate_mean(self, x, edge_index):
        nodes = x.size(0)
        feature_sum = torch.zeros_like(x)
        neighbor = torch.zeros(nodes, device=x.device)

        for i in range(edge_index.size(1)):
            src = edge_index[0, i]
            dest = edge_index[1, i]

            feature_sum[dest] += x[src]
            neighbor[dest] += 1

        neighbor = neighbor.clamp(min=1).unsqueeze(-1)
        return feature_sum / neighbor


class GCNConvolution(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.att = nn.Sequential(
            nn.Linear(in_dim * 2, 1),
            nn.LeakyReLU(),
        )

    def forward(self, x, edge_index):
        src, dest = edge_index

        h_src = x[src]
        h_dest = x[dest]

        score = self.att(torch.cat([h_src, h_dest], dim=-1))
        weighted = score * self.lin(h_src)

        nodes = x.size(0)
        features = torch.zeros(nodes, self.lin.out_features, device=x.device)
        attention_score = torch.zeros_like(features)

        features = features.index_add(0, dest, weighted)
        att_score = attention_score.index_add(0, dest, torch.abs(weighted))

        return features, att_score


class DeformableGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, smoothing_steps=3):
        super().__init__()
        self.pe = graphSmoothing(smoothing_steps=smoothing_steps)
        self.conv1 = GCNConvolution(input_dim, hidden_dim)
        self.conv2 = GCNConvolution(hidden_dim, out_dim)

    def forward(self, x, edge_index, y=None):
        x = self.pe(x, edge_index)
        x = x.mean(dim=1)

        h1, attention1 = self.conv1(x, edge_index)
        h1 = F.relu(h1)
        h1 = F.dropout(h1, p=0.5, training=self.training)

        out, attention2 = self.conv2(h1, edge_index)
        # print("Attention Shape: ", attention2.shape)  # [4177, 3]

        if y is not None:
            loss_separation = self.l_separation(out, y)
        else:
            loss_separation = 0

        if y is not None:
            loss_focus = self.l_focus(attention2, y)
        else:
            loss_focus = 0

        return out, loss_separation, loss_focus

    def l_separation(self, feature, label):
        if label is None:
            return 0
        loss = 0.0
        num_classes = label.max().item() + 1
        class_means = []

        for c in range(num_classes):
            mask = (label == c)
            if mask.sum() == 0:
                continue
            class_nodes = feature[mask]
            mean_vector = class_nodes.mean(dim=0).detach()
            class_means.append(mean_vector)

        for i in range(len(class_means)):
            for j in range(i + 1, len(class_means)):
                sim = F.cosine_similarity(class_means[i], class_means[j], dim=0)
                loss += sim
        return -loss

    def l_focus(self, attention_feats, labels):
        if labels is None:
            return 0
        loss = 0.0
        num_classes = labels.max().item() + 1

        for i in range(num_classes):
            mask = (labels == i)
            if mask.sum() == 0:
                continue
            group_feature = attention_feats[mask].detach()
            mean = group_feature.mean(dim=0)
            loss += ((group_feature - mean) ** 2).sum()
        return loss
