import torch.nn as nn
import torch.nn.functional as F
from layers import snowball_layer, GraphConvolution, GraphAttentionLayer, SpGraphAttentionLayer
import torch
import numpy as np

class GVCLN(nn.Module):
    def __init__(self, nfeat, nclass, nhid_1, dropout_1, nhid_2, dropout_2, alpha_2, nheads_2):
        super(GVCLN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid_1)
        self.gc2 = GraphConvolution(nhid_1*3, nclass)
        self.dropout_1 = dropout_1
        self.dropout_2 = dropout_2
        self.attentions = [SpGraphAttentionLayer(nfeat, nhid_2, dropout=dropout_2, alpha=alpha_2, concat=True) for _ in range(nheads_2)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = SpGraphAttentionLayer(nhid_2*nheads_2, nclass, dropout=dropout_2, alpha=alpha_2, concat=False)
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda()
        self.b1 = nn.Parameter(torch.FloatTensor(1), requires_grad=True).data.fill_(0.05).to(torch.device("cuda"))
        self.b2 = nn.Parameter(torch.FloatTensor(1), requires_grad=True).data.fill_(0.05).to(torch.device("cuda"))
        
    def forward(self, x, adj, idx_train, labels):
        y = F.relu(self.gc1(x, adj))
        y = torch.cat([y, y, y], dim=1)
        y = F.dropout(y, self.dropout_1, training=self.training)
        y = self.gc2(y, adj)
        semi_loss_1 = torch.nn.CrossEntropyLoss()(y[idx_train], labels[idx_train])

        z = F.dropout(x, self.dropout_2, training=self.training)
        z = torch.cat([att(z, adj) for att in self.attentions], dim=1)
        z = F.dropout(z, self.dropout_2, training=self.training)
        z = F.elu(self.out_att(z, adj))
        semi_loss_2 = torch.nn.CrossEntropyLoss()(z[idx_train], labels[idx_train])

        log_probs = self.logsoftmax(y)
        CL_loss_12 = (- F.softmax(z, dim=1).detach() * log_probs).mean(0).sum()

        loss_11 = semi_loss_1
        loss_21 = semi_loss_2

        loss_12 = semi_loss_1 + self.b1*CL_loss_12
        loss_22 = semi_loss_2 + self.b2*CL_loss_12
        return y, z, loss_11, loss_21, loss_12, loss_22

class graph_convolutional_network(nn.Module):
    def __init__(self, nfeat, nlayers, nhid, nclass, dropout):
        super(graph_convolutional_network, self).__init__()
        self.nfeat, self.nlayers, self.nhid, self.nclass = nfeat, nlayers, nhid, nclass
        self.dropout = dropout
        self.hidden = nn.ModuleList()

    def reset_parameters(self):
        for layer in self.hidden:
            layer.reset_parameters()
        self.out.reset_parameters()

class modified_GVCLN_snowball(graph_convolutional_network):
    def __init__(self, nfeat, nlayers_1, nhid_1, nclass_1, activation_1, dropout_1, nhid_2, dropout_2, alpha_2, nheads_2):
        super(modified_GVCLN_snowball, self).__init__(nfeat, nlayers_1, nhid_1, nclass_1, dropout_1)

        self.activation = activation_1
        for k in range(nlayers_1):
            self.hidden.append(snowball_layer(k * nhid_1 + nfeat, nhid_1))
        self.out = snowball_layer(nlayers_1 * nhid_1 + nfeat, nclass_1)


        self.dropout_1 = dropout_1
        self.dropout_2 = dropout_2
        self.attentions = [SpGraphAttentionLayer(nfeat, nhid_2, dropout=dropout_2, alpha=alpha_2, concat=True) for _ in range(nheads_2)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = SpGraphAttentionLayer(nhid_2*nheads_2, nclass_1, dropout=dropout_2, alpha=alpha_2, concat=False)
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda()
        self.b1 = nn.Parameter(torch.FloatTensor(1), requires_grad=True).data.fill_(0.05).to(torch.device("cuda"))
        self.b2 = nn.Parameter(torch.FloatTensor(1), requires_grad=True).data.fill_(0.05).to(torch.device("cuda"))
        
    def forward(self, x, adj, idx_train, labels):
        list_output_blocks = []
        for layer, layer_num in zip(self.hidden, np.arange(self.nlayers)):
            if layer_num == 0:
                list_output_blocks.append(F.dropout(self.activation(layer(x, adj)), self.dropout, training=self.training))
            else:
                list_output_blocks.append(F.dropout(self.activation(layer(torch.cat([x] + list_output_blocks[0: layer_num], 1), adj)), self.dropout, training=self.training))
        output = self.out(torch.cat([x] + list_output_blocks, 1), adj, eye=False)
        # y = F.log_softmax(output,dim=1)
        y = output
        semi_loss_1 = torch.nn.CrossEntropyLoss()(y[idx_train], labels[idx_train])

        z = F.dropout(x, self.dropout_2, training=self.training)
        z = torch.cat([att(z, adj) for att in self.attentions], dim=1)
        z = F.dropout(z, self.dropout_2, training=self.training)
        z = F.elu(self.out_att(z, adj))
        semi_loss_2 = torch.nn.CrossEntropyLoss()(z[idx_train], labels[idx_train])

        log_probs = self.logsoftmax(y)
        CL_loss_12 = (- F.softmax(z, dim=1).detach() * log_probs).mean(0).sum()

        loss_11 = semi_loss_1
        loss_21 = semi_loss_2

        loss_12 = semi_loss_1 + self.b1*CL_loss_12
        loss_22 = semi_loss_2 + self.b2*CL_loss_12
        return y, z, loss_11, loss_21, loss_12, loss_22