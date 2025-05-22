# _*_ coding:utf-8 _*_

import torch
from torch import nn
from base_model.mlp import MLP
from utils.data.scaler import StandardScaler, MeanScaler, LogScaler
from utils.Linear_Algebraic_Solution.LA_Solution import ridge_regression


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class hire_adjust(nn.Module):
    """
    Build the hierarchical structure, external trend guide module
    """
    def __init__(self,
                 args,
                 device,
                 dynamic_dim):
        super(hire_adjust, self).__init__()
        self.enc_in_be = args.enc_in_be
        self.enc_in = args.enc_in
        self.dynamic_dim = dynamic_dim
        self.node_number = args.node_number
        self.device = device
        self.query_projection = nn.Linear(self.dynamic_dim, self.dynamic_dim)
        self.key_projection = nn.Linear(self.dynamic_dim, self.dynamic_dim)
        self.value_projection = nn.Linear(self.dynamic_dim, self.dynamic_dim)
        nn.init.xavier_uniform_(self.query_projection.weight)
        nn.init.xavier_uniform_(self.key_projection.weight)
        nn.init.xavier_uniform_(self.value_projection.weight)

    def forward(self, y):
        B, N, D = y.shape
        B_group = (int)(B / self.node_number)

        # query,key shape [B, N, D], only label of parent node.
        y = y.reshape(B_group, self.node_number * N, D)
        query = self.query_projection(y)
        key = self.key_projection(y)
        Pa = torch.bmm(query, key.transpose(1, 2)) / (D ** 0.5)
        P = nn.Softmax(dim=-1)(Pa)
        value = self.value_projection(y)
        output_re = torch.bmm(P, value)
        y_pre = y + output_re
        y_pre = y_pre.reshape(B, N, D)

        return y_pre


class Model(nn.Module):
    def __init__(self,
                 args,
                 device,
                 dynamic_dim = 64,
                 kernel_size = 25):
        super(Model, self).__init__()
        self.device = device
        self.kernel_size = kernel_size
        self.predict_window = args.predict_window
        self.train_window = args.train_window
        self.enc_in_be = args.enc_in_be
        self.enc_in = args.enc_in
        self.dynamic_dim = dynamic_dim
        self.k_konwn = None
        self.k = None
        self.loss_ql = True if args.loss == 'ql' else False
        self.model = args.model

        # Decompsition Kernel Size
        self.decompsition = series_decomp(self.kernel_size)
        self.encoder_re = MLP(f_in=self.predict_window, f_out=self.dynamic_dim, hidden_dim=128)
        self.decoder_re = MLP(f_in=self.dynamic_dim, f_out=self.predict_window, hidden_dim=128)
        self.encoder = MLP(f_in=(self.train_window + self.predict_window), f_out=self.dynamic_dim, hidden_dim=128)
        self.decoder = MLP(f_in=self.dynamic_dim, f_out=(self.train_window + self.predict_window), hidden_dim=128)
        self.k_cnn = nn.Conv1d(in_channels=self.enc_in, out_channels=self.enc_in, kernel_size=3, padding=1, padding_mode='zeros')
        self.k_linear = nn.Linear(self.dynamic_dim, self.dynamic_dim)
        self.hire_adjust = hire_adjust(args, self.device, self.dynamic_dim)
        self.Linear_Seasonal = nn.Linear(self.train_window, self.predict_window)

        nn.init.xavier_uniform_(self.k_linear.weight)
        # nn.init.xavier_uniform_(self.k_linear_2.weight)
        nn.init.xavier_uniform_(self.Linear_Seasonal.weight)


    def forward(self, batch_x, behavior_x, behavior_y, behavior_matrix):
        # scaler normalization
        yscaler = MeanScaler()
        batch_x, x_mean = yscaler.fit_transform(batch_x)
        behavior_x, be_x_mean = yscaler.fit_transform(behavior_x)
        behavior_y = yscaler.transform(behavior_y, be_x_mean)
        behavior_y_known = behavior_y * behavior_matrix

        sea_init_x, trend_init_x = self.decompsition(batch_x)
        sea_init_be_x, trend_init_be_x = self.decompsition(behavior_x)

        # season part
        sea_init_x, sea_init_be_x = sea_init_x.transpose(1, 2), sea_init_be_x.transpose(1, 2)
        sea_output_x = self.Linear_Seasonal(sea_init_x)
        sea_output_be_x = self.Linear_Seasonal(sea_init_be_x)
        # print("sea_output_x.shape:{}, sea_output_be_x.shape:{}".format(sea_output_x.shape, sea_output_be_x.shape))

        # trend part
        # transpose
        trend_init_x, trend_init_be_x = trend_init_x.transpose(1, 2), trend_init_be_x.transpose(1, 2)
        behavior_y_known = behavior_y_known.transpose(1, 2)
        trend_output_x, trend_output_be_x, be_y_known = self.Linear_trend(trend_init_x, trend_init_be_x,
                                                                          behavior_y_known)
        # print("trend_output_x.shape:{}, trend_output_be_x.shape:{}, be_y_known.shape:{}".format(trend_output_x.shape, trend_output_be_x.shape, be_y_known.shape))

        # output change
        be_y = sea_output_be_x + trend_output_be_x
        y = sea_output_x + trend_output_x

        be_y_known = yscaler.inverse_transform(be_y_known, be_x_mean)
        be_y = yscaler.inverse_transform(be_y, be_x_mean)
        y = yscaler.inverse_transform(y, x_mean)

        be_y_known = torch.relu(be_y_known)
        be_y = torch.relu(be_y)
        y = torch.relu(y)

        be_y_known = be_y_known.transpose(1, 2)
        be_y = be_y.transpose(1, 2)
        y = y.transpose(1, 2)

        return y, be_y, be_y_known

    def Linear_trend(self, batch_x, behavior_x, behavior_y_known):
        # adjust series_len to fit predict_window, due to the koopman encoder part. #x [B, predict_window, N]
        B, N, _ = batch_x.shape
        _, M, _ = behavior_x.shape
        if (self.train_window < self.predict_window):
            pad = self.predict_window - self.train_window
            batch_x_re = torch.cat([torch.zeros(B, N, pad), batch_x], dim=2)
            behavior_x_re = torch.cat([torch.zeros(B, M, pad), behavior_x], dim=2)
        else:
            batch_x_re = batch_x[:, :, -self.predict_window:]
            behavior_x_re = behavior_x[:, :, -self.predict_window:]

        # koopman predictor module
        enc_be_x = self.encoder_re(behavior_x_re)
        enc_be_y_known = self.encoder_re(behavior_y_known)
        enc_x = self.encoder_re(batch_x_re)
        self.k_konwn = ridge_regression(enc_be_x, enc_be_y_known, alpha=0.1)
        enc_y = torch.bmm(enc_x, self.k_konwn)
        batch_y = self.decoder_re(enc_y)
        be_y_known = self.decoder_re(enc_be_y_known)
        batch_y = batch_y[:, :, -self.predict_window:]
        be_y_known = be_y_known[:, :, -self.predict_window:]

        # internal trend mining module
        # 数据concat
        input = torch.cat((batch_x, batch_y), dim=2)
        behavior_input = torch.cat((behavior_x, behavior_y_known), dim=2)

        # encode输入 [B, X, train_window + predict_window]
        input = self.encoder(input)
        behavior_input = self.encoder(behavior_input)

        # koopman adjust
        output = self.k_linear(input)
        behavior_output = self.k_linear(behavior_input)
        output = self.k_cnn(output)
        output = self.hire_adjust(output)

        behavior_output = self.decoder(behavior_output)
        output = self.decoder(output)

        behavior_output = behavior_output[:, :, -self.predict_window:]
        output = output[:, :, -self.predict_window:]

        return output, behavior_output, be_y_known
