"""
InceptionTime model for comparison against real data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as functional


class Flatten(nn.Module):
    def __init__(self, out_features):
        super(Flatten, self).__init__()
        self.output_dim = out_features

    def forward(self, x):
        return x.view(-1, self.output_dim)


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, n_filters, kernel_sizes=[11,21,41], bottleneck_channels=32, activation=nn.ReLU(), strides=[1,1,1], use_residual=True):
        """
        : param in_channels				Number of input channels (input features)
        : param n_filters				Number of filters per convolution layer => out_channels = 4*n_filters
        : param kernel_sizes			List of kernel sizes for each convolution.
                                        Each kernel size must be odd number that meets -> "kernel_size % 2 !=0".
                                        This is nessesery because of padding size.
                                        For correction of kernel_sizes use function "correct_sizes".
        : param bottleneck_channels		Number of output channels in bottleneck.
                                        Bottleneck wont be used if nuber of in_channels is equal to 1.
        : param activation				Activation function for output tensor (nn.ReLU()).
        """
        super(InceptionBlock, self).__init__()
        self.use_residual = use_residual
        self.activation = activation
        self.inception_1 = InceptionModule(
            in_channels=in_channels,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            activation=activation,
            stride=strides[0]
        )
        self.inception_2 = InceptionModule(
            in_channels=4 * n_filters,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            activation=activation,
            stride=strides[1]
        )
        self.inception_3 = InceptionModule(
            in_channels=4 * n_filters,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            activation=activation,
            stride=strides[2]
        )
        if self.use_residual:
            strides_total = strides[0]*strides[1]*strides[2]
            if strides_total == 1:
                self.residual = nn.Sequential(
                    nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=4 * n_filters,
                        kernel_size=1,
                        stride=1,
                        padding=0
                    ),
                    nn.BatchNorm1d(num_features=4 * n_filters)
                )
            else:
                """if strides_total % 2 == 0:
                    padding = 0 #math.floor(float(strides_total)/2)
                else:
                    padding = math.ceil(float(strides_total)/2)"""
                self.residual = nn.Sequential(
                    nn.MaxPool1d(kernel_size=strides_total, stride=strides_total, padding=strides_total//2),# padding=(math.ceil(float(strides_total-1)/2), math.floor(float(strides_total)/2))),
                    nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=4 * n_filters,
                        kernel_size=1,
                        stride=1,
                        padding=0
                    ),
                    nn.BatchNorm1d(num_features=4 * n_filters)
                )

    def forward(self, x):
        #print(x.shape)
        z = self.inception_1(x)
        z = self.inception_2(z)
        z = self.inception_3(z)

        if self.use_residual:
            #print(z.shape)
            z = z + self.residual(x)[...,:z.shape[-1]]
            z = self.activation(z)

        return z


class InceptionTime(nn.Module):
    def __init__(self, in_channels, label_dim, n_filters=32, kernel_sizes=[9, 19, 39], bottleneck_channels=32,
                 use_residual=True, activation=nn.ReLU(), dropout=0):
        super(InceptionTime, self).__init__()
        self.use_residual = use_residual
        self.activation = activation
        self.label_dim = label_dim
        self.inception_1 = InceptionModule(
            in_channels=in_channels,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            activation=activation,
            stride=1,
            dropout=dropout
        )
        self.inception_2 = InceptionModule(
            in_channels=4 * n_filters,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            activation=activation,
            stride=1,
            dropout=dropout
        )
        self.inception_3 = InceptionModule(
            in_channels=4 * n_filters,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            activation=activation,
            stride=1,
            dropout=dropout
        )
        if self.use_residual:
            self.residual = nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=4 * n_filters,
                    kernel_size=1,
                    stride=1,
                    padding=0
                ),
                nn.BatchNorm1d(
                    num_features=4 * n_filters
                )
            )

        self.last_layers = nn.Sequential(
            nn.AdaptiveAvgPool1d(output_size=1),
            Flatten(out_features=32 * 4 * 1),
            nn.Linear(in_features=4 * 32 * 1, out_features=self.label_dim)
        )

    def forward(self, x):
        nan_mask = ~x.isnan().any(axis=1).unsqueeze(1)
        x[~nan_mask] = 0
        z = self.inception_1(x)
        z = self.inception_2(z)
        z = self.inception_3(z)

        if self.use_residual:
            z = z + self.residual(x)
            z = self.activation(z)

        z = self.last_layers[0](z)
        z = self.last_layers[1](z)
        z_out = self.last_layers[2](z)

        return dict(output=z_out, features=z)


class InceptionTimeDeep(nn.Module):
    def __init__(self, in_channels, label_dim, n_filters=32, kernel_sizes=[11,21,41], bottleneck_channels=32,
                 use_residual=True, activation=nn.ReLU(), num_blocks=2, strides=[[1,2,1], [1,2,1]]):
        super(InceptionTimeDeep, self).__init__()
        self.use_residual = use_residual
        self.activation = activation
        self.label_dim = label_dim
        blocks = []
        for i in range(num_blocks):
            blocks.append(InceptionBlock(
                in_channels=in_channels,
                n_filters=n_filters,
                kernel_sizes=kernel_sizes,
                bottleneck_channels=bottleneck_channels,
                activation=activation,
                strides=strides[i],
                use_residual=use_residual
            ))
            in_channels = 4 * n_filters
        self.blocks = nn.Sequential(*blocks)
        """self.inception_1 = InceptionModule(
            in_channels=in_channels,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            activation=activation,
        )
        self.inception_2 = InceptionModule(
            in_channels=4 * n_filters,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            activation=activation,
        )
        self.inception_3 = InceptionModule(
            in_channels=4 * n_filters,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            activation=activation,
        )
        if self.use_residual:
            self.residual = nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=4 * n_filters,
                    kernel_size=1,
                    stride=1,
                    padding=0
                ),
                nn.BatchNorm1d(
                    num_features=4 * n_filters
                )
            )"""

        self.last_layers = nn.Sequential(
            nn.AdaptiveAvgPool1d(output_size=1),
            Flatten(out_features=n_filters * 4 * 1),
            nn.Linear(in_features=4 * n_filters * 1, out_features=self.label_dim),
        )

    def forward(self, x):
        """z = self.inception_1(x)
        z = self.inception_2(z)
        z = self.inception_3(z)"""
        nan_mask = ~x.isnan().any(axis=1).unsqueeze(1)
        x[~nan_mask] = 0
        z = self.blocks(x)

        """if self.use_residual:
            z = z + self.residual(x)
            z = self.activation(z)"""

        z = self.last_layers[0](z)
        z = self.last_layers[1](z)
        z_out = self.last_layers[2](z)

        return dict(output=z_out, features=z)


class InceptionModule(nn.Module):
    def __init__(self, in_channels, n_filters, kernel_sizes=[9, 19, 39], bottleneck_channels=32, activation=nn.ReLU(), stride=1, dropout=0):
        """
        : param in_channels				Number of input channels (input features)
        : param n_filters				Number of filters per convolution layer => out_channels = 4*n_filters
        : param kernel_sizes			List of kernel sizes for each convolution.
                                        Each kernel size must be odd number that meets -> "kernel_size % 2 !=0".
                                        This is nessesery because of padding size.
                                        For correction of kernel_sizes use function "correct_sizes".
        : param bottleneck_channels		Number of output channels in bottleneck.
                                        Bottleneck wont be used if nuber of in_channels is equal to 1.
        : param activation				Activation function for output tensor (nn.ReLU()).
        """
        super(InceptionModule, self).__init__()
        self.dropout = nn.Dropout(dropout)
        if in_channels > 1:
            self.bottleneck = nn.Conv1d(
                in_channels=in_channels,
                out_channels=bottleneck_channels,
                kernel_size=1,
                stride=1,
                bias=False
            )
        else:
            self.bottleneck = None
            bottleneck_channels = 1

        self.conv_from_bottleneck_1 = nn.Conv1d(
            in_channels=bottleneck_channels,
            out_channels=n_filters,
            kernel_size=kernel_sizes[0],
            stride=stride,
            padding=kernel_sizes[0] // 2,
            bias=False
        )
        self.conv_from_bottleneck_2 = nn.Conv1d(
            in_channels=bottleneck_channels,
            out_channels=n_filters,
            kernel_size=kernel_sizes[1],
            stride=stride,
            padding=kernel_sizes[1] // 2,
            bias=False
        )
        self.conv_from_bottleneck_3 = nn.Conv1d(
            in_channels=bottleneck_channels,
            out_channels=n_filters,
            kernel_size=kernel_sizes[2],
            stride=stride,
            padding=kernel_sizes[2] // 2,
            bias=False
        )
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv_from_maxpool = nn.Conv1d(
            in_channels=in_channels,
            out_channels=n_filters,
            kernel_size=1,
            stride=stride,
            padding=0,
            bias=False
        )
        self.batch_norm = nn.BatchNorm1d(num_features=4 * n_filters)
        self.activation = activation

    def forward(self, x):
        # step 1
        if self.bottleneck is None:
            Z_bottleneck = x
        else:
            Z_bottleneck = self.bottleneck(x)

        Z_maxpool = self.max_pool(x)
        # step 2
        Z1 = self.conv_from_bottleneck_1(Z_bottleneck)
        Z2 = self.conv_from_bottleneck_2(Z_bottleneck)
        Z3 = self.conv_from_bottleneck_3(Z_bottleneck)
        Z4 = self.conv_from_maxpool(Z_maxpool)
        # step 3
        z = torch.cat([Z1, Z2, Z3, Z4], axis=1)
        z = self.dropout(z)
        z = self.activation(self.batch_norm(z))
        return z
