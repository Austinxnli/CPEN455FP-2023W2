import torch.nn as nn
from layers import *


class PixelCNNLayer_up(nn.Module):
    def __init__(self, nr_resnet, nr_filters, resnet_nonlinearity):
        super(PixelCNNLayer_up, self).__init__()
        self.nr_resnet = nr_resnet
        self.u_stream = nn.ModuleList([gated_resnet(nr_filters, down_shifted_conv2d, resnet_nonlinearity, skip_connection=0, use_labels=True)
                                        for _ in range(nr_resnet)])
        self.ul_stream = nn.ModuleList([gated_resnet(nr_filters, down_right_shifted_conv2d, resnet_nonlinearity, skip_connection=1, use_labels=True)
                                        for _ in range(nr_resnet)])

    def forward(self, u, ul, labels):
        u_list, ul_list = [], []
        for i in range(self.nr_resnet):
            u = self.u_stream[i](u, labels=labels)
            ul = self.ul_stream[i](ul, a=u, labels=labels)
            u_list += [u]
            ul_list += [ul]
        return u_list, ul_list


class PixelCNNLayer_down(nn.Module):
    def __init__(self, nr_resnet, nr_filters, resnet_nonlinearity):
        super(PixelCNNLayer_down, self).__init__()
        self.nr_resnet = nr_resnet
        self.u_stream = nn.ModuleList([gated_resnet(nr_filters, down_shifted_conv2d, resnet_nonlinearity, skip_connection=1, use_labels=True)
                                       for _ in range(nr_resnet)])
        self.ul_stream = nn.ModuleList([gated_resnet(nr_filters, down_right_shifted_conv2d, resnet_nonlinearity, skip_connection=2, use_labels=True)
                                        for _ in range(nr_resnet)])

    def forward(self, u, ul, u_list, ul_list, labels):
        for i in range(self.nr_resnet):
            u = self.u_stream[i](u, a=u_list.pop(), labels=labels)
            ul = self.ul_stream[i](ul, a=torch.cat((u, ul_list.pop()), 1), labels=labels)
        return u, ul


class PixelCNN(nn.Module):
    def __init__(self, nr_resnet=5, nr_filters=80, nr_logistic_mix=10,
                 resnet_nonlinearity='concat_elu', input_channels=3, num_classes=10):
        super(PixelCNN, self).__init__()
        self.nr_filters = nr_filters
        self.input_channels = input_channels
        self.nr_logistic_mix = nr_logistic_mix
        self.label_embedding = nn.Embedding(num_classes, nr_filters)
        
        # Define non-linearity
        if resnet_nonlinearity == 'concat_elu':
            self.resnet_nonlinearity = lambda x : concat_elu(x)
        else:
            raise NotImplementedError('Currently, only concat_elu nonlinearity is supported.')

        # Initial convolution layers
        self.u_init = down_shifted_conv2d(input_channels + 1, nr_filters, filter_size=(2,3), shift_output_down=True)
        self.ul_init = nn.ModuleList([
            down_shifted_conv2d(input_channels + 1, nr_filters, filter_size=(1,3), shift_output_down=True),
            down_right_shifted_conv2d(input_channels + 1, nr_filters, filter_size=(2,1), shift_output_right=True)
        ])

        # Define up and down layers
        self.down_layers = nn.ModuleList([PixelCNNLayer_down(nr_resnet + i, nr_filters, self.resnet_nonlinearity) for i in range(3)])
        self.up_layers = nn.ModuleList([PixelCNNLayer_up(nr_resnet, nr_filters, self.resnet_nonlinearity) for _ in range(3)])

        # Downsampling and upsampling layers
        self.downsize_u_stream = nn.ModuleList([down_shifted_conv2d(nr_filters, nr_filters, stride=(2,2)) for _ in range(2)])
        self.downsize_ul_stream = nn.ModuleList([down_right_shifted_conv2d(nr_filters, nr_filters, stride=(2,2)) for _ in range(2)])
        self.upsize_u_stream = nn.ModuleList([down_shifted_deconv2d(nr_filters, nr_filters, stride=(2,2)) for _ in range(2)])
        self.upsize_ul_stream = nn.ModuleList([down_right_shifted_deconv2d(nr_filters, nr_filters, stride=(2,2)) for _ in range(2)])

        # Output layer
        self.nin_out = nin(nr_filters, 3 * nr_logistic_mix if input_channels == 1 else 10 * nr_logistic_mix)
        self.init_padding = None

    def forward(self, x, labels, sample=False):
        # Embed labels and reshape for broadcasting across the spatial dimensions
        labels = self.label_embedding(labels).unsqueeze(-1).unsqueeze(-1)
        labels = labels.expand(-1, -1, x.size(2), x.size(3))

        # Initialize padding if not already done
        if self.init_padding is None:
            self.init_padding = torch.ones(x.size(0), 1, x.size(2), x.size(3), device=x.device, requires_grad=False)

        # Concatenate input with padding for border pixels
        x = torch.cat((x, self.init_padding), 1) if not sample else x

        # Up pass
        u = self.u_init(x) + labels
        ul = self.ul_init[0](x) + self.ul_init[1](x) + labels
        u_list, ul_list = [u], [ul]
        for layer in self.up_layers:
            u_out, ul_out = layer(u_list[-1], ul_list[-1], labels)
            u_list.extend(u_out)
            ul_list.extend(ul_out)

            # Downsample if not the last layer
            if layer is not self.up_layers[-1]:
                u_list.append(self.downsize_u_stream[u_list[-1]])
                ul_list.append(self.downsize_ul_stream[ul_list[-1]])

        # Down pass
        u, ul = u_list.pop(), ul_list.pop()
        for layer in self.down_layers:
            u, ul = layer(u, ul, u_list, ul_list, labels)

            # Upsample if not the last layer
            if layer is not self.down_layers[-1]:
                u = self.upsize_u_stream[u]
                ul = self.upsize_ul_stream[ul]

        # Output layer
        out = self.nin_out(F.elu(ul))
        return out
    
    
class random_classifier(nn.Module):
    def __init__(self, NUM_CLASSES):
        super(random_classifier, self).__init__()
        self.NUM_CLASSES = NUM_CLASSES
        self.fc = nn.Linear(3, NUM_CLASSES)
        print("Random classifier initialized")
        # create a folder
        if 'models' not in os.listdir():
            os.mkdir('models')
        torch.save(self.state_dict(), 'models/conditional_pixelcnn.pth')
    def forward(self, x, device):
        return torch.randint(0, self.NUM_CLASSES, (x.shape[0],)).to(device)
    
    