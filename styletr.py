#import lib.copy as copy
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torchvision import transforms
from PIL import Image


def gram_matrix(input):
    batch_size, h, w, f_map_num = input.size()  # batch_size = 1
    # h, w - dimensions of a feature map
    # f_map_num - number of a future map
    features = input.view(batch_size * h, w * f_map_num)
    G = torch.mm(features, features.t())  # matrix multiplicate for gram matrix
    G = G.div(batch_size * h * w * f_map_num)  # normalize Gramm matrix
    return G


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean.clone().detach()).view(-1, 1, 1)
        self.std = torch.tensor(std.clone().detach()).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std  # normalize img

class ContentLoss(nn.Module):
    def __init__(self, target, ):
        super(ContentLoss, self).__init__()
        self.target = target.detach()  # detach from calculation graph
        self.loss = F.mse_loss(self.target, self.target)  # some initialization

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)  # mse from target image
        # and content at the moment image
        return input

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()  # detach from calculation graph
        self.loss = F.mse_loss(self.target, self.target)  # some initialization

    def forward(self, input):
        G = gram_matrix(input)  # G-matrix features
        self.loss = F.mse_loss(G, self.target)
        return input

class StyleTransfer:

    def image_loader(self, image_name):
        imsize = 256
        loader = transforms.Compose([
            transforms.Resize(imsize),
            transforms.CenterCrop(imsize),
            transforms.ToTensor()
        ])
        image = Image.open(image_name)
        image = loader(image).unsqueeze(0)  # Добавляем размерность(размер батча=1) для тензора
        return image


    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])




    content_layers_default = ['conv_4']
    style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
    cnn = models.vgg13(pretrained=True, )

    def get_input_optimizer(self, input_img):
        optimizer = optim.LBFGS([input_img.requires_grad_()])
        return optimizer

    def get_style_model_and_losses(self, cnn, normalization_mean, normalization_std,
                                   style_img, content_img,
                                   content_layers=content_layers_default,
                                   style_layers=style_layers_default):
        cnn = copy.deepcopy(cnn)
        normalization = Normalization(normalization_mean, normalization_std)
        content_losses = []
        style_losses = []
        model = nn.Sequential(normalization)

        i = 0
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)
            if name in content_layers:
                target = model(content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_lossL{}".format(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                target_feature = model(style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break
        model = model[:(i + 1)]
        return model, style_losses, content_losses


    def run_style_transfer(self, cnn, normalization_mean, normalization_std,
                           content_img, style_img, input_img, num_steps=50,
                           style_weight=100000, content_weight=1):
        model, style_losses, content_losses = self.get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                                                                         style_img, content_img)
        optimizer = self.get_input_optimizer(input_img)
        run = [0]
        def doSome():
            def closure():
                input_img.data.clamp_(0, 1)
                optimizer.zero_grad()
                model(input_img)

                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss

                style_score *= style_weight
                content_score *= content_weight

                loss = style_score + content_score
                loss.backward()
                run[0] += 1
                if run[0] % 50 == 0:
                    print('run {}:'.format(run))
                    print('Style Loss: {:4f} Content Loss: {:4f}'.format(style_score.item(),
                                                                         content_score.item()))
                    print()
                return style_score + content_score

            optimizer.step(closure)

        while run[0] <= num_steps:

            doSome()
        input_img.data.clamp_(0, 1)
        return input_img

    def __init__(self, pic, style):
        self.style = style
        self.pic = pic

    def getRes(self):
        self.style_img = self.image_loader(self.style)
        self.content_img = self.image_loader(self.pic)
        self.input_img = self.content_img.clone()




        self.output = self.run_style_transfer(self.cnn, self.cnn_normalization_mean,
                                              self.cnn_normalization_std, self.content_img, self.style_img,
                                              self.input_img)

        return self.output

#(img * 255).astype(np.uint8)
#