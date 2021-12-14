import torch
from torch import nn
import torchvision.models

from typing import List


class CombineMultiModel(nn.Module):
    """
        This will combine list of models have the same input
        The new output is the tensor that have output all model concat to one
    """

    def __init__(self, models):
        super(CombineMultiModel, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))

        return torch.cat(outputs, dim=1)


class EnsembleMultiModel(nn.Module):
    """
        This class will ensemble list of models have same input and output
        The new output will equal the mean of older outputs
    """

    def __init__(self, models: List):
        super(EnsembleMultiModel, self).__init__()
        self.models = models

    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        stack_output = torch.stack(outputs)
        output = torch.mean(stack_output, dim=0)

        return output


class CombineBaseModelWithClassifier(nn.Module):
    """
        Combine two part of model into one (backbone and classifier)
    """

    def __init__(self, backbone, classifier):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)

        return x


class MergeMultiTaskModel(nn.Module):
    """
        All model used to merge need to have 2 attribute .backbone and .classifier
        Where .backbone of all models are similar.
    """

    def __init__(self, models):
        super().__init__()
        self.num_models = len(models)
        self.backbone = models[0].backbone
        self.classifier = []
        for model_idx in range(self.num_models):
            self.classifier.append(models[model_idx].classifier)

    def forward(self, x):
        x = self.backbone(x)
        outputs = self.classifier[0](x)
        for model_idx in range(1, self.num_models):
            outputs = torch.cat((outputs, self.classifier[model_idx](x)), dim=1)

        return outputs


class ModelsGenerator:
    """
        To use this generator, you need to define new classifier by using create_new_classifier method.
        Generate new model base on pytorch model (Apply to all models in torchvision except Inception and GoogleNet)
        New model has a new classifier and connect with backbone by dropout layer with r=0

        Method:
            create_new_classifier (must overwrite): define the new classifier
            create_model: generate a normal model (one backbone and one classifier)
            create_multitask_model: generate a multitask model (one model and multi classifier)
    """

    def __init__(self, base_model):
        self.base_model = base_model

        # Determine classification layer name (because each model has a different name)
        self.classification_layer_name = None
        if hasattr(self.base_model, 'classifier'):
            self.classification_layer_name = 'classifier'
        elif hasattr(self.base_model, 'fc'):
            self.classification_layer_name = 'fc'
        else:
            raise ValueError(type(self).__name__ + ": This model has not contain fc and classifier layer")

    def create_new_classifier(self, num_classes):
        return nn.Sequential

    def create_single_model(self, num_classes):
        # Delete old classification layer
        setattr(self.base_model, self.classification_layer_name, nn.Dropout(p=0))
        new_classifier = self.create_new_classifier(num_classes)

        return CombineBaseModelWithClassifier(self.base_model, new_classifier)

    def create_multitask_model(self, num_models, num_cls_per_model):
        if type(num_cls_per_model) is int:
            num_cls_per_model = [num_cls_per_model] * num_models

        # Delete old classification layer
        setattr(self.base_model, self.classification_layer_name, nn.Dropout(p=0))

        models = nn.ModuleList()
        for model_idx in range(num_models):
            new_classifier = self.create_new_classifier(num_cls_per_model[model_idx])
            models.append(CombineBaseModelWithClassifier(self.base_model, new_classifier))

        return models


class EfficientNet(ModelsGenerator):
    def __init__(self, version):
        efficientnet_b = getattr(torchvision.models, 'efficientnet_b' + str(version))(pretrained=True)
        super().__init__(efficientnet_b)
        self.dropout_p = self.base_model.classifier[0].p
        self.in_features = self.base_model.classifier[1].in_features

    def create_new_classifier(self, num_classes):
        return nn.Sequential(nn.Dropout(p=self.dropout_p, inplace=True),
                             nn.Linear(self.in_features, num_classes),
                             nn.Sigmoid())


class ResNext(ModelsGenerator):
    def __init__(self, version):
        resnext = getattr(torchvision.models, 'resnext' + str(version))(pretrained=True)
        super().__init__(resnext)
        self.in_features = self.base_model.fc.in_features

    def create_new_classifier(self, num_classes):
        return nn.Sequential(nn.Linear(self.in_features, num_classes),
                             nn.Sigmoid())


class RegNet(ModelsGenerator):
    def __init__(self, version):
        regnet = getattr(torchvision.models, 'regnet' + str(version))(pretrained=True)
        super().__init__(regnet)
        self.in_features = self.base_model.fc.in_features

    def create_new_classifier(self, num_classes):
        return nn.Sequential(nn.Linear(self.in_features, num_classes),
                             nn.Sigmoid())


class VGG_bn(ModelsGenerator):
    def __init__(self, version):
        vgg_bn = getattr(torchvision.models, 'vgg' + str(version) + '_bn')(pretrained=True)
        super().__init__(vgg_bn)
        self.dropout_p = self.base_model.classifier[2].p

    def create_new_classifier(self, num_classes):
        return nn.Sequential(nn.Linear(512 * 7 * 7, 4096),
                             nn.ReLU(True),
                             nn.Dropout(p=self.dropout_p),
                             nn.Linear(4096, 4096),
                             nn.ReLU(True),
                             nn.Dropout(p=self.dropout_p),
                             nn.Linear(4096, num_classes),
                             nn.Sigmoid())


class ResNet(ModelsGenerator):
    def __init__(self, version):
        super().__init__(getattr(torchvision.models, 'resnet' + str(version))(pretrained=True))
        self.in_features = self.base_model.fc.in_features

    def create_new_classifier(self, num_classes):
        return nn.Sequential(nn.Linear(self.in_features, num_classes),
                             nn.Sigmoid())


class DenseNet(ModelsGenerator):
    def __init__(self, version):
        super().__init__(getattr(torchvision.models, 'densenet' + str(version))(pretrained=True))
        self.in_features = self.base_model.classifier.in_features

    def create_new_classifier(self, num_classes):
        return nn.Sequential(nn.Linear(self.in_features, num_classes),
                             nn.Sigmoid())


MODEL_GEN = {'efficientnet': EfficientNet,
             'vgg_bn': VGG_bn,
             'regnet': RegNet,
             'resnext': ResNext,
             'resnet': ResNet,
             'densenet': DenseNet, }


def turn_off_trainable_backbone(model: ModelsGenerator):
    """
        This function only work with the model create by ModelsGenerator class
    """
    for name, param in model.backbone.named_parameters():
        param.requires_grad = False

    return model


if __name__ == '__main__':
    pass
    # from torchvision.io import read_image
    # from torchvision.transforms import Resize
    # import torch
    #
    # image1 = read_image(r'D:\Machine Learning Project\5kCompliance\dataset\train\images\2.jpg').float()
    # image1 = Resize((300, 300))(image1)
    # image2 = read_image(r'D:\Machine Learning Project\5kCompliance\dataset\train\images\21.jpg').float()
    # image2 = Resize((300, 300))(image1)
    # image = torch.stack((image1, image2), dim=0)
    # print(image.size())
    #
    # models = DenseNet(version=201).create_single_model(num_classes=1)
    # print(type(models))
    # # print(models)
    # # model = MergeMultiTaskModel(models)
    #
    # # print(model(image))
