import torch

class DissimilarityMetric:
    """compute the distance of two featuremaps
    """
    def __init__(self, metric):
        self.__metric = metric

    def __call__(self, A, B, **kwargs):
        if self.__metric == "vanilla":
            return self.__vanilla(A, B)
        elif self.__metric == "cosine":
            return self.__cosine_similarity(A, B, **kwargs)
        else:
            raise ValueError("Unknown metric")

    def __vanilla(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """compute the vanilla distance between two featuremaps, Frobenius norm

        Args:
            A (torch.Tensor): the featuremap A, shape: (N, D)
            B (torch.Tensor): the featuremap B, shape: (N, D)
        """
        assert A.shape == B.shape, \
            f"the shape of A - {A.shape} should be the same as the shape of B - {B.shape}"
        A = A.view(A.shape[0], -1) # shape: (N, D)
        B = B.view(B.shape[0], -1) # shape: (N, D)

        def norm_square(A: torch.Tensor) -> torch.Tensor:
            return torch.sum(A ** 2) # shape: (1, )

        return norm_square(A - B) / torch.norm(A, p="fro") / torch.norm(B, p="fro")


    def __cosine_similarity(self, A: torch.Tensor, B: torch.Tensor, **kwargs) -> torch.Tensor:
        """compute the cosine similarity between two matrices

        dist = 1 - <A, B> / (||A|| * ||B||)

        Args:
            A (torch.Tensor): the featuremap A, shape: (N, D)
            B (torch.Tensor): the featuremap B, shape: (N, D)
            kwargs: the keyword arguments

        Return:
            dist (torch.Tensor): the distance between A and B
        """
        get_coef = kwargs.get("get_coef", False)

        assert A.shape == B.shape, \
            f"the shape of A - {A.shape} should be the same as the shape of B - {B.shape}"

        A = A.view(A.shape[0], -1).double() # shape: (N, D)
        B = B.view(B.shape[0], -1).double() # shape: (N, D)

        # compute the frobenius inner product of A and B
        inner_product = torch.sum(A * B) # shape: (1, 1)
        # compute the frobenius norm of A and B
        A_norm = torch.norm(A, p="fro") # shape: (1, 1)
        B_norm = torch.norm(B, p="fro") # shape: (1, 1)

        # cal the distance
        dist = 1 - torch.abs(inner_product) / (A_norm * B_norm)

        # compute the coefficient
        coef = inner_product / (B_norm ** 2)

        assert torch.abs(inner_product) <= A_norm * B_norm * (1 + 1e-10), \
            f"the inner product - {inner_product} should be less than the product of the norm - {A_norm * B_norm}"

        if get_coef:
            return dist, coef
        else:
            return dist


class DissimilarityMetricOverSamples:
    """compute the distance of two featuremaps, for each sample
    """
    def __init__(self, metric):
        self.__metric = metric

    def __call__(self, A, B, **kwargs):
        if self.__metric == "vanilla":
            return self.__vanilla(A, B)
        elif self.__metric == "cosine":
            return self.__cosine_similarity(A, B, **kwargs)

    def __vanilla(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """compute the vanilla distance between two featuremaps, Frobenius norm

        Args:
            A (torch.Tensor): the featuremap A, shape: (N, D)
            B (torch.Tensor): the featuremap B, shape: (N, D)
        """
        assert A.shape == B.shape, \
            f"the shape of A - {A.shape} should be the same as the shape of B - {B.shape}"
        A = A.view(A.shape[0], -1) # shape: (N, D)
        B = B.view(B.shape[0], -1) # shape: (N, D)

        def norm_square(A: torch.Tensor) -> torch.Tensor:
            return torch.sum(A ** 2, dim=-1) # shape: (N, )
        return norm_square(A - B) / torch.norm(A, p="fro", dim=-1) / torch.norm(B, p="fro", dim=-1) # shape: (N, )


    def __cosine_similarity(self, A: torch.Tensor, B: torch.Tensor, **kwargs) -> torch.Tensor:
        """compute the cosine similarity between two matrices

        dist = 1 - <A, B> / (||A|| * ||B||)

        Args:
            A (torch.Tensor): the featuremap A, shape: (N, D)
            B (torch.Tensor): the featuremap B, shape: (N, D)
            kwargs: the keyword arguments

        Return:
            dist (torch.Tensor): the distance between A and B
        """
        get_coef = kwargs.get("get_coef", False)

        assert A.shape == B.shape, \
            f"the shape of A - {A.shape} should be the same as the shape of B - {B.shape}"

        A = A.view(A.shape[0], -1).double() # shape: (N, D)
        B = B.view(B.shape[0], -1).double() # shape: (N, D)

        # compute the frobenius inner product of A and B
        inner_product = torch.sum(A * B, dim=-1) # shape: (N, )
        # compute the frobenius norm of A and B
        A_norm = torch.norm(A, p="fro", dim=-1) # shape: (N, )
        B_norm = torch.norm(B, p="fro", dim=-1) # shape: (N, )

        # cal the distance
        dist = 1 - torch.abs(inner_product) / (A_norm * B_norm) # shape: (N, )

        # compute the coefficient
        coef = inner_product / (B_norm ** 2) # shape: (N, )

        if get_coef:
            return dist, coef
        else:
            return dist
        

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from functools import reduce
from collections import defaultdict
from tqdm import tqdm

# import internal libs
from utils import get_logger

class FeatureMap:
    """class used to extract feature map from a given model
    """
    def __init__(self,
                 device: torch.device,
                 model: nn.Module,) -> None:
        """initialize the feature map extractor

        Args:
            device (torch.device): device to run the model
            model (nn.Module): model to extract feature map
        """
        self.__device = device
        self.__model = model.to(device)
        self.__model_name = model.__class__.__name__
            
        # make sure the model is in eval mode
        self.__model.eval()
        # initialize the hooks
        self.__hooks = {}
        self.__featuremaps = defaultdict(list)
        
    def __get_module(self, module_name: str) -> nn.Module:
        """get the module from the model

        Args:
            module_name (str): name of the module

        Returns:
            nn.Module: the module
        """
        return reduce(getattr, module_name.split('.'), self.__model)

    def __get_conv_fc_layer_names(self) -> list:
        """get the names of the conv layers and the fc layers

        Returns:
            list: names of the conv layers
        """
        layer_names = []
        for name, module in self.__model.named_modules():
            if self.__model_name == 'ResNet':
                import model.cifar_resnet as resnet
                import model.cifar_resnet_plus as resnet_plus
                if isinstance(module, (resnet.BasicBlock, resnet.Bottleneck, \
                    resnet_plus.BasicBlock, nn.Linear)):
                    layer_names.append(name)
                elif name == "conv1":
                    layer_names.append(name)
            elif self.__model_name == "DenseNet":
                if isinstance(module, (nn.Sequential, nn.Linear)):
                    layer_names.append(name)
            else:
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    layer_names.append(name)
        return layer_names

    def __get_conv_fc_modules(self, layer_names: list = None) -> dict:
        """get the conv and fc modules

        Args:
            layer_names (list, optional): names of the layers. Defaults to None.

        Returns:
            dict: conv modules
        """
        if layer_names is None:
            layer_names = self.__get_conv_fc_layer_names()
        modules = dict()
        for name in layer_names:
            modules[name] = self.__get_module(name)
        return modules
    
    def __register_single_hook(self, 
                               layer_name: str,
                               module: nn.Module,
                               get_input: bool) -> None:
        """register the hook for a single module

        Args:
            layer_name (str): name of the layer
            module (nn.Module): module to register the hook
            get_input (bool): whether to get the input of the model
        """
        def forward_hook(module, input, output):
            if get_input:
                assert len(input) == 1, "only support single input"
                self.__featuremaps[layer_name].append(input[0].detach().cpu())
            else:
                self.__featuremaps[layer_name].append(output.detach().cpu())
        self.__hooks[layer_name] = module.register_forward_hook(forward_hook)

    def __register_hooks(self, 
                         layer_names: list = None,
                         get_input: bool = False) -> None:
        """register the hooks

        Args:
            layer_names (list, optional): names of the layers to register the hooks. Defaults to None.
            get_input (bool, optional): whether to get the input of the model. Defaults to False.
        """
        conv_modules = self.__get_conv_fc_modules(layer_names)
        for name, module in conv_modules.items():
            self.__register_single_hook(name, module, get_input)

    def __remove_hooks(self) -> None:
        """remove the hooks
        """
        for hook in self.__hooks.values():
            hook.remove()
        self.__hooks = {}

    def __remove_featuremaps(self) -> None:
        """remove the feature maps
        """
        self.__featuremaps = defaultdict(list)

    def __convert_featuremaps_to_tensor(self) -> dict:
        """convert the feature maps to tensor

        Returns:
            dict: feature maps
        """
        featuremaps = dict()
        for key, value in self.__featuremaps.items():
            featuremaps[key] = torch.cat(value, dim=0)
        return featuremaps

    def get_featuremaps(self,
                        dataloader: DataLoader,
                        layer_names: list = None,
                        get_input: bool = False) -> tuple:
        """get the feature maps from the conv layers
            tips: use the hook to get the feature maps

        Args:
            dataloader (DataLoader): data loader to load the data
            layer_names (list): names of the layers to extract feature map. Defaults to None.
            get_input (bool): whether to get the input of the model. Defaults to False.

        Returns:
            (feature maps, preds): feature maps and preds
        """
        logger = get_logger(
            f"{__name__}.{self.__class__.__name__}.get_featuremaps"
        )

        # clear the feature maps
        self.__remove_featuremaps()
        
        # register the hooks
        self.__register_hooks(layer_names, get_input)

        # initialize the preds
        preds = []

        # get the feature maps
        with torch.no_grad():
            for inputs, labels in tqdm(dataloader):
                # move the data to the device
                inputs = inputs.to(self.__device)
                # forward
                outputs = self.__model(inputs)
                # update predictions
                assert len(outputs.shape) == 2
                preds.extend(outputs.argmax(dim=-1).detach().cpu().numpy())
        
        # convert the feature maps to tensor
        featuremaps = self.__convert_featuremaps_to_tensor()
        # clear the feature _maps
        self.__remove_featuremaps()
        # remove the hooks
        self.__remove_hooks()

        logger.info(f"feature maps: {featuremaps.keys()}")
        logger.info(f"preds[:10]: {preds[:10]}")
        return featuremaps, torch.tensor(preds)