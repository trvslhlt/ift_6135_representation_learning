import torch
import unittest
from mlp import Linear, MLP
from mobileNet import DepthwiseSeparableConv, MobileNet
from unet import UNet
from utils import DiceLoss
import monai

class TestLinear(unittest.TestCase):
    def test_linear_attributes(self):
        in_feat = 30
        out_feat = 20
        my_linear = Linear(in_features=in_feat, out_features=out_feat)
        assert hasattr(my_linear, 'weight')
        assert hasattr(my_linear, 'bias')
        
        assert len(my_linear.weight.shape) == 2
        assert my_linear.weight.shape[0] == out_feat
        assert my_linear.weight.shape[1] == in_feat
        
        assert len(my_linear.bias.shape) == 1
        assert my_linear.bias.shape[0] == out_feat
    
    def test_linear_forward(self):
        in_feat = 30
        out_feat = 20
        my_linear = Linear(in_features=in_feat, out_features=out_feat)
        
        gt_linear = torch.nn.Linear(in_features=in_feat, out_features=out_feat)
        my_linear.weight.data[:] = gt_linear.weight.data
        my_linear.bias.data[:] = gt_linear.bias.data
        
        batch = 10
        inputs = torch.randn(batch, in_feat)
        my = my_linear(inputs)
        assert len(my.shape) == 2
        assert my.shape[0] == batch
        assert my.shape[1] == out_feat
        
        gt = gt_linear(inputs)
        assert torch.allclose(my, gt)
        
class TestMLP(unittest.TestCase):
    input_size = 50
    hidden_sizes = [100, 200]
    output_size = 20
    batch = 10
    
    def test_mlp(self):
        model = MLP(self.input_size, self.hidden_sizes, self.output_size)
        assert len(model.hidden_layers) == len(self.hidden_sizes)
        
        sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
        for layer_id, layer in enumerate(model.hidden_layers + [model.output_layer]):
            assert isinstance(layer, Linear)
            in_feat = sizes[layer_id]
            out_feat = sizes[layer_id + 1]
            assert layer.weight.shape[0] == out_feat
            assert layer.weight.shape[1] == in_feat
    
    def test_activation(self):
        model = MLP(self.input_size, self.hidden_sizes, self.output_size)
        inputs = torch.randn(self.batch, self.input_size)
        
        names = ['relu', 'tanh', 'sigmoid']
        gtfuncs = [
            torch.relu, 
            torch.tanh, 
            torch.sigmoid]
        
        for activation_name, gtfunc in zip(names, gtfuncs):
            gt = gtfunc(inputs)
            my = model.activation_fn(activation_name, inputs)
            assert torch.allclose(my, gt)
    
    def test_forward(self):     
        model = MLP(self.input_size, self.hidden_sizes, self.output_size)
        inputs = torch.randn(self.batch, self.input_size)
        outputs = model(inputs)
        assert len(outputs.shape) == 2
        assert outputs.shape[0] == self.batch
        assert outputs.shape[1] == self.output_size


class TestMobileNet(unittest.TestCase):
    def test_DepthwiseSeparableBlock(self):
        block = DepthwiseSeparableConv(64, 64, 2, 1)
        inputs = torch.randn(32, 64, 8, 8)
        outputs = block(inputs)
        assert len(outputs.shape) == 4
        assert outputs.shape[0] == 32
        assert outputs.shape[1] == 64
        assert outputs.shape[2] == 4
        assert outputs.shape[3] == 4

    def test_MobileNet(self):
        model = MobileNet(10)
        inputs = torch.randn(4, 3, 224, 224)
        logits = model(inputs)
        assert len(logits.shape) == 2
        assert logits.shape[0] == 4
        assert logits.shape[1] == 10


class TestUNet(unittest.TestCase):
    def test_UNet_forward(self):
        model = UNet(input_shape=3, num_classes=5)
        batch_size = 2
        inputs = torch.randn(2, 3, 160, 160)
        outputs = model(inputs)
        self.assertEqual(outputs.shape, (2, 5, 160, 160))

    def test_DiceLoss(self):
        my_dice = DiceLoss()
        monai_dice = monai.losses.DiceLoss(smooth_nr=1, smooth_dr=1, sigmoid=True, batch=True)
        logits = torch.randn(2, 1, 32, 32)
        targets = torch.randint(0, 2, (2, 1, 32, 32)).float()
        my_loss = my_dice(logits, targets)
        monai_loss = monai_dice(logits, targets)
        self.assertTrue(torch.allclose(my_loss, monai_loss, atol=1e-5))


if __name__ == '__main__':
    unittest.main(verbosity=2)