
from torch import nn
import torch 
class Block(nn.Module):
    def __init__(self, in_channels, output_channels, apply_activation=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,
                                       output_channels,
                                       kernel_size=3,
                                       padding=1)
                                       
        self.bn = nn.BatchNorm2d(output_channels)
        self.dim = output_channels
        if apply_activation:
            self.activation = nn.LeakyReLU()
        else:
            self.activation = None 
        
    def forward(self, inputs):
        x = self.bn(self.conv(inputs))
        if self.activation is not None:
            x = self.activation(x)

        return x

class UpBlock(nn.Module):
    def __init__(self, in_channels, output_channels, apply_activation=True):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels,
                                       output_channels,
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1)
                                       
        self.bn = nn.BatchNorm2d(output_channels)
        self.dim = output_channels
    
        if apply_activation:
            self.activation = nn.LeakyReLU()
        else:
            self.activation = None 
        
    def forward(self, inputs):
        x = self.bn(self.conv(inputs))
        if self.activation is not None:
            x = self.activation(x)
        return x

class SubModule(nn.Module):
    def __init__(self, in_channels, block_props):
        super().__init__()
        module_list = []
        for i, block_prop in enumerate(block_props):
            if block_prop["upsample"]:
                block = UpBlock(in_channels, block_prop["dim"], apply_activation=block_prop["apply_activation"])
            else:
                block = Block(in_channels, block_prop["dim"], apply_activation=block_prop["apply_activation"])
            module_list.append(block)
            in_channels = block_prop["dim"]
        self.module_list = nn.ModuleList(module_list)
    def forward(self, inputs):
        x = inputs
        for i in range(len(self.module_list)):
            x = self.module_list[i](x)
        return x
        

class TerminalBlock(nn.Module):
    def __init__(self, in_channels, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1)
                                       
        self.bn = nn.BatchNorm2d(dim)
    
        self.activation = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(dim, 3, kernel_size=3, padding=1)
        self.output_activation = nn.Tanh()
        # self.output_activation = nn.Sigmoid()


    def forward(self, inputs):
        x = self.activation(self.bn(self.conv1(inputs)))
        x = self.output_activation(self.conv2(x))
        # x =  self.conv2(x)
        return x
class SubModuleWithTerminal(nn.Module):
    def __init__(self,  in_channels, block_props, terminal_dim):
        super().__init__()
        self.sm = SubModule(in_channels,block_props)
        self.terminal = TerminalBlock(self.sm.module_list[-1].dim, terminal_dim)
    def forward(self, inputs, return_terminal=False):
        x = self.sm(inputs)
        if return_terminal:
            return self.terminal(x)
        return x

class StudentModel(nn.Module):
    def __init__(self):
        super().__init__()

        bp1 = [
            {"upsample": True, "dim":16, "apply_activation": True}
        ]
        bp2 = [
            {"upsample": False, "dim":16, "apply_activation": True}
        ]
        bp3 = [
            {"upsample": False, "dim":32, "apply_activation": True},
            {"upsample": True, "dim":32, "apply_activation": True},
        ]
        bp4 = [
            {"upsample": False, "dim":32, "apply_activation": True},
            {"upsample": False, "dim":32, "apply_activation": True},
        ]
        
        bp5 = [
            {"upsample": False, "dim":64, "apply_activation": True},
            {"upsample": True, "dim":64, "apply_activation": True},
        ]
        bp6 = [
            {"upsample": False, "dim":64, "apply_activation": True},
            {"upsample": False, "dim":64, "apply_activation": True},
        ]

        bp7 = [
            {"upsample": False, "dim":128, "apply_activation": True},
            {"upsample": True, "dim":128, "apply_activation": True},
        ]
        bp8 = [
            {"upsample": False, "dim":128, "apply_activation": True},
            {"upsample": False, "dim":128, "apply_activation": True},
        ]

        bp9 = [
            {"upsample": False, "dim":128, "apply_activation": True},
            {"upsample": True, "dim":128, "apply_activation": True},
        ]
        bp10 = [
            {"upsample": False, "dim":128, "apply_activation": True},
            {"upsample": False, "dim":128, "apply_activation": True},
        ] 
        
        bp11 = [
            {"upsample": False, "dim":128, "apply_activation": True},
            {"upsample": True, "dim":128, "apply_activation": True},
        ]


        all_block_props = [bp1, bp2, bp3, bp4, bp5, bp6, bp7, bp8, bp9, bp10,bp11]
        blocks = []
        prev_channels = 3
        for block_props in all_block_props:
            blocks.append(SubModuleWithTerminal(prev_channels, block_props, block_props[-1]["dim"]))
            prev_channels = block_props[-1]["dim"]
        self.blocks = nn.ModuleList(blocks)
    def forward(self, inputs, step=10):
        x = inputs
        for index, block in enumerate(self.blocks):
            if index == step:
                x = block(x, return_terminal=True)
                return x
            else:
                x = block(x, return_terminal=False)


if __name__ == "__main__":
    st = StudentModel()
    inputs = torch.randn(1, 3, 8, 8)
    outputs = st(inputs, 9)
    print(outputs.shape)