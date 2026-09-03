# Vendored from the official implementation. Only imports were rewritten so the module
# resolves inside nnunetv2.unified.nets; the network definitions themselves are untouched
# except where explicitly marked with '# [unified]'.
import torch.nn as nn


class IntermediateSequential(nn.Sequential):
    def __init__(self, *args, return_intermediate=True):
        super().__init__(*args)
        self.return_intermediate = return_intermediate

    def forward(self, input):
        if not self.return_intermediate:
            return super().forward(input)

        intermediate_outputs = {}
        output = input
        for name, module in self.named_children():
            output = intermediate_outputs[name] = module(output)

        return output, intermediate_outputs
        
