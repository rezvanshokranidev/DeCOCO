# dic1={'a':['f','m'],'b':123}
# dic2={'a':['female','male'],'b':456}

# MATCH = {key:[dic1[key], dic2[key]] for key in dic1 if key in dic2}

# print(MATCH)



import torch
x = torch.tensor([1, 2, 3, 4])
print(x.shape)
y=torch.unsqueeze(x, 0)
print(y.shape)
# import torch.nn as nn

# input = torch.randn(20, 5, 10, 10)
# # With Learnable Parameters
# m = nn.LayerNorm(input.size()[1:])
# # print(m)
# m = nn.BatchNorm1d(input.size()[1:])
# # print(m)
# # Without Learnable Parameters
# m = nn.LayerNorm(input.size()[1:], elementwise_affine=False)
# # Normalize over last two dimensions
# m = nn.LayerNorm([10, 10])
# # Normalize over last dimension of size 10
# m = nn.LayerNorm(10)
# # Activating the module
# output = m(input)
# print(output)