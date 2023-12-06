import torch
import torch.nn as nn
torch.manual_seed(123)
conv1 = nn.Conv2d(2, 4, 1, bias=False)
class prototype(nn.Module):
    def __init__(self, n, c):
        super(prototype, self).__init__()
        self.b = nn.Parameter(torch.randn(n, c), requires_grad=True)
    def forward(self, x):
        x = torch.einsum('bchw, nc -> bnhw', x, self.b)
        return x

# b = nn.Parameter(torch.randn(2, 4), requires_grad=True)
bi_graph1 = nn.Parameter(torch.randn(4, 3), requires_grad=True)
bi_graph2 = nn.Parameter(torch.randn(4, 2), requires_grad=True)
b = prototype(4, 4)
# def hook_fn(module, grad_input, grad_output):
#     print(grad_input)
#     print(grad_output)
#     gin1 = grad_input[0]
#     gin1 *= 2
#     grad_input = (gin1, )
#     go1 = grad_output[0]
#     go1 *= 2
#     grad_out = (go1, )
#     print("module: ", module.b.grad)
#     module.b.grad = torch.ones_like(module.b.grad)
#     # print("hook_fn: ", grad)
#     return grad_input

# def hook_fn2(grad):
#     print("hook_fn2: ", grad)
#     # grad = torch.ones_like(grad)
#     return grad

# # b.register_full_backward_hook(hook_fn)
# b.b.register_hook(hook_fn2)
x = torch.randn(2, 2, 3, 3)
y = conv1(x)
# print(y.shape)
# print(b.shape)
# z = torch.einsum('bchw, nc -> bnhw', y, b)
z = b(y)
datasets_id = torch.tensor([0, 1])
z1 = torch.einsum('bchw, cn -> bnhw', z[datasets_id == 0], bi_graph1)
z2 = torch.einsum('bchw, cn -> bnhw', z[datasets_id == 1], bi_graph2)
criteria = nn.CrossEntropyLoss()
lb1 = torch.randint(0, 3, (1, 3, 3))
lb2 = torch.randint(0, 2, (1, 3, 3))
lb = torch.cat((lb1, lb2), dim=0)

print(lb)
loss = criteria(z1, lb[datasets_id == 0]) + criteria(z2, lb[datasets_id == 1]) 
loss.backward()

print("b.grad: ", b.b.grad)
print("conv.grad: ", conv1.weight.grad)


class MyConv1x1Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, datasets_id, M):
        # 保存反向传播所需的参数
        ctx.save_for_backward(x, weight, datasets_id, M)

        output = torch.einsum('bchw, nc -> bnhw', x, weight)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve the saved input and weight
        x, weight, datasets_id, M = ctx.saved_tensors
        
        for i in range(0, int(torch.max(datasets_id))+1):
            if not (datasets_id == i).any():
                continue
            grad_output[datasets_id == i] = torch.einsum('bchw, c -> bchw', grad_output[datasets_id == i], M[i])
        print(grad_output)
        grad_input = torch.einsum('bchw, cn -> bnhw', grad_output, weight)
        grad_weight = torch.einsum('bchw, bnhw -> cn', grad_output, x)

        # Reshape the gradients back to the original shape
        # grad_input = grad_input_reshaped.view(x.size())
        return grad_input, grad_weight, None, None

# 使用自定义的1x1卷积层
class MyConv1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MyConv1x1, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels))

    def forward(self, x):
        return MyConv1x1Function.apply(x, self.weight, datasets_id, torch.tensor([[1,1,1,0], [0,1,1,1]]))

conv1.zero_grad()
b.zero_grad()
x.grad = None
bi_graph1.grad = None
bi_graph2.grad = None
b2 = MyConv1x1(4, 4)
b2.weight.data = b.b
y = conv1(x)
# print(y.shape)
# print(b.shape)
# z = torch.einsum('bchw, nc -> bnhw', y, b)
z = b2(y)
z1 = torch.einsum('bchw, cn -> bnhw', z[datasets_id == 0], bi_graph1)
z2 = torch.einsum('bchw, cn -> bnhw', z[datasets_id == 1], bi_graph2)

loss = criteria(z1, lb[datasets_id == 0]) + criteria(z2, lb[datasets_id == 1]) 
# loss = criteria(z, lb)
loss.backward()

print("b.grad: ", b2.weight.grad)
print("conv.grad: ", conv1.weight.grad)
print('bigraph1.grad: ', bi_graph1.grad)


# a = torch.ones(1,2,1,2)
# b = torch.ones(3,2)
# b[0, 0] = 2
# print("einsum: ", torch.einsum('bchw, nc -> bnhw', a, b))

# x_reshaped = a.reshape(-1, a.size(1))
# output_reshaped = x_reshaped.matmul(b.t())
# print(output_reshaped)
# output = output_reshaped.reshape(a.size(0), -1, a.size(2), a.size(3))
# print("mat: ", output)