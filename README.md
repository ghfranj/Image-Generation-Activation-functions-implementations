# Image-Generation-Activation-functions-implementations
This repository shows different activation functions and their implementations, and compares them. It also shows an additional activation function that can perform better

Steps:
1- Implementing a linear or convolutional neural network of two hidden layers.
2- Training it on the MNIST dataset.
3- Comparing the quality of training when using different activation functions (ReLU, GELU, Swish, Softplus etc...), where these activation functions are manually implemented!
4- Plotting the accuracy activation functions against epochs.
5- Coming up with a more effective activation function in this case.

1- Dataset used: MNIST dataset
~~~
!kaggle datasets download -d hojjatk/mnist-dataset
~~~

2- Model Used:
~~~

class Net(nn.Module):
    def __init__(self, num_classes, activation_fn = nn.ReLU):
        self.activation_fn = activation_fn
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=5),
            nn.Dropout(0.3),
            activation_fn(),
        ).to(device)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1728, num_classes)
        ).to(device)

    def forward(self, x):
        x = self.model(x)
        x = self.classifier(x)
        return x
~~~
3- Activation functions used:
Relu:
~~~
class ReLU_imp(torch.nn.Module):
    def __init__(self):
        super(ReLU_imp, self).__init__()
        return
    def forward(self, x):
        return torch.max(tensor(0), x)
~~~
Tanh:
~~~
class Tanh_imp(torch.nn.Module):
  def __init__(self):
      super(Tanh_imp, self).__init__()
      return
  def forward(self, x):
      return torch.tanh(x)
~~~
GELU:
~~~
class GELU_imp(torch.nn.Module):
  def __init__(self):
      super(GELU_imp, self).__init__()
      return
  def forward(self, x):
      return 0.5 * x * ( 1 + torch.tanh( (2/torch.pi)**0.5 * (x + 0.044715 * x**3 ) ))
~~~
Swish:
~~~
class Swish_imp(torch.nn.Module):
  def __init__(self):
      super(Swish_imp, self).__init__()
      return
  def forward(self, x):
      return x * torch.sin(x)
~~~
SoftPlus:
~~~
class SoftPlus_imp(torch.nn.Module):
  def __init__(self):
      super(SoftPlus_imp, self).__init__()
      return
  def forward(self, x):
      return torch.log(1 + torch.exp(x))
~~~

##Results obtained:
![image](https://github.com/ghfranj/Image-Generation-Activation-functions-implementations/assets/98123238/a674088e-2277-4b43-88f2-3970310a1583)

###The additional activation function:
~~~
class GELU_Tanh_edit_imp(torch.nn.Module):
  def __init__(self):
      super(GELU_Tanh_edit_imp, self).__init__()
      return
  def forward(self, x):
      return (torch.tanh(torch.max(tensor(2),x)))*x**2*0.4
~~~
