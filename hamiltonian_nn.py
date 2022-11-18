import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import ReLU
from torch.nn.parameter import Parameter
import torchvision
import tqdm
from torchvision import transforms

import math
n_inner_layers = 3
PATH = f'./cifar_net_{n_inner_layers}.pth'
h = .5
batch_size = 128
reg2_factor = 1e-5 / math.sqrt(batch_size)
class HamiltonianIdentityBlock(nn.Module):
    def __init__(self,  n_channels, kernel_size):
        super(HamiltonianIdentityBlock, self).__init__()
        assert n_channels % 2 == 0
        self.n_y_channels = n_channels // 2
        self.padding = kernel_size // 2
        self.K1 = Parameter(torch.rand((self.n_y_channels, self.n_y_channels, kernel_size, kernel_size)))
        self.bias1 = Parameter(torch.zeros((self.n_y_channels,)))
        torch.nn.init.kaiming_normal_(self.K1)
        self.K2 =  Parameter(torch.rand((self.n_y_channels, self.n_y_channels, kernel_size, kernel_size)))
        self.bias2 = Parameter(torch.zeros((self.n_y_channels,)))
        torch.nn.init.kaiming_normal_(self.K2)

        self.act = ReLU()

    def forward(self, x):
        # x: [BS, C, H, W]
        y = x[:, self.n_y_channels:, :, :]
        z = x[:, :self.n_y_channels, :, :]
        for i in range(int(1 // h)):
            K1z = F.conv2d(z, self.K1, bias=self.bias1, padding=self.padding)
            K1zT = torch.transpose(self.K1, -2, -1)
            y_2 = y + F.conv2d(self.act(K1z), K1zT, padding=self.padding) / h
    
            K2y = F.conv2d(y, self.K2, bias=self.bias2, padding=self.padding)
            K2yT = torch.transpose(self.K2, -2, -1)
            z_2 = z - F.conv2d(self.act(K2y), K2yT, padding=self.padding) / h
            y, z = y_2, z_2

        return torch.cat([y_2, z_2], dim=1)

class HamiltonianStack(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers):
        super(HamiltonianStack, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(HamiltonianIdentityBlock(
                in_channels, kernel_size=3))
        self.avg_pool = nn.AvgPool2d(2, 2)
        self.out_channels = out_channels

    def forward(self, x):
        self.act_squares = torch.sum((reg2_factor * x)**2)
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            self.act_squares += torch.sum((reg2_factor * x)**2)
        x = self.avg_pool(x)
        zero_channel_pad = torch.zeros(
            (x.shape[0],
            self.out_channels - x.shape[1],
            x.shape[2],
            x.shape[3])).to(x.device)
        x = torch.cat([x, zero_channel_pad], dim=1)
        return x
    
    def reg(self):
        reg = 0
        for i in range(len(self.layers) - 1):
            reg += torch.sum((self.layers[i + 1].K1 - self.layers[i].K1)**2)
            reg += torch.sum((self.layers[i + 1].K2 - self.layers[i].K2)**2)
        return reg * h, self.act_squares

class Hamiltonian(nn.Module):
    def __init__(self, n_inner_layers=2, n_input_channels=3, n_classes=10):
        super(Hamiltonian, self).__init__()
        layer_sizes = [64, 128, 128]
        self.conv1 = nn.Conv2d(
            in_channels=n_input_channels, out_channels=layer_sizes[0],
            kernel_size=3, padding=1)
        self.hams = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.hams.append(HamiltonianStack(layer_sizes[i], layer_sizes[i + 1], n_layers=n_inner_layers))
        self.fc = nn.Linear(layer_sizes[-1], n_classes)

    def forward(self, x):
        x = self.conv1(x)
        for layer in self.hams:
            x = layer(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1,1 )).flatten(1)
        x = self.fc(x)
        return x

    def reg(self):
        reg_time = 0
        reg_act = 0
        for ham in self.hams:
            time_smooth, activation_size = ham.reg()
            reg_time += time_smooth
            reg_act += activation_size
        return reg_time, reg_act

transform = transforms.Compose(
    [transforms.ToTensor()])
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def train_hamiltonian_nn(device):
    gc.collect()
    torch.cuda.empty_cache()
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)
    torch.random.manual_seed(0)
    net = Hamiltonian(n_inner_layers=n_inner_layers)
    try:
        net.load_state_dict(torch.load(PATH))
        print("Loaded existing network")
    except FileNotFoundError:
        print("Loading new network")
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=.96)
    n_epochs = 30
    pbar = tqdm.tqdm(range(n_epochs))
    pbar_dict = {}
    reg_factor = 5e-5
    for epoch in pbar:
        running_loss = 0.0
        running_w_reg = 0.0
        running_act_reg = 0.0
        total = 0
        correct = 0
        for i, data in enumerate(trainloader, 1):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            reg_w, reg_act = net.reg()
            reg_w = reg_w * reg_factor
            reg_act = reg_act * reg2_factor 
            loss = criterion(outputs, labels)
            (loss + reg_w + reg_act).backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            running_w_reg += reg_w.item()
            running_act_reg += reg_act.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            pbar_dict['loss'] = running_loss / i
            pbar_dict['acc'] = correct / total
            pbar_dict['reg'] = running_w_reg / i
            pbar_dict['reg2'] = running_act_reg / i
            pbar_dict['seen'] = total
            pbar.set_postfix(pbar_dict)
        torch.save(net.state_dict(), PATH)
        test_acc = test_hamiltonian_nn(device, net, testloader)

def test_hamiltonian_nn(device, net, testloader):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    print(f'Accuracy: {correct}/{total}')
    return acc


def robust_test_hamiltonian_nn():
    threat_model = "Linf"  # one of {"Linf", "L2", "corruptions"}
    dataset = "cifar10"  # one of {"cifar10", "cifar100", "imagenet"}

    model = Hamiltonian(n_inner_layers=n_inner_layers).eval()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(PATH, map_location=device))
    model_name = "Hamiltonian2022Reversible"

    clean_acc, robust_acc = benchmark(model, model_name=model_name, n_examples=10000, dataset=dataset,
                                    threat_model=threat_model, eps=8/255, device=device,
                                    to_disk=True)
    print(f"Clean acc {clean_acc} robust acc {robust_acc}")

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    from robustbench import benchmark
    robust_test_hamiltonian_nn()
