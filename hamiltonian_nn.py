# https://arxiv.org/pdf/1705.03341.pdf
# https://ojs.aaai.org/index.php/AAAI/article/view/11668
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import ReLU
import torchvision
import tqdm
from torchvision import transforms

PATH = './cifar_net.pth'

class HamiltonianIdentityBlock(nn.Module):
    def __init__(self,  n_channels, kernel_size):
        super(HamiltonianIdentityBlock, self).__init__()
        assert n_channels % 2 == 0
        self.n_y_channels = n_channels // 2
        shared_w_k1 =  torch.rand((self.n_y_channels, self.n_y_channels, kernel_size, kernel_size))
        torch.nn.init.kaiming_normal_(shared_w_k1)
        shared_w_k2 =  torch.rand((self.n_y_channels, self.n_y_channels, kernel_size, kernel_size))
        torch.nn.init.kaiming_normal_(shared_w_k2)
        
        self.K1 = nn.Conv2d(in_channels=self.n_y_channels,
                               out_channels=self.n_y_channels,
                               kernel_size=kernel_size,
                               padding=kernel_size//2)
        self.K2 = nn.Conv2d(in_channels=self.n_y_channels,
                               out_channels=self.n_y_channels,
                               kernel_size=kernel_size,
                               padding=kernel_size//2)
        self.K1.weight = nn.Parameter(shared_w_k1)
        self.K2.weight = nn.Parameter(shared_w_k2)
        self.K1T = nn.Conv2d(in_channels=self.n_y_channels,
                               out_channels=self.n_y_channels,
                               kernel_size=kernel_size, bias=False,
                               padding=kernel_size//2)
        self.K2T = nn.Conv2d(in_channels=self.n_y_channels,
                               out_channels=self.n_y_channels,
                               kernel_size=kernel_size, bias=False,
                               padding=kernel_size//2)
        self.K1T.weight = nn.Parameter(torch.transpose(shared_w_k1, -2, -1))
        self.K2T.weight = nn.Parameter(torch.transpose(shared_w_k2, -2, -1))

        self.act = ReLU()

    def forward(self, x):
        # x: [BS, C, H, W]
        y = x[:, self.n_y_channels:, :, :]
        z = x[:, :self.n_y_channels, :, :]
        y_2 = y + self.K1T(self.act(self.K1(z)))
        z_2 = z - self.K2T(self.act(self.K2(y)))
        assert not torch.any(torch.isnan(self.K1.weight))
        assert self.K1.weight[3][2][1][0] == self.K1T.weight[3][2][0][1]
        return torch.cat([y_2, z_2], dim=1)
    

class HamiltonianStack(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers):
        super(HamiltonianStack, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(HamiltonianIdentityBlock(
                in_channels, kernel_size=3))
        self.bn = nn.BatchNorm2d(in_channels)
        self.avg_pool = nn.AvgPool2d(2, 2)
        self.out_channels = out_channels

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.bn(x)
        x = self.avg_pool(x)
        zero_channel_pad = torch.zeros(
            (x.shape[0],
            self.out_channels - x.shape[1],
            x.shape[2],
            x.shape[3]))
        x = torch.cat([x, zero_channel_pad], dim=1)
        return x

class Hamiltonian(nn.Module):
    def __init__(self, n_input_channels=3, n_classes=10):
        super(Hamiltonian, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=n_input_channels, out_channels=64,
            kernel_size=3, padding=1)
        self.ham1 = HamiltonianStack(64, 128, n_layers=4)
        self.ham2 = HamiltonianStack(128, 256, n_layers=4)
        self.ham3 = HamiltonianStack(256, 256, n_layers=4)
        self.fc = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.ham1(x)
        x = self.ham2(x)
        x = self.ham3(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1,1 )).flatten(1)
        x = self.fc(x)
        return x

def train_hamiltonian_nn(device):
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    batch_size = 64
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    torch.random.manual_seed(0)
    net = Hamiltonian()
    try:
        net.load_state_dict(torch.load(PATH))
    except FileNotFoundError:
        print("Loading new network")
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=2e-4)
    print_every = 500
    n_epochs = 10
    pbar = tqdm.tqdm(range(n_epochs))
    pbar_dict = {}
    for epoch in pbar:
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if (i + 1) % print_every == 0:    # print every 2000 mini-batches
                pbar_dict['loss'] = running_loss / print_every
                pbar_dict['i'] = i
                running_loss = 0.0
                torch.save(net.state_dict(), PATH)
                pbar.set_postfix(pbar_dict)
        test_acc = test_hamiltonian_nn(device)
        pbar_dict['test_acc'] = test_acc

def test_hamiltonian_nn(device):
    net = Hamiltonian()
    net.load_state_dict(torch.load(PATH))
    net.to(device)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)
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
    print(f'Accuracy of the network on the 10000 test images: {acc} %')
    return acc




def robest_test_hamiltonian_nn():
    from robustbench import benchmark

    threat_model = "Linf"  # one of {"Linf", "L2", "corruptions"}
    dataset = "cifar10"  # one of {"cifar10", "cifar100", "imagenet"}

    model = Hamiltonian()
    model_name = "Hamiltonian2022Reversible"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    clean_acc, robust_acc = benchmark(model, model_name=model_name, n_examples=10000, dataset=dataset,
                                    threat_model=threat_model, eps=8/255, device=device,
                                    to_disk=True)
    print(f"Clean acc {clean_acc} robust acc {robust_acc}")

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_hamiltonian_nn(device)