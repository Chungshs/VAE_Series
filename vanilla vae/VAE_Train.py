import argparse
import os
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm

from vanilla_VAE import Vanilla_VAE

""" The code was adapted from:
https://github.com/pytorch/examples/blob/master/vae/main.py
https://github.com/orybkin/sigma-vae-pytorch (training method is reffered)
 """

## Arguments
parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--opt', type=str, default='mse', metavar='N',
                    help='which model to use: mse_vae,  gaussian_vae, or sigma_vae or optimal_sigma_vae')
parser.add_argument('--log_dir', type=str, default='test', metavar='N', required=True)
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

transfrom = transforms.Compose( [transforms.Resize((28, 28)), transforms.ToTensor()])
train_dataset = datasets.SVHN('../../data', split='train', download=True, transform=transfrom)
test_dataset = datasets.SVHN('../../data', split='train', transform=transfrom)
kwargs = {'num_workers': 10, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)


os.makedirs('vae_logs/{}'.format(args.log_dir), exist_ok=True)
summary_writer = SummaryWriter(log_dir='vae_logs/' + args.log_dir, purge_step=0)

model = Plane_VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def total_loss(opt, recon_x, x, mu, logvar):
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
	return reconstruction_loss(opt, recon_x, x), kld_loss(mu, logvar)

def reconstruction_loss(opt, recon_x, x):
	if opt == 'mse':
		return F.mse_loss(recon_x, x, reduction='sum')
	elif opt == 'BE':
		return F.binary_cross_entropy(recon_x, x, reduction='sum')

def kld_loss(mu, logvar):
	return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()

        mu, logvar, z,  recon_x = model(data)
        rec, kl = total_loss(args.opt, recon_x, data, mu, logvar)
        total_loss_norm = rec + kl
        total_loss_norm.backward()
        train_loss += total_loss_norm.item()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tMse: {:.6f}\tKL: {:.6f}\t'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                rec.item() / len(data),
                kl.item() / len(data)))
    train_loss /= len(train_loader.dataset)
    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss))
    summary_writer.add_scalar('train/elbo', train_loss, epoch)
    summary_writer.add_scalar('train/rec', rec.item() / len(data), epoch)
    summary_writer.add_scalar('train/kld', kl.item() / len(data), epoch)

def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(tqdm(test_loader)):
            data = data.to(device)
            mu, logvar, z,  recon_x = model(data)
            # Pass the second value from posthoc VAE
            rec, kl = total_loss(args.opt, recon_x, data, mu, logvar)
            test_loss += rec + kl
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n], recon_x.view(args.batch_size, -1, 28, 28)[:n]])
                save_image(comparison.cpu(), 'vae_logs/{}/reconstruction_{}.png'.format(args.log_dir, str(epoch)), nrow=n)
                
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    summary_writer.add_scalar('test/elbo', test_loss, epoch)

if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            sample = model.sample(64).cpu()
            save_image(sample.view(64, -1, 28, 28),
                       'vae_logs/{}/sample_{}.png'.format(args.log_dir, str(epoch)))
        summary_writer.file_writer.flush()
        
    torch.save(model.state_dict(), 'vae_logs/{}/checkpoint_{}.pt'.format(args.log_dir, str(epoch)))
