from __future__ import print_function, division
import scanpy as sc
import pandas as pd
# from dca.api import dca
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import sys
import math
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.rc('pdf', fonttype=42)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'

sc.settings.set_figure_params(dpi=200)
plt.rcParams['axes.grid'] = False
plt.rcParams['figure.figsize'] = (7,7)

def parse_args(gene_num, ncls, jb_name):
    global opt, max_ncls
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=1, help='number of epochs of training')
    parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
    parser.add_argument('--kt', type=float, default=0, help='kt parameters')
    parser.add_argument('--lambda_k', type=float, default=0.001, help='lambda_k')
    parser.add_argument('--gamma', type=float, default=0.95, help='gamma parameters')
    parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
    parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
    parser.add_argument('--img_size', type=int, default=100, help='size of each image dimension')
    parser.add_argument('--channels', type=int, default=1, help='number of image channels')
    parser.add_argument('--n_critic', type=int, default=1, help='number of training steps for discriminator per iter')
    parser.add_argument('--sample_interval', type=int, default=400, help='interval between image sampling')
    parser.add_argument('--dpt', type=str, default='', help='load discrimnator model')
    parser.add_argument('--gpt', type=str, default='', help='load generator model')
    parser.add_argument('--train', type=bool, default=False, help='train the network')
    parser.add_argument('--impute', type=bool, default=False, help='do imputation')
    parser.add_argument('--sim_size', type=int, default=200, help='number of sim_imgs in each type')
    parser.add_argument('--file_d', type=str, default='', help='path of data file')
    parser.add_argument('--file_c', type=str, default='', help='path of cls file')
    parser.add_argument('--ncls', type=int, default=4, help='number of clusters')
    parser.add_argument('--knn_k', type=int, default=10, help='neighbours used')
    parser.add_argument('--lr_rate', type=int, default=10, help='rate for slow learning')
    parser.add_argument('--threthold', type=float, default=0.01, help='the convergence threthold')
    parser.add_argument('--job_name', type=str, default="",
                        help='the user-defined job name, which will be used to name the output files.')
    parser.add_argument('--outdir', type=str, default="for_dca", help='the directory for output.')
    parser.add_argument('--label_index', type=int, default=0, help='for label column indexing')
    parser.add_argument('--skip_label_first', type=bool, default=False, help='skip label file s first row')
    parser.add_argument('--data_from_csv', type=bool, default=False, help='if the data read from csv file')

    opt = parser.parse_args()
    opt.img_size = math.floor(math.sqrt(gene_num))
    opt.ncls = ncls
    max_ncls = ncls

    opt.job_name = jb_name


def init_job():
    global model_basename, cuda, GANs_models
    opt.train = True

    job_name = opt.job_name
    GANs_models = opt.outdir + '/GANs_models'
    if job_name == "":
        job_name = os.path.basename(opt.file_d) + "-" + os.path.basename(opt.file_c)
    model_basename = job_name + "-" + str(opt.latent_dim) + "-" + str(opt.n_epochs) + "-" + str(opt.ncls) + "-" + str(
        opt.b1 * 10) + "-" + str(opt.lr * 10000)
    if not os.path.isdir(GANs_models):
        os.makedirs(GANs_models)

    # img_shape = (opt.channels, opt.img_size, opt.img_size)
    cuda = True if torch.cuda.is_available() else False


class MyDataset(Dataset):
    # load src data & src labels
    def __init__(self, matrix, label, transform=None):
        self.data = matrix
        self.data_cls = label
        self.transform = transform
        self.fig_h = opt.img_size
        # print('[init] data len:', len(self.data), ' label len:', len(self.data_cls))
        # print(self.data_cls, ' len:', len(self.data_cls))
        # print(self.data)

    def __len__(self):
        return len(self.data_cls)

    def __getitem__(self, idx):
        # use astype('double/float') to sovle the runtime error caused by data mismatch.
        # iloc[行, 列]. values:
        data = self.data.iloc[:, idx].values[0:(self.fig_h * self.fig_h), ].reshape(self.fig_h, self.fig_h, 1).astype(
            'double')
        label = np.array(self.data_cls[idx]).astype('int32')
        sample = {'data': data, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        data, label = sample['data'], sample['label']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        data = data.transpose((2, 0, 1))

        return {'data': torch.from_numpy(data),
                'label': torch.from_numpy(label)
                }


def one_hot(batch, depth):
    ones = torch.eye(depth)
    return ones.index_select(0, batch)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4
        self.cn1 = 32
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, self.cn1 * (self.init_size ** 2)))
        self.l1p = nn.Sequential(nn.Linear(opt.latent_dim, self.cn1 * (opt.img_size ** 2)))

        self.conv_blocks_01p = nn.Sequential(
            nn.BatchNorm2d(self.cn1),
            #           nn.Upsample(scale_factor=2),
            nn.Conv2d(self.cn1, self.cn1, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.cn1, 0.8),
            nn.ReLU(),
        )

        self.conv_blocks_02p = nn.Sequential(
            #            nn.BatchNorm2d(9),
            nn.Upsample(scale_factor=opt.img_size),  # torch.Size([bs, 128, 16, 16])
            nn.Conv2d(max_ncls, self.cn1 // 4, 3, stride=1, padding=1),  # torch.Size([bs, 128, 16, 16])
            nn.BatchNorm2d(self.cn1 // 4),
            nn.ReLU(),
        )

        self.conv_blocks_1 = nn.Sequential(
            nn.BatchNorm2d(40, 0.8),
            nn.Conv2d(40, self.cn1, 3, stride=1, padding=1),  # torch.Size([bs, 1, 32, 32])
            nn.BatchNorm2d(self.cn1),
            nn.ReLU(),
            nn.Conv2d(self.cn1, opt.channels, 3, stride=1, padding=1),  # torch.Size([bs, 1, 32, 32])
            nn.Sigmoid()
        )

    def forward(self, noise, label_oh):
        out = self.l1p(noise)
        out = out.view(out.shape[0], self.cn1, opt.img_size, opt.img_size)
        out01 = self.conv_blocks_01p(out)  # ([4, 32, 124, 124])
        #
        label_oh = label_oh.unsqueeze(2)
        label_oh = label_oh.unsqueeze(2)
        out02 = self.conv_blocks_02p(label_oh)  # ([4, 8, 124, 124])

        out1 = torch.cat((out01, out02), 1)
        out1 = self.conv_blocks_1(out1)
        return out1


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.cn1 = 32
        self.down_size0 = 64
        self.down_size = 32
        # pre
        self.pre = nn.Sequential(
            nn.Linear(opt.img_size ** 2, self.down_size0 ** 2),
        )

        # Upsampling
        self.down = nn.Sequential(
            nn.Conv2d(opt.channels, self.cn1, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(self.cn1),
            nn.ReLU(),
            nn.Conv2d(self.cn1, self.cn1 // 2, 3, 1, 1),
            #            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(self.cn1 // 2),
            nn.ReLU(),
        )

        self.conv_blocks02p = nn.Sequential(
            #            nn.BatchNorm2d(9),
            nn.Upsample(scale_factor=self.down_size),  # torch.Size([bs, 128, 16, 16])
            nn.Conv2d(max_ncls, self.cn1 // 4, 3, stride=1, padding=1),  # torch.Size([bs, 128, 16, 16])
            nn.BatchNorm2d(self.cn1 // 4),
            nn.ReLU(),
        )

        # Fully-connected layers
        down_dim = 24 * self.down_size ** 2
        self.fc = nn.Sequential(
            nn.Linear(down_dim, 16),
            nn.BatchNorm1d(16, 0.8),
            nn.ReLU(),
            nn.Linear(16, down_dim),
            nn.BatchNorm1d(down_dim),
            nn.ReLU()
        )

        # Upsampling 32X32
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=4),
            nn.Conv2d(24, 16, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 4, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 4, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 4, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 4, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, opt.channels, 2, 1, 0),
            nn.Sigmoid(),
        )

    def forward(self, img, label_oh):
        out00 = self.pre(img.view((img.size()[0], -1))).view((img.size()[0], 1, self.down_size0, self.down_size0))
        out01 = self.down(out00)  # ([4, 16, 32, 32])

        label_oh = label_oh.unsqueeze(2)
        label_oh = label_oh.unsqueeze(2)
        out02 = self.conv_blocks02p(label_oh)  # ([4, 16, 32, 32])

        out1 = torch.cat((out01, out02), 1)

        out = self.fc(out1.view(out1.size(0), -1))
        out = self.up(out.view(out.size(0), 24, self.down_size, self.down_size))
        return out


def my_knn_type(data_imp_org_k, sim_out_k, knn_k=10):
    sim_size = sim_out_k.shape[0]
    out = data_imp_org_k.copy()
    q1k = data_imp_org_k.reshape((opt.img_size * opt.img_size, 1))
    q1kl = np.int8(q1k > 0)  # get which part in cell k is >0
    q1kn = np.repeat(q1k * q1kl, repeats=sim_size, axis=1)  # get >0 part of cell k
    sim_out_tmp = sim_out_k.reshape((sim_size, opt.img_size * opt.img_size)).T
    sim_outn = sim_out_tmp * np.repeat(q1kl, repeats=sim_size, axis=1)  # get the >0 part of simmed ones
    diff = q1kn - sim_outn  # distance of cell k to simmed ones
    diff = diff * diff
    rel = np.sum(diff, axis=0)
    locs = np.where(q1kl == 0)[0]
    #        locs1 = np.where(q1kl==1)[0]
    sim_out_c = np.median(sim_out_tmp[:, rel.argsort()[0:knn_k]], axis=1)
    out[locs] = sim_out_c[locs]
    return out


def init_generator(matrix, label):
    global dataloader, optimizer_G, optimizer_D, Tensor, discriminator, generator
    # Initialize generator and discriminator
    generator = Generator()
    discriminator = Discriminator()

    if cuda:
        generator.cuda()
        discriminator.cuda()
        print("scIGANs is running on GPUs. \n")
    else:
        print("scIGANs is running on CPUs. \n")

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # Configure data loader
    transformed_dataset = MyDataset(matrix, label,
                                    transform=transforms.Compose([
                                        #                                               Rescale(256),
                                        #                                               RandomCrop(224),
                                        ToTensor()
                                    ]))
    # print('[data] \n  len:', len(transformed_dataset.data), '\n  data:', transformed_dataset.data)
    # print('[label] \n  len:', len(transformed_dataset.data_cls), '\n  data:', transformed_dataset.data_cls)

    dataloader = DataLoader(transformed_dataset, batch_size=opt.batch_size,
                            shuffle=True, num_workers=0, drop_last=False)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


# 通过模型进行插补
def do_compute(matrix, label):
    print('[Info] execute imputation')
    file_name = opt.outdir + '/scIGANs-' + model_basename + '.csv'
    file_exists = os.path.isfile(file_name)
    if file_exists:
        overwrite = input(
            "WARNING: An imputed file exists with the same settings for your data.\n   Do you want to impute and "
            "overwrite it?: (y/n)\n")
        if overwrite != "y":
            print("The training was deprecated since optical model exists.")
            print("scIGANs reserve the exists imputed file")
            imputed_data = pd.read_csv(file_name, header=0, index_col=0)
            my_data = pd.DataFrame(imputed_data)
            return np.array(my_data.values)

    if opt.gpt == '':
        model_g = GANs_models + '/' + model_basename + '-g.pt'
        model_exists = os.path.isfile(model_g)
        if not model_exists:
            print("ERROR: There is no model exists with the given settings for your data.")
            print("Please set --train instead of --impute to train a model fisrt.")
            sys.exit("scIGANs stopped!!!")  # if model exists and do not want to train again, exit the program
    else:
        model_g = opt.gpt
    if cuda:
        generator.load_state_dict(torch.load(model_g))
    else:
        generator.load_state_dict(torch.load(model_g, map_location=lambda storage, loc: storage))
    ######################################################
    ### impute by type
    ######################################################
    # number of images by each type, default = 200
    sim_size = opt.sim_size
    sim_out = list()
    for i in range(opt.ncls):
        label_oh = one_hot(torch.from_numpy(np.repeat(i, sim_size)).type(torch.LongTensor), max_ncls).type(Tensor)

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (sim_size, opt.latent_dim))))

        # Generate a batch of images, number = 20
        fake_imgs = generator(z, label_oh).detach().data.cpu().numpy()
        # images: 200 个 9x9 image
        # if i == 0: print('[fake image] len:', len(fake_imgs), ' count:', len(fake_imgs[0][0]), ' column:',
        # len(fake_imgs[0][0][0]), ' content:', fake_imgs)
        sim_out.append(fake_imgs)
    mydataset = MyDataset(matrix, label)
    # print('DataSet len:', len(mydataset), ' SimOut len:', len(sim_out))
    data_imp_org = np.asarray(
        [mydataset[idx]['data'].reshape((opt.img_size * opt.img_size)) for idx in range(len(mydataset))]).T
    # data_imp = data_imp_org.copy()

    # by type
    sim_out_org = sim_out
    rels = [my_knn_type(data_imp_org[:, idx], sim_out_org[int(mydataset[idx]['label']) - 1], knn_k=opt.knn_k) for idx in
            range(len(mydataset))]
    pd.DataFrame(rels).to_csv(opt.outdir + '/scIGANs-' + model_basename + '.csv')        # save imputed data
    return rels

# ----------
#  Training
# ----------

def start_train():
    model_exists = os.path.isfile(GANs_models + '/' + model_basename + '-g.pt')
    if model_exists:
        overwrite = input(
            "WARNING: A trained model exists with the same settings for your data.\n   Do you want to train and "
            "overwrite it?: (y/n)\n")
        if overwrite != "y":
            print("The training was deprecated since optical model exists.")
            print("scIGANs continues imputation using existing model...")
            return

    # hyper parameters, opt.kt, opt.lambda_k, opt.gama
    k = opt.kt

    max_M = sys.float_info.max
    min_dM = 0.001
    dM = 1
    for epoch in range(opt.n_epochs):
        cur_M = 0
        cur_dM = 1
        for i, batch_sample in enumerate(dataloader):
            # if i == 0:
            #    continue
            # print('[info] i = ', i)
            imgs = batch_sample['data'].type(Tensor)
            label = batch_sample['label']
            label_oh = one_hot((label).type(torch.LongTensor), max_ncls).type(Tensor)

            # Configure input
            real_imgs = Variable(imgs.type(Tensor))

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

            # Generate a batch of images
            gen_imgs = generator(z, label_oh)

            # Loss measures generator's ability to fool the discriminator
            g_loss = torch.mean(torch.abs(discriminator(gen_imgs, label_oh) - gen_imgs))

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            d_real = discriminator(real_imgs, label_oh)
            d_fake = discriminator(gen_imgs.detach(), label_oh)

            d_loss_real = torch.mean(torch.abs(d_real - real_imgs))
            d_loss_fake = torch.mean(torch.abs(d_fake - gen_imgs.detach()))
            d_loss = d_loss_real - k * d_loss_fake

            d_loss.backward()
            optimizer_D.step()

            # ----------------
            # Update weights
            # ----------------
            diff = torch.mean(opt.gamma * d_loss_real - d_loss_fake)

            # Update weight term for fake samples
            k = k + opt.lambda_k * np.asscalar(diff.detach().data.cpu().numpy())
            k = min(max(k, 0), 1)  # Constraint to interval [0, 1]

            # Update convergence metric
            M = (d_loss_real + torch.abs(diff)).item()
            cur_M += M

            # --------------
            # Log Progress
            # --------------
            sys.stdout.write("\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] -- M: %f, delta_M: %f,k: %f" % (
                epoch + 1, opt.n_epochs, i + 1, len(dataloader),
                np.asscalar(d_loss.detach().data.cpu().numpy()), np.asscalar(g_loss.detach().data.cpu().numpy()),
                M, dM, k))
            sys.stdout.flush()
            batches_done = epoch * len(dataloader) + i
        # get the M of current epoch
        cur_M = cur_M / len(dataloader)
        if cur_M < max_M:  # if current model is better than previous one
            torch.save(discriminator.state_dict(), GANs_models + '/' + model_basename + '-d.pt')
            torch.save(generator.state_dict(), GANs_models + '/' + model_basename + '-g.pt')
            dM = min(max_M - cur_M, cur_M)
            if dM < min_dM:  # if convergence threthold meets, stop training
                print(
                    "Training was stopped after " + str(epoch + 1) + " epoches since the convergence threthold (" + str(
                        min_dM) + ".) reached: " + str(dM))
                break
            cur_dM = max_M - cur_M
            max_M = cur_M
        if epoch + 1 == opt.n_epochs and cur_dM > min_dM:
            print("Training was stopped after " + str(epoch + 1) + " epoches since the maximum epoches reached: " + str(
                opt.n_epochs) + ".")
            print("WARNING: the convergence threthold (" + str(min_dM) + ") was not met. Current value is: " + str(
                cur_dM))
            print("You may need more epoches to get the most optimal model!!!")


# 传入 matrix, label; 返回插补后的 matrix
# gene_num: => 计算 img_size
def execute_and_impute(matrix, label, gene_num, ncls, job_name):
    print('\n[Info] matrix shape:', matrix.shape, ' gene_num:', gene_num, ' ncls:', ncls)
    parse_args(gene_num, ncls, job_name)

    init_job()
    init_generator(matrix, label)
    start_train()

    return do_compute(matrix, label)

# --------------------------


# 热图
def draw_heapmap(src_data, ip_data):
    # [Figure 9. c,d] 不同种类间的相关热图 (关注 Pu.1 和 Gata1 的相关性--低)
    krumsiek_genes = 'Gfi1 Fli1 Cebpa Pu.1 Gata2 Egr1 Tal1 Gata1 Klf1'.split()

    kr_orig_cor_mat = np.corrcoef(src_data[:, krumsiek_genes].X, rowvar=False)
    kr_ae_cor_mat = np.corrcoef(ip_data[:, krumsiek_genes].X, rowvar=False)

    v1 = min(kr_orig_cor_mat.min(), kr_ae_cor_mat.min())
    v2 = max(kr_orig_cor_mat.max(), kr_ae_cor_mat.max())

    kr_orig_cor_mat = pd.DataFrame(kr_orig_cor_mat, index=krumsiek_genes, columns=krumsiek_genes)
    kr_ae_cor_mat = pd.DataFrame(kr_ae_cor_mat, index=krumsiek_genes, columns=krumsiek_genes)

    ax = sns.heatmap(kr_orig_cor_mat, vmin=v1, vmax=v2, square=True, cbar_kws={'ticks': np.linspace(-1, 1.0, 6)})
    plt.figure()
    sns.heatmap(kr_ae_cor_mat, vmin=v1, vmax=v2, square=True, cbar_kws={'ticks': np.linspace(-1, 1.0, 6)})

# 伪时间下的反基因表达模式
def draw_diff_fig(src_data, ip_data):
    def sqz(x):
        x = x - x.min()
        return x / x.max()

    src_sorted = src_data[src_data.obs.dpt_order_indices].copy()
    ip_sorted = ip_data[src_data.obs.dpt_order_indices].copy()

    obs = src_sorted.obs.copy()
    obs.loc[obs.dpt_groups.values == '1', 'dpt_pseudotime'] = -sqz(
        obs.loc[obs.dpt_groups.values == '1', 'dpt_pseudotime'])
    obs.loc[obs.dpt_groups.values == '2', 'dpt_pseudotime'] = sqz(
        obs.loc[obs.dpt_groups.values == '2', 'dpt_pseudotime'])

    src_sorted.obs = obs
    ip_sorted.obs['dpt_pseudotime'] = src_sorted.obs.dpt_pseudotime

    f, axs = plt.subplots(1, 2, figsize=(10, 4))

    X = ip_sorted[np.isin(ip_sorted.obs.dpt_groups.values, ('1', '2'))]
    axs[0].scatter(X.obs.dpt_pseudotime.values, X[:, 'Pu.1'].X, s=1, c='C1', label='Pu.1')
    axs[0].scatter(X.obs.dpt_pseudotime.values, X[:, 'Gata1'].X, s=1, c='C2', label='Gata1')
    axs[0].legend()

    X = src_sorted[np.isin(src_sorted.obs.dpt_groups.values, ('1', '2'))]
    axs[1].scatter(X.obs.dpt_pseudotime.values, X[:, 'Pu.1'].X, s=1, c='C1', label='Pu.1')
    axs[1].scatter(X.obs.dpt_pseudotime.values, X[:, 'Gata1'].X, s=1, c='C2', label='Gata1')
    axs[1].legend()

    # sc.pl.diffmap(src_sorted, color='dpt_pseudotime',
    #               title='GMP-MEP branches',
    #               color_map='coolwarm', size=90)
    # [Figure 9] E, F
    gene1 = 'Gata1'
    gene2 = 'Pu.1'

    f, ax = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)

    coef = np.corrcoef(src_sorted.X[:, src_data.var_names == gene1].reshape(-1),
                       src_sorted.X[:, src_data.var_names == gene2].reshape(-1))[0, 1]
    sc.pl.scatter(src_sorted[np.isin(src_sorted.obs.dpt_groups.values, ('1', '2'))],
                  gene1, gene2, color='dpt_pseudotime', size=90, use_raw=False, ax=ax[0],
                  color_map='coolwarm', title='Original (pearson: %.3f)' % coef, show=True)

    coef = np.corrcoef(ip_sorted.X[:, src_data.var_names == gene1].reshape(-1),
                       ip_sorted.X[:, src_data.var_names == gene2].reshape(-1))[0, 1]
    sc.pl.scatter(ip_sorted[np.isin(ip_sorted.obs.dpt_groups.values, ('1', '2'))],
                  gene1, gene2, color='dpt_pseudotime', size=90, use_raw=False, ax=ax[1],
                  color_map='coolwarm', title='Denoised (pearson: %.3f)' % coef, show=True)

    plt.subplots_adjust(right=0.8)


    # zero_idx = (src_sorted.X[:, ip_data.var_names.values == 'Pu.1'] == 0) & (
    #         src_sorted.X[:, ip_data.var_names.values == 'Gata1'] == 0)
    # zero_idx = zero_idx.ravel()
    #
    # coef = np.corrcoef(ip_sorted.X[zero_idx, ip_data.var_names == gene1].reshape(-1),
    #                    ip_sorted.X[zero_idx, ip_data.var_names == gene2].reshape(-1))[0, 1]

    # sc.pl.scatter(ip_sorted[zero_idx, :],
    #               gene1, gene2, color='dpt_pseudotime',
    #               color_map='coolwarm', title='Denoised-2 (pearson: %.3f)' % coef,
    #               show=True, size=90, use_raw=False)

def impute_paul15():
    # Load src data
    src_data = sc.datasets.paul15()
    # 数据处理: scIGANs 中以 image 形式存储, 需要截断为 X^2 宽度
    base = math.floor(math.sqrt(src_data.n_vars))
    src_data = src_data[:, :base * base]  # 与 ip_data 保持一致

    genes = src_data.var_names.to_native_types()
    genes[genes == 'Sfpi1'] = 'Pu.1'
    src_data.var_names = pd.Index(genes)
    src_data.raw = src_data.copy()

    sc.pp.log1p(src_data)
    sc.pp.pca(src_data)
    sc.pp.neighbors(src_data, n_neighbors=20, use_rep='X', method='gauss')
    sc.tl.dpt(src_data, n_branchings=1)

    # Load imputation data
    ip_data = sc.datasets.paul15()
    ip_data = ip_data[:, :base * base]

    genes = ip_data.var_names.to_native_types()
    genes[genes == 'Sfpi1'] = 'Pu.1'
    ip_data.var_names = pd.Index(genes)

    tv_data = np.transpose(ip_data.X)

    df = pd.DataFrame(data=tv_data, index=None, columns=None, dtype=None, copy=False)

    labels = pd.Categorical(ip_data.obs['paul15_clusters'].values).codes

    df_data = execute_and_impute(df, labels, ip_data.n_vars, len(np.unique(labels)), 'for_dca')
    if len(ip_data) == 0:
        print('[Error] the imputed data is None!')
        return

    ip_data.X = df_data

    # TODO: is prev ?
    sc.pp.log1p(ip_data)
    sc.pp.pca(ip_data)
    sc.pp.neighbors(ip_data, n_neighbors=20, use_rep='X', method='gauss')
    sc.tl.dpt(ip_data, n_branchings=1)

    draw_heapmap(src_data, ip_data)

    draw_diff_fig(src_data, ip_data)


def test_func():
    src_data = sc.datasets.paul15()
    # 数据处理: scIGANs 中以 image 形式存储, 需要截断为 X^2 宽度
    base = math.floor(math.sqrt(src_data.n_vars))
    src_data = src_data[:, :base * base]  # 与 ip_data 保持一致

    print(src_data.shape, ' ', src_data.n_vars, ' ', src_data.n_obs)


if __name__ == '__main__':
    impute_paul15()
    # test_func()
