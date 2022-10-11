import torch
import torch.nn as nn
import os
import time
from detr import DETR
from backbone import Backbone
import argparse
from torch import optim
import torch.nn.functional as F
import os
import numpy as np
import random
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from torchvision.models import resnet50

def dfs_freeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        dfs_freeze(child)

def adaptive_min_pooling_loss(x):
    # This loss does not affect the highest performance, but change the optimial background score (alpha)
    n,c,h,w = x.size()
    k = h*w//4
    x = torch.max(x, dim=1)[0]
    y = torch.topk(x.view(n,-1), k=k, dim=-1, largest=False)[0]
    y = F.relu(y, inplace=False)
    loss = torch.sum(y)/(k*n)
    return loss


class anomalyDETR(nn.Module):

    def __init__(self, n_detect):
        super().__init__()
        self.backbone = Backbone()
        # self.backbone = resnet50()
        # del self.backbone.fc
        #
        # self.conv = nn.Conv2d(2048, 128, 1)
        # self.dropout_cam = torch.nn.Dropout2d(0.5)
        # self.fc_cam = nn.Conv2d(128, 2, 1, bias=False)
        # self.conv = nn.Conv2d(3, 128, kernel_size=1, stride=8)

        self.anomaly_model = DETR(n_detect)

    def forward(self, input):


        feature, cam = self.backbone(input)
        # put cam
        # input_add = self.conv(input.view(-1, input.shape[2], input.shape[3],input.shape[4]))
        # input_add = F.sigmoid(input_add)

        # input_add = input_add.view(feature.shape)

        # feature_new = feature/torch.max(feature) + input_add/torch.max(input_add)
        # feature_new = feature_new/torch.max(/feature_new)
        anomaly= self.anomaly_model(feature,cam)

        # anomaly= self.anomaly_model(feature,cam)
        return anomaly



class Dataset(torch.utils.data.Dataset):
    def __init__(self, train=True, aug=False):
        super(Dataset, self).__init__()

        # if train == True:
        #     data_name = '/root/dataset/pt_dataset/train_data.pt'
        #     label_name = '/root/dataset/pt_dataset/train_label.pt'
        # else:
        #     data_name = '/root/dataset/pt_dataset/valid_data.pt'
        #     label_name = '/root/dataset/pt_dataset/valid_label.pt'

        if train == True:
            data_name = '/home/ymkim/dsec/dataset/detr_dataset_test/pt_dataset/train_data.pt'
            label_name = '/home/ymkim/dsec/dataset/detr_dataset_test/pt_dataset/train_label.pt'
        else:
            data_name = '/home/ymkim/dsec/dataset/detr_dataset_test/pt_dataset/valid_data.pt'
            label_name = '/home/ymkim/dsec/dataset/detr_dataset_test/pt_dataset/valid_label.pt'
        # normalization, train
        data_mean = 0.5081
        data_std = 0.29
        self.inputs = (torch.load(data_name) - data_mean) / data_std
        self.targets = torch.load(label_name).long()


        counts = [(self.targets == 0).sum().item(),
                  (self.targets == 1).sum().item()]

        counts = [counts[1] / sum(counts), counts[0] / sum(counts)]
        counts = torch.tensor(counts)
        self.weight = counts

        if train is True:
            print('Trainset size:')
        else:
            print('Validset size:')

        print(self.inputs.size(), self.targets.size())

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]

    def __len__(self):
        return len(self.inputs)



def load_checkpoint(model, optimizer, filename='checkpoint.pth.tar'):
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch

if __name__ == '__main__':



    seed = [730]
    for seed_num in seed:

        torch.manual_seed(seed_num)
        torch.cuda.manual_seed(seed_num)

        np.random.seed(seed_num)
        torch.cuda.manual_seed_all(seed_num)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(seed_num)


        parser = argparse.ArgumentParser(description='DETR_Anomaly')
        parser.add_argument('--batch_size', type=int, default=1)
        parser.add_argument('--num_epochs', type=int, default=31)
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--gpu_ids', nargs='+', type=int, default=[0])
        parser.add_argument('--n_defect', type=int, default=1)

        args = parser.parse_args()

        train_dataset = Dataset(train=True, aug=False)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

        model = anomalyDETR(args.n_defect)

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[10, 20, 70], gamma=0.5)
        if torch.cuda.device_count() > 1:
            if args.gpu_ids == None:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                device = torch.device('cuda:0')
            else:
                print("Let's use", len(args.gpu_ids), "GPUs!")
                device = torch.device('cuda:' + str(args.gpu_ids[0]))
        else:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        print('args.gpu_ids', args.gpu_ids)

        # model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)
        model = model.to(device)
        param_name = 'k4'
        # param_name = 'parameters_seed_' + str(seed_num)
        filename = param_name + '/50_parameter'
        model, optimizer, epoch = load_checkpoint(model, optimizer, filename)

        criterion = torch.nn.BCELoss()
        # criterion = torch.nn.CrossEntropyLoss(weight=train_dataset.weight.to(device))

        if not os.path.isdir(param_name):
            os.mkdir(param_name)

        if epoch == '':
            num_epochs = 0
        else:
            num_epochs = epoch

        for epoch in range(num_epochs, args.num_epochs):
            start_time = time.time()

            each_train_loss = []
            model.train()

            with torch.set_grad_enabled(True):
                for inputs, targets in train_dataloader:

                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    # make target frame-level
                    targets_frame = torch.zeros(targets.shape[0], targets.shape[1], args.n_defect).to(device)
                    for i in range(targets.shape[0]):
                        for j in range(targets.shape[1]):
                            if torch.sum(targets[i][j]).item() > 400:
                                targets_frame[i][j] = 1
                            else:
                                targets_frame[i][j] = 0


                    optimizer.zero_grad()
                    # s = 2
                    #
                    # for i in range int((inputs.shape[1]/s)):
                    #
                    # cam_epoch = 10
                    #
                    # if epoch > cam_epoch:
                    #     dfs_freeze(model.backbone)
                    #     # model.backbone.eval()
                    #     # for p in model.backbone.parameters():
                    #     #     p.requires_grad = False
                    #     optimizer = optim.Adam(model.parameters(), lr=args.lr)


                    map, anomaly, attn, cam = model(inputs)

                    label1 = F.adaptive_avg_pool2d(cam[:,0,:,:], (1, 1))
                    label1 = label1.view(1,10,1)
                    label1 = F.sigmoid(label1)

                    # label1 = F.relu(label1, inplace=False)

                    loss_cam =criterion(label1, targets_frame)

                    # loss_rvmin1 = adaptive_min_pooling_loss((cam_rv1 * label)[:, 1:, :, :])
                    # cam1 = F.interpolate(visualization.max_norm(cam1), scale_factor=scale_factor, mode='bilinear',
                    #                      align_corners=True) * label
                    # cam_rv1 = F.interpolate(visualization.max_norm(cam_rv1), scale_factor=scale_factor, mode='bilinear',
                    #                         align_corners=True) * label
                    loss_class =criterion(anomaly, targets_frame)


                    err = loss_cam + loss_class

                    # if epoch < cam_epoch:
                    #     err = loss_cam
                    # else:
                    #     err = loss_class
                    # err = loss_cam



                    err.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()
                    each_train_loss.append(err.item())
            scheduler.step()

            epoch_train_loss = sum(each_train_loss) / len(each_train_loss) + 1e-5
            print('epoch: ', epoch, 'loss: ', epoch_train_loss, ', time: %4.2f' % (time.time() - start_time), ', lr: ', optimizer.param_groups[0]['lr'])

            if epoch >10:
                state = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                par_filepath = param_name + '/%d_parameter' % (epoch)
                torch.save(state, par_filepath)




