import os
import time
import torch
import torch.nn as nn
import torch.utils.data
import argparse
from torch import optim
import torchvision
import torch.nn.functional as nnf
import numpy as np
import cv2
import torchvision.transforms as transforms
from torchvision.models import resnet50
import random

from detr import DETR
from backbone import Backbone
from PIL import Image, ImageDraw
from sklearn.metrics import precision_recall_fscore_support

data_mean = 0.5081
data_std = 0.29

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


        anomaly= self.anomaly_model(feature,cam)


        return anomaly


class Dataset(torch.utils.data.Dataset):
    def __init__(self, train=True, aug=False):
        super(Dataset, self).__init__()

        # if train == True:
        #     data_name = '/home/ymkim/detr_anomaly/pt_dataset/train_data.pt'
        #     label_name = '/home/ymkim/detr_anomaly/pt_dataset/train_label.pt'
        # else:
        #     data_name = '/home/ymkim/detr_anomaly/pt_dataset/valid_data.pt'
        #     label_name = '/home/ymkim/detr_anomaly/pt_dataset/valid_label.pt'
        if train == True:
            data_name = '/home/ymkim/dsec/dataset/detr_dataset_test/pt_dataset/train_data.pt'
            label_name = '/home/ymkim/dsec/dataset/detr_dataset_test/pt_dataset/train_label.pt'
        else:
            data_name = '/home/ymkim/dsec/dataset/detr_dataset_test/pt_dataset/valid_data.pt'
            label_name = '/home/ymkim/dsec/dataset/detr_dataset_test/pt_dataset/valid_label.pt'


        # normalization

        self.inputs = (torch.load(data_name) - data_mean) / data_std
        self.targets = torch.load(label_name).long()

        if train is True:
            print('Trainset size:')
        else:
            print('Validset size:')

        print(self.inputs.size(), self.targets.size())

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]

    def __len__(self):
        return len(self.inputs)

def find_bbox(map):
    # cam = (map[:, 0, :, :] * 255.)
    bboxes = []
    for i in range(map.shape[1]):
        cam = map[:,i,:,:] * 255.
        map_thr = 0.2 * torch.max(cam)
        a = cam.cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        _, thr_gray_heatmap = cv2.threshold(a, int(map_thr), 255, cv2.THRESH_TOZERO)

        contours, _ = cv2.findContours(thr_gray_heatmap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) != 0:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)

            estimated_bbox = [x, y, x + w, y + h]
        else:
            estimated_bbox = [0, 0, 1, 1]
        bboxes.append(estimated_bbox)

    return bboxes

def blend_cam(image, cam, es_box=[0,0,1,1]):
    cam = cam[:, 0, :, :].cpu().numpy().transpose(1, 2, 0)
    image = image[0 , 0, :, :,:].cpu().numpy().transpose(1, 2, 0)

    I = np.zeros_like(cam)
    x1, y1, x2, y2 = es_box
    I[y1:y2, x1:x2] = 1
    cam = cam * I
    cam = (cam * 255.).astype(np.uint8)
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    blend = image * 0.2 + heatmap * 0.8

    return blend, heatmap


def draw_bbox(img,  bbox, color2=(255, 0, 0)):
    # for i in range(len(box1)):
    #     cv2.rectangle(img, (box1[i, 0], box1[i, 1]), (box1[i, 2], box1[i, 3]), color1, 2)
    tf = transforms.ToPILImage()
    # image = img[0,0,:,:,:]
    image = (img.cpu().numpy().transpose(1, 2, 0)*255.).astype(np.uint8)
    a = cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color2, 2)

    # a = cv2.rectangle(image, (10, 10), (100, 100), (255, 0, 0), -1)
    # tf(image).show()
    # cv2.imwrite('a.jpg', a)
    return image

def cal_iou(box1, box2, method='iou'):

    box1 = np.asarray(box1, dtype=float)
    box2 = np.asarray(box2, dtype=float)
    if box1.ndim == 1:
        box1 = box1[np.newaxis, :]
    if box2.ndim == 1:
        box2 = box2[np.newaxis, :]

    iw = np.minimum(box1[:, 2], box2[:, 2]) - np.maximum(box1[:, 0], box2[:, 0]) + 1
    ih = np.minimum(box1[:, 3], box2[:, 3]) - np.maximum(box1[:, 1], box2[:, 1]) + 1

    i_area = np.maximum(iw, 0.0) * np.maximum(ih, 0.0)
    box1_area = (box1[:, 2] - box1[:, 0] + 1) * (box1[:, 3] - box1[:, 1] + 1)
    box2_area = (box2[:, 2] - box2[:, 0] + 1) * (box2[:, 3] - box2[:, 1] + 1)

    if method == 'iog':
        iou_val = i_area / (box2_area)
    elif method == 'iob':
        iou_val = i_area / (box1_area)
    else:
        iou_val = i_area / (box1_area + box2_area - i_area)
    return iou_val

if __name__ == '__main__':

    # seed_num = 730
    #
    # torch.manual_seed(seed_num)
    # torch.cuda.manual_seed(seed_num)
    #
    # np.random.seed(seed_num)
    # torch.cuda.manual_seed_all(seed_num)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # random.seed(seed_num)
    # seed = list(range(730, 750))
    seed = [730]

    for seed_num in seed:

        parser = argparse.ArgumentParser(description='DETR_Anomaly')
        parser.add_argument('--batch_size', type=int, default=1)
        parser.add_argument('--num_epochs', type=int, default=101)
        parser.add_argument('--lr', type=float, default=1e-4)
        parser.add_argument('--gpu_ids', nargs='+', type=int, default=[0])
        parser.add_argument('--n_defect', type=int, default=1)
        parser.add_argument('--use_attn', type=int, default=0)
        parser.add_argument('--use_cam', type=int, default=0)
        parser.add_argument('--save_img', type=int, default=0)


        args = parser.parse_args()
        # if args.use_attn == 1:
        #     vis_dir = 'visualization_cam2'
        # else:
        #     vis_dir = 'visualization_cam2'
        # if not os.path.isdir(vis_dir):
        #     os.mkdir(vis_dir)

        train_dataset = Dataset(train=True, aug=False)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        valid_dataset = Dataset(train=False, aug=False)
        valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                                                       num_workers=0)

        model = anomalyDETR(args.n_defect)

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

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
        # 오리지날에서는 param 27 / seed 735가 가장 좋다.
        # 이번에는 로스에서 weight를 주어 계산할 예정이다. 또한 데이터를 늘려서 실험할 예정이다.
        criterion = torch.nn.BCELoss()
        # criterion = torch.nn.CrossEntropyLoss(weight=train_dataset.weight.to(device))
        # param = list(range(11, 30))
        param = [23]
        # param = torch.linspace(500, 1000, 51).int().tolist()
        # param = torch.linspace(1, 20, 21).int().tolist()
        # param_name = 'parameters_deform1'+ str(seed_num)
        param_name = 'parameters_seed_' + str(seed_num)
        # param_name = 'k4'

        vis_name = 'visualization'
        output_array = []

        result_file = open('result.txt', 'w')
        for i in param:

            if args.use_attn == 1:
                vis_dir = vis_name + str(i)
            else:
                vis_dir = vis_name + str(i)
            if not os.path.isdir(vis_dir):
                os.mkdir(vis_dir)

            param_num = int(i)
            state = torch.load('/home/ymkim/detr_anomaly/detr_yh_std/' + param_name + '/%d_parameter' % param_num, map_location=device)
            model.load_state_dict(state['state_dict'])
            optimizer.load_state_dict(state['optimizer'])

            model.eval()

            origin_inputs = []
            origin_outputs = []
            frame_outputs = []
            out_prob = []
            out_cam = []
            out_cam1 = []
            out_blend = []
            k = 0
            anomaly_prob = []
            iogs = []
            threshold = torch.linspace(0, 1, 100).tolist()
            tp, fp, fn, tn, f1 = {}, {}, {}, {}, {}
            predicted_framelevel = {}

            for th in threshold:
                tp[th] = 0;
                fp[th] = 0;
                fn[th] = 0;
                tn[th] = 0;
                f1[th] = 0;
                predicted_framelevel[th] = []

            # validation
            time_process_model = []
            q = 0
            with torch.set_grad_enabled(False):
                for inputs, targets in valid_dataloader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    if (len(inputs) != args.batch_size):
                        break

                    # make target frame-level
                    targets_frame = torch.zeros(targets.shape[0], targets.shape[1], args.n_defect).to(device)

                    for i in range(targets.shape[0]):
                        for j in range(targets.shape[1]):
                            if targets[i][j].sum() > 500:
                                targets_frame[i][j] = 1
                            else:
                                targets_frame[i][j] = 0

                    origin_inputs.append(inputs.cpu() * data_std + data_mean)
                    # start_time = time.time()
                    map, anomaly, attn, cam = model(inputs)

                    # q+=1
                    # if q > 30:
                    #     print("30 time mean: ", np.mean(time_process_model))
                    # else:
                    #     time_process_model.append(time.time() - start_time)

                    cam = nnf.sigmoid(cam)

                    map = map.squeeze(2)
                    if args.use_attn == 1:
                        # map = map * attn
                        map = attn


                    # frame_outputs.append(targets_frame.cpu())
                    # for k in range(targets_frame.shape[1]):
                    #     frame_outputs.append(targets_frame[0,k,0].item())

                    if targets_frame.sum().item() > 1:
                        frame_outputs.append(1)
                    else:
                        frame_outputs.append(0)

                    # map
                    map = nnf.interpolate(map, size=(256, 256), mode='bilinear', align_corners=False)
                    # cam_rv = nnf.interpolate(cam_rv, size=(256, 256), mode='bilinear', align_corners=False)
                    output_array.append(map.squeeze(0).cpu())

                    cam_rv1 = nnf.interpolate(cam[:, 0, :, :].unsqueeze(1), size=(256, 256), mode='bilinear', align_corners=False)
                    # out_cam.append(cam_rv.cpu())
                    out_cam.append(cam_rv1.cpu())


                    # bbox test
                    #test: attention map + cam
                    # map = map + cam_rv1.view(cam_rv1.shape[1], cam_rv1.shape[0],cam_rv1.shape[2],cam_rv1.shape[3])
                    # map = (map/torch.max(map))

                    estimated_bbox = find_bbox(map)
                    # estimated_bbox = find_bbox(map-0.5)
                    gt_bbox = find_bbox(targets)

                    map_list = []
                    for i in range(len(estimated_bbox)):
                        if gt_bbox[i] == [0,0,1,1]:
                            # pass
                            temp = map[0, i, :, :].cpu()
                            temp = temp.unsqueeze(0).repeat(3,1,1)
                            map_list.append(temp)
                        else:
                            if targets[0][i].sum() > 1000:
                                if estimated_bbox[i] == [0,0,1,1]:
                                    iog = 'nobox'
                                    iogs.append(iog)
                                else:
                                    iog = cal_iou(estimated_bbox[i], gt_bbox[i], method='iou')
                                    iogs.append(iog)
                            else:
                                iog = 'none'

                            #save img - debug
                            ori_img = inputs[0,i,:,:,:].cpu() * data_std + data_mean
                            ori_img = ori_img.numpy().transpose(1, 2, 0)*255
                            #

                            rect_est_img = draw_bbox(map[0,i,:,:].unsqueeze(0), estimated_bbox[i])
                            # rect_est_img = draw_bbox(cam_rv1[0, i, :, :].unsqueeze(0), estimated_bbox[i])
                            rect_gt_img = draw_bbox(targets[0,i,:,:].unsqueeze(0), gt_bbox[i])
                            rect_est_img = cv2.cvtColor(rect_est_img, cv2.COLOR_GRAY2BGR)

                            cam_img = cam_rv1[i,:,:,:]
                            cam_img = (cam_img.cpu().numpy().transpose(1, 2, 0) * 255.).astype(np.uint8)



                            cam_img = cv2.cvtColor(cam_img, cv2.COLOR_GRAY2BGR)

                            #temp
                            # map_tmp = map[0, i, :, :].unsqueeze(0).cpu().numpy().transpose(1,2,0)*255
                            # map_tmp = cv2.cvtColor(map_tmp, cv2.COLOR_GRAY2BGR)

                            # put iou text
                            # cv2.putText(rect_est_img, str(iog), (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))


                            rect_gt_img = cv2.cvtColor(rect_gt_img, cv2.COLOR_GRAY2BGR)
                            temp = np.hstack((ori_img, rect_gt_img, rect_est_img, cam_img))
                            # temp = np.hstack((ori_img, rect_gt_img, map_tmp, cam_img))

                            # map_tmp = torch.tensor(rect_est_img).permute(2, 0, 1)
                            to_tensor = transforms.ToTensor()
                            map_tmp = to_tensor(rect_est_img)

                            map_list.append(map_tmp)

                            # cv2.imwrite('temp/'+str(k)+'.jpg', temp)
                            k=k+1



                    # blend, heatmap = blend_cam(inputs.cpu() * 0.5 + 0.5, map, estimated_bbox)
                    # blend = torch.Tensor(heatmap)
                    temp = []
                    # for i in range(10):
                    #     temp.append(blend.cpu())
                    # out_blend.append(torch.cat(temp, 0).view(10,3,256,256))

                    map_s = torch.cat(map_list, 0)  # N by 10 by 3 by h by w
                    map_s = map_s.unsqueeze(0)
                    map_s.view(-1, 3, map_s.shape[2],map_s.shape[3])
                    out_prob.append(map_s)
                    # out_prob.append(map.cpu())
                    origin_outputs.append(targets.cpu())

                    for th in threshold:

                        # put output frame-level prediction
                        th_temp = (map > th).int()
                        # th_temp = th_temp.view(-1, 10, 93, 161)
                        # for i in range(1):

                        total_pixel = 1
                        # for j in range(origin_outputs[i].element_size()):
                        # for j in range(3):
                        total_pixel = th_temp.shape[0] * th_temp.shape[1] * th_temp.shape[2] * th_temp.shape[3]

                        defect_ratio = (th_temp == 1).sum().item() / total_pixel

                        if (defect_ratio > 0.01):
                            predicted_framelevel[th].append(1)
                        else:
                            predicted_framelevel[th].append(0)
            #
            # torch.save(torch.stack(anomaly_prob), 'output.pt')

            #metric
            #localization accuracy
            tp = 0; fp = 0; fn = 0;
            iou_thres = 0.5

            for i in range(len(iogs)):
                if iogs[i] == 'nobox':
                    fn = fn+1
                elif iogs[i] >= iou_thres:
                    tp = tp + 1
                elif iogs[i] < iou_thres:
                    fp = fp + 1
                    fn = fn + 1

            p = tp / (tp + fp + 1e-7)
            r = tp / (tp + fn + 1e-7)
            f1 = 2 / (1 / (p+ 1e-7) + 1 / (r + 1e-7))
            a = tp / (len(iogs) + 1e-7)
            # print('epoch:', param_num, ' accuracy:', a, ' precision:', p, ' recall:', r, ' f1score:', f1)
            # print('tp:', tp, ' fp: ', fp, ' fn: ', fn )
            # result = 'epoch:'+ str(param_num) + ' accuracy:' + str(a) + ' precision:'+ str(p) + ' recall:' + str(r) + ' f1score:' + str(f1) + '\n'
            result = 'epoch:'+ str(param_num) + ' f1score:' + str(f1) + '\n'
            print('epoch:', param_num,  ' loc acc:', a)
            result_file.writelines(result)




            origin_inputs = torch.cat(origin_inputs, 0)  # N by 10 by 3 by h by w
            origin_outputs = torch.cat(origin_outputs, 0)
            output_array = torch.cat(output_array, 0)

            # anomaly_prob = torch.cat(anomaly_prob, 0)
            frame_outputs = torch.tensor(frame_outputs)
            out_prob = torch.cat(out_prob, 0)  # 10*k by 10 by h by w
            out_prob = out_prob.view(out_prob.shape[0], -1, 3, out_prob.shape[2], out_prob.shape[3])
            out_cam = torch.cat(out_cam, 0)
            out_cam = out_cam.view(-1, out_prob.shape[1], out_cam.shape[2], out_cam.shape[3])
            # out_cam1 = torch.cat(out_cam1, 0)
            # out_blend = torch.cat(out_blend, 0)
            # out_blend = out_blend.view(-1, 10,3,256,256)
            # map
            # out_prob = out_prob.unsqueeze(2)
            out_cam = out_cam.unsqueeze(2)
            # out_cam1 = out_cam1.unsqueeze(2)

            origin_outputs = origin_outputs.unsqueeze(2)

            torch.save(origin_outputs.view(-1,256,256), 'target.pt')
            torch.save(output_array, 'output.pt')
            # print(1)

            # for th in threshold:
            #     y_pred = (anomaly_prob.cpu() > th).int()
            #     result = precision_recall_fscore_support(frame_outputs.squeeze().int().cpu(), y_pred.squeeze(),
            #                                              average='micro')
            #     print(result)
            p_frame = [];            r_frame = [];            f1_frame = [];            a_frame = []
            for th in threshold:

                tp = 0
                fp = 0
                tn = 0
                fn = 0
                for i in range(len(predicted_framelevel[0])):
                    if (predicted_framelevel[th][i] == 0 and frame_outputs.int()[i].item() == 0):
                        tn = tn + 1
                    elif (predicted_framelevel[th][i] == 0 and frame_outputs.int()[i].item() == 1):
                        fn = fn + 1
                    elif (predicted_framelevel[th][i] == 1 and frame_outputs.int()[i].item() == 0):
                        fp = fp + 1
                    elif (predicted_framelevel[th][i] == 1 and frame_outputs.int()[i].item() == 1):
                        tp = tp + 1

                precision_framelevel = tp / (tp + fp + 1e-7)
                recall_framelevel = tp / (tp + fn + 1e-7)
                f1score_framelevel = 2 * precision_framelevel * recall_framelevel / (
                            precision_framelevel + recall_framelevel + 1e-7)
                accuracy_framelevel = (tp + tn) / (tp + fp + tn + fn + 1e-7)

                p_frame.append(precision_framelevel)
                r_frame.append(recall_framelevel)
                a_frame.append(accuracy_framelevel)

            print('cls acc: ', max(a_frame))


            if args.save_img == 1:

                for i in range(len(out_prob)):

                    if not os.path.isdir('%s/iter_%03d' % (vis_dir, state['epoch'])):
                        os.mkdir('%s/iter_%03d' % (vis_dir, state['epoch']))

                    scaled_file = '%s/iter_%03d/%04d.png' % (vis_dir, state['epoch'], i)
                    pred_file = '%s/iter_%03d/%04d_pred.png' % (vis_dir, state['epoch'], i)
                    scaled_input = torchvision.utils.make_grid(origin_inputs[i], nrow=10, padding=2, pad_value=1)
                    # output_and_target = torch.cat([origin_outputs[i].float(), out_prob[i]], 0).unsqueeze(1)
                    output = out_prob[i]
                    # out_cam = out_cam[i]
                    # output = nnf.interpolate(output, size=(256, 256), mode='bicubic', align_corners=False)
                    scaled_output = torchvision.utils.make_grid(output, nrow=10, padding=2, pad_value=1)
                    scaled_output_cam = torchvision.utils.make_grid(out_cam[i], nrow=10, padding=2, pad_value=1)
                    # scaled_output_cam1 = torchvision.utils.make_grid(out_cam1[i], nrow=10, padding=2, pad_value=1)
                    # scaled_output_cam2 = torchvision.utils.make_grid(out_blend[i], nrow=10, padding=2, pad_value=1)
                    scaled_target = torchvision.utils.make_grid(origin_outputs[i], nrow=10, padding=2, pad_value=1)

                    scaled_result = torch.cat([scaled_input, scaled_target.float()], 1)
                    scaled = torch.cat([scaled_result, scaled_output], 1)
                    # scaled = torch.cat([scaled_result, scaled_output_cam1], 1)
                    scaled = torch.cat([scaled, scaled_output_cam], 1)
                    # scaled = torch.cat([scaled, scaled_output_cam2], 1)

                    torchvision.utils.save_image(scaled, scaled_file)

        result_file.close()













