import os
import time
import argparse
import numpy as np

from dataload.dataset_entire import MILDataset

# from models.attention_modelparallel import ModelParallelABMILLight, ABMIL
from models.model import ABMIL

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchnet import meter
from tensorboardX import SummaryWriter
from ec_tools import colorful_str as cs
from ec_tools import procedure


def get_args():
    parser = argparse.ArgumentParser(description="Attention-MIL")
    parser.add_argument('--input', '-i', type=str, default=None, help="path of input")
    parser.add_argument('--output', '-o', type=str, default=None, help="path of output")
    parser.add_argument('--gpus', '-g', type=str, nargs='+', help="device id of GPU")
    parser.add_argument('--optim', '-opt', type=str, default="Adam", help="type of optimizer")
    parser.add_argument('--lr', '-l', type=float, default=0.001, help="learning rate")
    parser.add_argument('--checkpoint', '-ckpt', type=str, default=None, help="path of checkpoint")
    parser.add_argument("--epoch", "-ep", type=int, default=500, help="max epoch")
    parser.add_argument("--instance_eval", '-inst_eval', action='store_true', default=False, help="use instance constraint")
    parser.add_argument("--instance_weight", '-inst_weight', type=float, default=0.3, help="the weight of instance constraint in Loss fn")
    parser.add_argument("--k_sample", '-k', type=int, default=3, help="sample num")
    args = parser.parse_args()
    return args


def train():
    args = get_args()
    # print(args.gpus, args.input)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpus)

    with procedure("Init Model") as p:
        # model = ABMIL('resnet101', 2048).cuda()

        # model = ABMIL('mobilenetv2_10', 1280).cuda()
        # model = ModelParallelABMILLight(fc_input_dims=1280, num_classes=2, width_mult=0.5)
        if args.instance_eval:
            model = ABMIL(encoder_name='mobilenetv2_10', instance_loss_fn=nn.CrossEntropyLoss(), k_sample=args.k_sample)
        else:
            model = ABMIL(encoder_name='mobilenetv2_10', k_sample=args.k_sample)
        model.relocate(model_parallel=False)

    if args.checkpoint is not None:
        with procedure("Load param") as p:
            ckpt = torch.load(args.checkpoint)
            params = {"encoder."+k: v for k, v in ckpt['state_dict'].items()}

            state_dict = model.state_dict()
            state_dict.update(params)
            model.load_state_dict(state_dict)

    with procedure("Init Loss and optimizer") as p:
        for k, v in model.named_parameters():
            if 'bn' in k:
                v.requires_grad = False
        ceriterion = torch.nn.CrossEntropyLoss()
        if args.optim == 'Adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
        else:
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.9, patience=200, verbose=True)
        p.add_log(cs("(#y)optimizer:{}\tLR:{}".format(args.optim, args.lr)))

    with procedure("Prepare Dataset") as p:
        train_dset = MILDataset(args.input, train=True)
        val_dset = MILDataset(args.input, train=False)
        train_loader = DataLoader(train_dset, batch_size=1, shuffle=True, num_workers=1, pin_memory=True)
        val_loader = DataLoader(val_dset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    loss_meter = meter.AverageValueMeter()
    instance_loss_meter = meter.AverageValueMeter()
    bag_loss_meter = meter.AverageValueMeter()

    best_acc = 0.0
    best_acc_epoch = 0
    best_rec = 0.0
    best_rec_epoch = 0

    start_epoch = 0

    # if args.checkpoint is not None:
    #     with procedure("Load Param") as p:
    #         checkpoint = torch.load(args.checkpoint)
    #         model.load_state_dict(checkpoint['state_dict'])
    #         start_epoch = chectpint['epoch']
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         scheduler.load_state_dict(checkpoint['scheduler'])
    #         best_acc = checkpoint['best_acc']
    #         p.add_log(cs("(#y)Load ckpt from {}:\t Epoch: {}\t Lr:{}".format(config.MODEL.CHECKPOINT, start_epoch,
    #                                                           optimizer.param_groups[0]['lr'])))

    # 训练结果记录文件的路径
    output_root_dir = os.path.join(args.output, "Attention",
                                   time.strftime("%Y%m%d-%H%M%S", time.localtime()))
    log_root_dir = os.path.join(output_root_dir, 'log')
    ckpt_root_dir = os.path.join(output_root_dir, 'checkpoint')

    tensorboard_root_dir = os.path.join(output_root_dir, 'tensorboard')
    os.makedirs(log_root_dir, exist_ok=True)
    os.makedirs(ckpt_root_dir, exist_ok=True)
    os.makedirs(tensorboard_root_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_root_dir)
    log_handle = open(os.path.join(log_root_dir, 'log.txt'), 'a')
    log_out("Start training from epoch: {} ...".format(start_epoch), log_handle)

    for epoch in range(start_epoch, args.epoch):
        loss_meter.reset()
        bag_loss_meter.reset()
        instance_loss_meter.reset()
        model.train()
        for name, m in model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

        for each_iter, (imgs, labels, name) in enumerate(train_loader):
            input = imgs.squeeze().to("cuda:0")
            optimizer.zero_grad()
            output, y_hat, y_prob, instance_results, weights = model(input, label=labels)
            target = labels.to(output.device)
            # if each_iter == 0:
            #     print("train", output)
            # print(output.size(), output2.size(), target.size())\
            # print("train", name, output, target)
            if args.instance_eval:
                bag_loss = ceriterion(output, target)
                loss = (1 - args.instance_weight) * instance_results["instance_loss"] + args.instance_weight * bag_loss
                bag_loss_meter.add(bag_loss.item())
                instance_loss_meter.add(instance_results['instance_loss'].item())
                loss_meter.add(loss.item())
                writer.add_scalar('Train/iter_loss', loss.item(), epoch * len(train_loader) + each_iter)
                writer.add_scalar('Train/instance_loss', instance_results["instance_loss"].item(), epoch * len(train_loader) + each_iter)
                writer.add_scalar('Train/bag_loss', bag_loss.item(), epoch * len(train_loader) + each_iter)
            else:

                loss = ceriterion(output, target)
                loss_meter.add(loss.item())
                writer.add_scalar('Train/iter_loss', loss.item(), epoch * len(train_loader) + each_iter)
            # print(cs("(#r) [train] [{} {}\{}]: loss: {}".format(epoch, each_iter, len(train_loader), loss.data)))

            loss.backward()
            # print(model.classifier.weight.grad)
            optimizer.step()
            # print(cs("(#y)[{}] train_time: {}".format(each_iter, time.time() - train_time)))

        epoch_loss = loss_meter.value()[0]
        writer.add_scalar('Train/epoch_loss', epoch_loss, epoch)
        log_out("[Train] Epoch:{}\tLoss:{}\tlr:{}".format(epoch + 1, epoch_loss, optimizer.param_groups[0]['lr']),
                log_handle)
        val_loss, val_acc, val_rec, val_cm = val(model, val_loader, ceriterion, writer, epoch, args.instance_eval, args.instance_weight)

        log_out("[Val] Epoch:{}\t val_loss:{}\tval_acc:{} \t val_rec:{}\t val_cm:{}".format(epoch+1, val_loss, val_acc, val_rec, val_cm), log_handle)

        writer.add_scalar('Val/val_acc', val_acc, epoch)
        writer.add_scalar('Val/val_rec', val_rec, epoch)

        if val_acc > best_acc:
            best_acc = val_acc
            best_acc_epoch = epoch + 1
            checkpoint = {
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': best_acc_epoch
            }
            pth_name = os.path.join(ckpt_root_dir, 'BestAcc.pth')
            torch.save(checkpoint, pth_name)
        if (epoch + 1) % 2 == 0:
            checkpoint = {
                'state_dict': model.state_dict(),
                'best_acc': val_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch+1
            }
            pth_name = os.path.join(ckpt_root_dir, 'epoch-{}.pth'.format(epoch + 1))
            torch.save(checkpoint, pth_name)
        if val_rec > best_rec:
            best_rec = val_rec
            best_rec_epoch = epoch + 1

        log_out("[Val] best_acc: {}\tbest_acc_epoch: {}".format(best_acc, best_acc_epoch), log_handle)
        log_out("[Val] best_rec: {}\tbest_rec_epoch: {}".format(best_rec, best_rec_epoch), log_handle)
        # scheduler.step(val_acc)
        # if epoch > 5:
        #     optimizer.param_groups[0]['lr'] = 0.001
    log_handle.close()


def val(model, val_loader, ceriterion, writer, epoch, instance_eval, instance_weight):
    model.eval()
    loss_val_meter = meter.AverageValueMeter()
    cm = meter.ConfusionMeter(2)

    with torch.no_grad():
        for each_iter, (imgs, labels, name) in enumerate(val_loader):
            input = imgs.squeeze().to("cuda:0")

            output, y_hat, y_prob, instance_results, weights = model(input, label=labels)
            target = labels.to(output.device)

            if instance_eval:
                loss = (1 - instance_weight) * instance_results[
                    "instance_loss"] + instance_weight * ceriterion(output, target)
            else:
                loss = ceriterion(output, target)
            # print(cs("(#r) [val]_{}: loss: {}\t output: {}\t label: {}".format(each_iter, loss.data, output.data, target)))

            loss_val_meter.add(loss.item())
            # print(target.unsqueeze(0).size(), output.size())
            cm.add(output.detach(), target.type(torch.LongTensor))
            writer.add_scalar('Val/iter_loss', loss.item(), epoch * len(val_loader) + each_iter)

        val_loss = loss_val_meter.value()[0]
        writer.add_scalar('Val/val_loss', val_loss, epoch)
        val_cm = cm.value()
        val_acc = 100.0 * (val_cm[0][0] + val_cm[1][1]) / val_cm.sum()
        val_rec = 100.0 * val_cm[1][1] / (val_cm[1][0] + val_cm[1][1])
        return val_loss, val_acc, val_rec, val_cm


def log_out(out_str, f_out, verbose=True):
    f_out.write(out_str + '\n')
    f_out.flush()
    if verbose:
        print(out_str)


if __name__ == '__main__':
    train()
