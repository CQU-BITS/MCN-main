
import os, random
import warnings
import torch
import time
import numpy as np
import datasets
from torch import nn, optim
from torch.utils.data import DataLoader
from postprocessing.myplots import plotLossAcc, plotTSNECluster, plotConfusionMatrix
import models


speed = 9
cond = str(speed)
batch_size = 128
lr = 1e-3
max_epochs = 200
model_name = 'LaplaceAlexNet'  # choices=['WDCNN', 'LaplaceAlexNet']
domain = 'Time'  # choices=['Time']
dataset_name = 'SQ_bearing_dataset'  # choices=['CQU_gear_dataset', 'SQ_bearing_dataset']
result_file_save_path = os.path.join('Results', dataset_name, model_name, cond)


class Iterator_utils(object):
    def setup(self):
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.domain = domain
        self.lr = lr
        self.result_file_save_path = result_file_save_path
        self.num_workers = 0
        # Consider the gpu or cpu condition
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.device_count = torch.cuda.device_count()
            print('using {} gpus'.format(self.device_count))
            assert batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            warnings.warn("gpu is not available")
            self.device = torch.device("cpu")
            self.device_count = 1
            print('using {} cpu'.format(self.device_count))
        # Load the datasets
        self.datasets = {}
        if dataset_name in ['CQU_gear_dataset', 'SQ_bearing_dataset']:
            Dataset = getattr(datasets, dataset_name)
            self.datasets['train'], self.datasets['val'], self.datasets['test'], _ = Dataset(speed=speed,
                                                                                             snr=None,
                                                                                             normlizetype=None,
                                                                                             domain=self.domain).data_preprare()
        else:
            assert "The dataset is not involved"
        self.dataloaders = {x: DataLoader(self.datasets[x],
                                          batch_size=(self.batch_size if 'train' in x else 2000),
                                          shuffle=(True if 'train' in x else False),
                                          num_workers=self.num_workers,
                                          pin_memory=(True if self.device == 'cuda' else False)) for x in ['train', 'val', 'test']}
        # 定义模型、loss 准则和优化器
        if self.model_name in ['WDCNN', 'LaplaceAlexNet']:
            self.model = getattr(models, self.model_name)(in_channels=1, num_classes=Dataset.num_classes).to(self.device)
            self.criterion = nn.CrossEntropyLoss().to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        else:
            assert "Selected model is not involved"


    def my_trainer(self):  # Training and validating process
        start_time = time.time()
        best_val_acc = 0.
        n_step = 0

        trainLoss, trainAcc, valLoss, valAcc = [], [], [], []
        for epoch in range(self.max_epochs):
            for phase in ['train', 'val']:
                epoch_loss = 0.0
                epoch_acc = 0
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()
                for batch_idx, (data, labels) in enumerate(self.dataloaders[phase]):
                    data, labels = data.to(self.device), labels.to(self.device, dtype=torch.long)
                    with torch.set_grad_enabled(phase == 'train'):
                        logits = self.model(data)
                        loss = self.criterion(logits, labels)
                        pred = logits.argmax(dim=1)
                        correct_num = torch.eq(pred, labels).float().sum().item()
                        loss_temp = loss.item() * data.size(0)  # criterion返回的 loss 是每个 batch 所有样本的平均值
                        epoch_loss += loss_temp
                        epoch_acc += correct_num

                        # 训练集,模型更新
                        if phase == 'train':
                            self.optimizer.zero_grad()  # 梯度清零
                            loss.backward()  # 梯度计算与反向传播
                            self.optimizer.step()  # 参数更新
                        else:
                            pass

                if phase == 'train':
                    epoch_train_loss = epoch_loss / len(self.dataloaders[phase].dataset)
                    epoch_train_acc = epoch_acc / len(self.dataloaders[phase].dataset)
                    trainLoss.append(epoch_train_loss)
                    trainAcc.append(epoch_train_acc)
                    print('{} epoch, 训练Loss: {:.8f}, 训练精度: {:.4%}  >> [{}/{}]'.format(
                        epoch, epoch_train_loss, epoch_train_acc, int(epoch_acc), len(self.dataloaders[phase].dataset)))
                else:  # phase == 'val'
                    epoch_val_loss = epoch_loss / len(self.dataloaders[phase].dataset)
                    epoch_val_acc = epoch_acc / len(self.dataloaders[phase].dataset)
                    valLoss.append(epoch_val_loss)
                    valAcc.append(epoch_val_acc)
                    print('{} epoch, 验证Loss: {:.8f}, 验证精度: {:.4%}  >> [{}/{}]'.format(
                        epoch, epoch_val_loss, epoch_val_acc, int(epoch_acc), len(self.dataloaders[phase].dataset)))

                    # get the best model and best val accuracy according to the validation results
                    model_state_dic = self.model.state_dict()
                    if epoch_val_acc >= best_val_acc:
                        best_val_acc = epoch_val_acc
                        best_model_state_dic = model_state_dic
                        train_time = time.time() - start_time
                        print('最佳 epoch {}, 最佳验证精度 {:.4%}, 训练用时：{:.4f} sec'.format(epoch, best_val_acc, train_time))
                        os.makedirs(self.result_file_save_path) if os.path.exists(self.result_file_save_path) == False else None

            # Judge model convergence
            if epoch >= 2 and trainAcc[-1] <= trainAcc[-2]:
                n_step += 1
            else:
                n_step = 0
            if epoch_train_loss < 0.001 and n_step >= 15:
                break
        torch.save(best_model_state_dic, os.path.join(self.result_file_save_path, self.model_name + '_best_model.pth'))
        return trainLoss, trainAcc, valLoss, valAcc, best_val_acc, train_time

    def my_tester(self):
        model_save_path = os.path.join(self.result_file_save_path, self.model_name + '_best_model.pth')
        self.model.load_state_dict(torch.load(model_save_path))
        self.model.eval()
        LOGs, LABs = torch.tensor([]), torch.tensor([])  # 特征及对应标签
        start_time = time.time()
        test_acc = 0
        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(self.dataloaders['test']):
                data, labels = data.to(self.device), labels.to(self.device)
                logits = self.model(data)
                # test_loss = self.criterion(logits, labels)
                pred = logits.argmax(dim=1)
                correct = torch.eq(pred, labels).float().sum().item()
                test_acc += correct
                LOGs = torch.cat((LOGs, logits.detach().cpu()), dim=0)
                LABs = torch.cat((LABs, labels.unsqueeze(dim=-1).cpu()), dim=0)
        test_acc = test_acc / len(self.dataloaders['test'].dataset)
        test_time = time.time() - start_time
        print('=========== 测试精度: {:.4%}, 测试时间: {:.4f} sec ==============='.format(test_acc, test_time))

        return LOGs.numpy(), LABs.numpy(), test_acc


if __name__ == '__main__':
    Iterator = Iterator_utils()
    Iterator.setup()
    trainLoss, trainAcc, valLoss, valAcc, best_val_acc, train_time = Iterator.my_trainer()
    LOGs, LABs, test_acc = Iterator.my_tester()
    if dataset_name == 'CQU_gear_dataset':
        typeName = ['He', 'RC', 'SS', 'SP', 'BT']
    else:  # dataset_name == 'SQ_bearing_dataset'
        typeName = ['No', 'OF1', 'O2F', 'OF3', 'IF1', 'IF2', 'IF3']
    plotLossAcc(trainLoss, valLoss, trainAcc, valAcc, result_file_save_path, '收敛曲线')
    # plotConfusionMatrix(torch.from_numpy(LOGs).argmax(dim=-1), LABs, typeName, result_file_save_path, '混淆矩阵')
    # plotTSNECluster(LOGs, LABs, len(typeName), typeName, result_file_save_path, '聚类图')

    try_times = 20
    try_train_Time = []
    try_test_Acc = []
    for ii in range(try_times):
        Iterator = Iterator_utils()
        Iterator.setup()
        trainLoss, trainAcc, valLoss, valAcc, best_val_acc, train_time = Iterator.my_trainer()
        logits, labels, test_acc = Iterator.my_tester()
        try_train_Time.append(train_time)
        try_test_Acc.append(test_acc)
    mean_test_acc = np.mean(try_test_Acc)
    std_test_acc = np.std(try_test_Acc)
    try_test_Acc.append(mean_test_acc)
    try_test_Acc.append(std_test_acc)
    np.savetxt(result_file_save_path + '/' + 'try_test_Acc.csv', try_test_Acc, fmt="%.8f")
    print('循环', try_times, '次，平均测试精度：{:.4%}, 标注差：{:.4%}'.format(mean_test_acc, std_test_acc))


