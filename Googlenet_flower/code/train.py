import os
import sys
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm

from model import GoogLeNet,Studentmodel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))
#将图像随机裁剪为224X224大小
#以0.5的概率水平翻转
#将RGB三个通道值标准化为[-1,1]区间
data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "val": transforms.Compose([transforms.Resize((224, 224)),###图像大小为224X224
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  #
image_path = os.path.join(data_root, "Googlenet_flower/data_set", "flower_data")  
assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                     transform=data_transform["train"])
train_num = len(train_dataset)

#获取类别名，并以daisy:0, dandelion:1, roses:2, sunflower:3, tulips:4的形式写入到json文件中
flower_list = train_dataset.class_to_idx
cla_dict = dict((val, key) for key, val in flower_list.items())

json_str = json.dumps(cla_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)
#每次训练32个样本
batch_size = 32
# nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  
# print('Using {} dataloader workers every process'.format(nw))

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           )#每个epoch开始时，对数据重新排序

validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                        transform=data_transform["val"])
val_num = len(validate_dataset)
validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                              batch_size=batch_size, shuffle=False,
                                              )

print("using {} images for training, {} images for validation.".format(train_num,
                                                                 val_num))
def train_teacher(loss_logits_wt=1,loss_aux_logits2_wt=0.3,loss_aux_logits1_wt=0.3):
    #需要两个辅助分类器  初始化权重
    model = GoogLeNet(num_classes=5, aux_logits=True, init_weights=True)
    model.to(device)
    #损失函数CrossEntropyLoss
    #优化器Adm，学习率0.0003
    #30个epoch
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    epochs = 30
    best_acc = 0.0
    save_path = './googleNet.pth'##保存模型参数位置
    train_steps = len(train_loader)
    for epoch in range(epochs):
        #训练 self.training=True
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits, aux_logits2, aux_logits1 = net(images.to(device))
            loss0 = loss_function(logits, labels.to(device))
            loss1 = loss_function(aux_logits1, labels.to(device))
            loss2 = loss_function(aux_logits2, labels.to(device))
            loss = loss0*loss_logits_wt + loss1 *loss_aux_logits1_wt + loss2 * loss_aux_logits2_wt#总loss
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,)
        #测试  self.training=False 不再使用辅助分类器，只有一个输出                                                          loss)
        model.eval()
        acc = 0.0  
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = model(val_images.to(device))  
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
        val_accurate = acc / val_num#计算正确率
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(model.state_dict(), save_path)

    print('Teacher model training completed')
    
def train_dk(temp=5,hard_loss_wt=0.55,soft_loss_wt=0.45,loss_logits_wt=1,loss_aux_logits2_wt=0.3,loss_aux_logits1_wt=0.3):
    teacher_model = GoogLeNet(num_classes=5, aux_logits=True).to(device)
    weights_path = "./googleNet.pth"  ##训练好的模型参数保存位置
    ####导入训练好的教师模型
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    teacher_model.load_state_dict(torch.load(weights_path, map_location=device),
                          strict=False)

    student_model=Studentmodel().to(device)
    #hardloss采用交叉熵CrossEntropyLoss，softloss采用相对熵KL散度KLDivLoss，二者作用原理相似
    #优化器Adm，学习率0.0001
    #30个epoch
    student_loss_fn = nn.CrossEntropyLoss()
    divergence_loss_fn = nn.KLDivLoss(reduction="batchmean")
    optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-4)

    teacher_model.train()#teacher_model不需要训练，由于要用到辅助分类器输出结果，此句仅为了将self.training置为true
    student_model.train()

    epochs = 30
    best_acc = 0.0
    save_path = './googleDKNet.pth'##保存模型参数位置
    train_steps = len(train_loader)
    for epoch in range(epochs):
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            images=images.to(device)
            labels=labels.to(device)
            with torch.no_grad():
                teacher_preds,teacher_preds_aux2,teacher_preds_aux1 = teacher_model(images)

            student_preds=student_model(images)
            student_loss=student_loss_fn(student_preds,labels)

            ##蒸馏温度=5  学生网络与教师网络的loss等于与教师网络三个输出（两个辅助分类器）的loss加权和
            dist_loss0 = divergence_loss_fn(F.softmax(student_preds / temp, dim=1),
                                                    F.softmax(teacher_preds / temp, dim=1))
            dist_loss1 = divergence_loss_fn(F.softmax(student_preds / temp, dim=1),
                                                    F.softmax(teacher_preds_aux1 / temp, dim=1))
            dist_loss2 = divergence_loss_fn(F.softmax(student_preds / temp, dim=1),
                                                    F.softmax(teacher_preds_aux2 / temp, dim=1))

            distillation_loss=loss_logits_wt*dist_loss0+loss_aux_logits1_wt*dist_loss1+loss_aux_logits2_wt*dist_loss2
            
            total_loss=student_loss*hard_loss_wt+distillation_loss*soft_loss_wt

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     total_loss)
        student_model.eval()
        acc_num = 0.0  
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = student_model(val_images.to(device))  
                predict_y = torch.max(outputs, dim=1)[1]
                acc_num += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc_num / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(student_model.state_dict(), save_path)

    print('DK model training completed')

    
def train_student():
    student_model = Studentmodel().to(device)
    #损失函数CrossEntropyLoss
    #优化器Adm，学习率0.0001
    #30个epoch
    student_loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-4)
    student_model.train()

    epochs = 30
    best_acc = 0.0
    save_path = './studentNet.pth'  ##保存模型参数位置
    train_steps = len(train_loader)
    for epoch in range(epochs):
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            student_preds = student_model(images)
            loss = student_loss_fn(student_preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)
        student_model.eval()
        acc = 0.0  
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = student_model(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(student_model.state_dict(), save_path)

    print('Student model training completed')

if __name__ == '__main__':
    train_teacher()
    train_dk()
    train_student()