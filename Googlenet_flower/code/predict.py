import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import GoogLeNet,Studentmodel


def predict():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # load image
    img_path = "../test.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)
 
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)


    th_model = GoogLeNet(num_classes=5, aux_logits=False).to(device)
    dk_model = Studentmodel().to(device)
    st_model = Studentmodel().to(device)

    
    weights_path1 = "./googleNet.pth"##训练好的模型参数保存位置
    assert os.path.exists(weights_path1), "file: '{}' dose not exist.".format(weights_path1)
    th_model.load_state_dict(torch.load(weights_path1, map_location=device),
                                                          strict=False)
    weights_path2 = "./googleDKNet.pth"  ##训练好的模型参数保存位置
    assert os.path.exists(weights_path2), "file: '{}' dose not exist.".format(weights_path2)
    dk_model.load_state_dict(torch.load(weights_path2, map_location=device),
                            )
    weights_path3 = "./studentNet.pth"  ##训练好的模型参数保存位置
    assert os.path.exists(weights_path3), "file: '{}' dose not exist.".format(weights_path3)
    st_model.load_state_dict(torch.load(weights_path3, map_location=device),
                                                          strict=False)
    st_model.eval()
    dk_model.eval()
    th_model.eval()

    with torch.no_grad():
        # predict class
        output_tc = torch.squeeze(th_model(img.to(device))).cpu()
        predict_tc = torch.softmax(output_tc, dim=0)
        predict_cla1 = torch.argmax(predict_tc).numpy()
#     with torch.no_grad():
        # predict class
        output_dk = torch.squeeze(dk_model(img.to(device))).cpu()
        predict_dk = torch.softmax(output_dk, dim=0)
        predict_cla2 = torch.argmax(predict_dk).numpy()
#     with torch.no_grad():
        # predict class
        output_st = torch.squeeze(st_model(img.to(device))).cpu()
        predict_st = torch.softmax(output_st, dim=0)
        predict_cla3 = torch.argmax(predict_st).numpy()
    # print_res = " class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
    #                                              predict[predict_cla].numpy())
    print_res = " predict: Teacher:{},DK:{},Student:{}  ".format(class_indict[str(predict_cla1)],
                                                                 class_indict[str(predict_cla2)],
                                                                class_indict[str(predict_cla3)])
    
    plt.title(print_res)
    plt.show()
    for i in range(len(predict_tc)):
        print(" Teachermodel     class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict_tc[i].numpy()))
    for i in range(len(predict_dk)):
        print("DKmodel     class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict_dk[i].numpy()))
    for i in range(len(predict_st)):
        print("Studentmodel     class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict_st[i].numpy()))


if __name__ == '__main__':
    predict()