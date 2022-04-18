# 课设项目：基于GoogLeNet的知识蒸馏实验
## 代码运行方法
* 1.先将flower数据集下载解压到data_set/flower_data文件夹下，具体方法见data_set的README文件
* 2.执行split_data.py
* 3.其余代码均在code文件夹内，train是训练代码，分别对教师网络、蒸馏网络、学生网络进行训练
* 4.训练好模型后，可用predict进行预测，预测使用图像是code同一目录下的test.jpg图片，可将其删除，使用自己的图片，注意名称格式不能变
* 5.另外提供visual文件对知识蒸馏温度T和训练过程进行可视化
