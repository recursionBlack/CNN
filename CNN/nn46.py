import torch
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torch.utils.data as data_utils
# import cv2

# 手写数字识别
# data
# 导数数据集, 已经被集成进torchvision.datasets里了
# 存放到本地的mnist这个文件夹下
train_data = dataset.MNIST(root="mnist",
                          train=True,
                          transform=transforms.ToTensor(),
                          download=True)
test_data = dataset.MNIST(root="mnist",
                          train=False,
                          transform=transforms.ToTensor(),
                          download=False)
# batchsize
# 由于数据总数可能很大，所有有时候，要一点点的训练，不要一次性加载全部
# 每次取出batchsize的数量
train_loader = data_utils.DataLoader(dataset=train_data,
                                     batch_size=64,
                                     shuffle=True)

test_loader = data_utils.DataLoader(dataset=test_data,
                                     batch_size=64,
                                     shuffle=True)
# net
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 卷积层，卷积操作
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        # 线性层，FC操作....0-9,个数字，是10维的
        self.fc = torch.nn.Linear(14 * 14 * 32, 10)

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        return out

cnn = CNN()
cnn = cnn.cuda()    # 从cpu转到GPU上
# loss
loss_func = torch.nn.CrossEntropyLoss()

# optimizer
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.01)

# trainning
# 整个样本过10遍
for epoch in range(10):
    for i, (images, labels) in enumerate(train_loader):
        # 从cpu转到GPU上
        images = images.cuda()
        labels = labels.cuda()

        outputs = cnn(images)
        loss = loss_func(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("epoch is {}, ite is {}/{}, loss is {}". format(epoch+1, i,
                                                        len(train_data) // 64,
                                                        loss.item()))

    # eval/test
    loss_test = 0
    accuracy = 0

    for i, (images, labels) in enumerate(test_loader):
        # 从cpu转到GPU上
        images = images.cuda()
        labels = labels.cuda()
        outputs = cnn(images)
        # [batchsize]
        # outputs = batchsize * cls_num     # cls_num = 10个数字
        loss_test += loss_func(outputs, labels)
        _, pred = outputs.max(1)
        accuracy += (pred == labels).sum().item()

        # 放回到cpu上并显示
        images = images.cpu().numpy()
        labels = labels.cpu().numpy()
        pred = pred.cpu().numpy()

        # batchsize * 1 * 28 * 28
        for idx in range(images.shape[0]):
            im_data = images[idx]
            im_label = labels[idx]
            im_pred = pred[idx]
            # 把通道移动到最后面
            # im_data = im_data.transpose(1, 2, 0)
            print("label", im_label)
            print("pred", im_pred)
            """
            libtiff.so.6: undefined symbol: jpeg12_write_raw_data, version LIBJPEG_8.0
            这是一条在程序运行过程中出现的错误信息。
            “libtiff.so.6”指的是一个名为libtiff的共享库文件，版本号为6。“undefined symbol”表示未定义的符号，
            这里具体指的是“jpeg12_write_raw_data”这个符号，它在程序中被引用，但却没有找到对应的定义。
            “version LIBJPEG_8.0”说明程序期望这个符号来自版本号为8.0的LIBJPEG库。
            整体意思是，在运行与libtiff.so.6相关的程序时，它试图调用LIBJPEG_8.0库中的jpeg12_write_raw_data函数，
            但在该库中却找不到这个函数的定义，从而导致错误。
            例如，在一个使用libtiff库进行图像文件处理，且依赖于特定版本JPEG库函数的程序中，
            就可能出现这种因为库函数符号未定义而报错的情况。 
            """
            # cv2.imshow("imdata", im_data)
            # cv2.waitKey(0)

    accuracy = accuracy / len(test_data)
    loss_test = loss_test / (len(test_data) // 64)
    # 打印出精度和损失
    print("epoch is {}, accuracy is {}, "
          "loss test is {}".format(epoch + 1,
                                   accuracy, loss_test.item()))
# save
torch.save(cnn, "model/mnist_model.pkl")

# load
