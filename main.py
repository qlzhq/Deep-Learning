import numpy as np
import pickle
import os
from collections import OrderedDict
from common import SoftmaxWithLoss, Relu, Adam, Affine, Convolution, Pooling, Dropout, SGD
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import recall_score, mean_squared_error


# 简单的 ConvNet
class SimpleConvNet:
    def __init__(self, input_dim=(3, 32, 32),
                 conv_param={'filter_num': 32, 'filter_size': 3, 'pad': 1, 'stride': 1},
                 hidden_size=128, output_size=2, weight_init_std=0.01):
        """
        初始化 SimpleConvNet 的网络结构。

        参数:
        - input_dim (tuple): 输入数据的维度，格式为 (通道数, 高度, 宽度)。默认为 (3, 32, 32)。
        - conv_param (dict): 卷积层的参数，包含以下键值对：
            - 'filter_num' (int): 卷积核的数量。默认为 32。
            - 'filter_size' (int): 卷积核的大小。默认为 3。
            - 'pad' (int): 填充的像素数。默认为 1。
            - 'stride' (int): 卷积的步长。默认为 1。
        - hidden_size (int): 第一个全连接层的神经元数量。默认为 128。
        - output_size (int): 输出层的神经元数量，即分类的类别数。默认为 2。
        - weight_init_std (float): 权重初始化的标准差。默认为 0.01。
        """

        # 从 conv_param 中提取卷积层的参数
        filter_num = conv_param['filter_num']  # 卷积核的数量
        filter_size = conv_param['filter_size']  # 卷积核的大小
        filter_pad = conv_param['pad']  # 填充的像素数
        filter_stride = conv_param['stride']  # 卷积的步长

        # 计算第一个卷积层的输出尺寸
        conv_output_size1 = (input_dim[1] - filter_size + 2 * filter_pad) // filter_stride + 1
        # 计算第一个池化层的输出尺寸
        pool_output_size1 = conv_output_size1 // 2

        # 计算第二个卷积层的输出尺寸
        conv_output_size2 = (pool_output_size1 - filter_size + 2 * filter_pad) // filter_stride + 1
        # 计算第二个池化层的输出尺寸
        pool_output_size2 = conv_output_size2 // 2

        # 计算第一个全连接层的输入维度
        affine1_input_size = filter_num * pool_output_size2 * pool_output_size2

        # 初始化权重
        self.params = {}
        # # 第一个卷积层的权重和偏置
        # self.params['W1'] = weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        # self.params['b1'] = np.zeros(filter_num)
        # # 第二个卷积层的权重和偏置
        # self.params['W2'] = weight_init_std * np.random.randn(filter_num, filter_num, filter_size, filter_size)
        # self.params['b2'] = np.zeros(filter_num)
        # # 第一个全连接层的权重和偏置
        # self.params['W3'] = weight_init_std * np.random.randn(affine1_input_size, hidden_size)
        # self.params['b3'] = np.zeros(hidden_size)
        # # 第二个全连接层的权重和偏置
        # self.params['W4'] = weight_init_std * np.random.randn(hidden_size, output_size)
        # self.params['b4'] = np.zeros(output_size)

        # He初始化
        # 第一个卷积层的权重和偏置
        fan_in1 = input_dim[0] * filter_size * filter_size  # 输入神经元数量
        self.params['W1'] = np.sqrt(2 / fan_in1) * np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)

        # 第二个卷积层的权重和偏置
        fan_in2 = filter_num * filter_size * filter_size  # 输入神经元数量
        self.params['W2'] = np.sqrt(2 / fan_in2) * np.random.randn(filter_num, filter_num, filter_size, filter_size)
        self.params['b2'] = np.zeros(filter_num)

        # 第一个全连接层的权重和偏置
        fan_in3 = affine1_input_size  # 输入神经元数量
        self.params['W3'] = np.sqrt(2 / fan_in3) * np.random.randn(affine1_input_size, hidden_size)
        self.params['b3'] = np.zeros(hidden_size)

        # 第二个全连接层的权重和偏置
        fan_in4 = hidden_size  # 输入神经元数量
        self.params['W4'] = np.sqrt(2 / fan_in4) * np.random.randn(hidden_size, output_size)
        self.params['b4'] = np.zeros(output_size)

        # 生成网络层
        self.layers = OrderedDict()
        # 第一个卷积层
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'], conv_param['stride'],
                                           conv_param['pad'])
        # 第一个 ReLU 激活层
        self.layers['Relu1'] = Relu()
        # 第一个池化层
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)

        # 第二个卷积层
        self.layers['Conv2'] = Convolution(self.params['W2'], self.params['b2'], conv_param['stride'],
                                           conv_param['pad'])
        # 第二个 ReLU 激活层
        self.layers['Relu2'] = Relu()
        # 第二个池化层
        self.layers['Pool2'] = Pooling(pool_h=2, pool_w=2, stride=2)
        # 第一个全连接层
        self.layers['Affine1'] = Affine(self.params['W3'], self.params['b3'])
        # 第三个 ReLU 激活层
        self.layers['Relu3'] = Relu()

        # 第二个全连接层
        self.layers['Affine2'] = Affine(self.params['W4'], self.params['b4'])

        # 输出层
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=32):
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        acc = 0.0
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i * batch_size:(i + 1) * batch_size]
            tt = t[i * batch_size:(i + 1) * batch_size]
            y = self.predict(tx)
            y_pred = np.argmax(y, axis=1)
            acc += np.sum(y_pred == tt)
        return acc / x.shape[0]

    def gradient(self, x, t):
        self.loss(x, t)
        dout = 1
        dout = self.last_layer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        grads['W2'], grads['b2'] = self.layers['Conv2'].dW, self.layers['Conv2'].db
        grads['W3'], grads['b3'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W4'], grads['b4'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
        return grads

    def save_params(self, file_name="catdog_cnn_params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="catdog_cnn_params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val
        for i, key in enumerate(['Conv1', 'Conv2', 'Affine1', 'Affine2']):
            self.layers[key].W = self.params['W' + str(i + 1)]
            self.layers[key].b = self.params['b' + str(i + 1)]


# 加载猫狗数据集
# def load_catdog():
#     data = []
#     labels = []
#     label_dict = {"cat": 0, "dog": 1}
#
#     for split in ["training_set", "test_set", "val_set"]:
#         folder_path = f'./catdog/{split}'
#         for label in ["cat", "dog"]:
#             class_path = os.path.join(folder_path, label)
#             for img_name in os.listdir(class_path):
#                 img_path = os.path.join(class_path, img_name)
#                 img = Image.open(img_path).convert('RGB')
#                 img = img.resize((32, 32))
#                 img = np.array(img).astype(np.float32) / 255.0
#                 img = np.transpose(img, (2, 0, 1))
#                 data.append(img)
#                 labels.append(label_dict[label])
#
#     data = np.array(data)
#     labels = np.array(labels)
#
#     train_size = int(0.6 * len(data))
#     val_size = int(0.2 * len(data))
#     test_size = len(data) - train_size - val_size
#     x_train, x_val, x_test = data[:train_size], data[train_size:train_size + val_size], data[train_size + val_size:]
#     t_train, t_val, t_test = labels[:train_size], labels[train_size:train_size + val_size], labels[
#                                                                                             train_size + val_size:]
#
#     return (x_train, t_train), (x_val, t_val), (x_test, t_test)
# 随机裁剪
def random_crop(img, crop_size):
    h, w = img.shape[1:]
    crop_h, crop_w = crop_size
    top = np.random.randint(0, h - crop_h)
    left = np.random.randint(0, w - crop_w)
    return img[:, top:top + crop_h, left:left + crop_w]


# 随机水平翻转
def random_horizontal_flip(img):
    if np.random.rand() < 0.5:
        return np.flip(img, axis=2).copy()
    return img


def load_catdog():
    data = []
    labels = []
    label_dict = {"cat": 0, "dog": 1}

    for split in ["training_set", "test_set", "val_set"]:
        folder_path = f'./catdog/{split}'
        for label in ["cat", "dog"]:
            class_path = os.path.join(folder_path, label)
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                img = Image.open(img_path).convert('RGB')
                img = img.resize((36, 36))  # 先放大一点，以便后续裁剪
                img = np.array(img).astype(np.float32) / 255.0
                img = np.transpose(img, (2, 0, 1))

                if split == "training_set":
                    img = random_crop(img, (32, 32))
                    img = random_horizontal_flip(img)
                else:
                    img = img[:, 2:34, 2:34]  # 测试集和验证集取中心裁剪

                data.append(img)
                labels.append(label_dict[label])

    data = np.array(data)
    labels = np.array(labels)

    train_size = int(0.6 * len(data))
    val_size = int(0.2 * len(data))
    test_size = len(data) - train_size - val_size
    x_train, x_val, x_test = data[:train_size], data[train_size:train_size + val_size], data[train_size + val_size:]
    t_train, t_val, t_test = labels[:train_size], labels[train_size:train_size + val_size], labels[
                                                                                            train_size + val_size:]

    return (x_train, t_train), (x_val, t_val), (x_test, t_test)


# 将标签转换为 one-hot 编码
def to_one_hot(labels, num_classes=2):
    return np.eye(num_classes)[labels]


# 评估函数
def evaluate(file_name="catdog_cnn_params.pkl"):
    print(f"Load {file_name}")
    (_, _), (x_val, t_val), (x_test, t_test) = load_catdog()
    t_val_one_hot = to_one_hot(t_val)
    t_test_one_hot = to_one_hot(t_test)

    network = SimpleConvNet(input_dim=(3, 32, 32),
                            conv_param={'filter_num': 32, 'filter_size': 3, 'pad': 1, 'stride': 1},
                            hidden_size=128, output_size=2, weight_init_std=0.01)
    network.load_params(file_name)

    val_loss = network.loss(x_val, t_val_one_hot)
    val_acc = network.accuracy(x_val, t_val_one_hot)
    test_loss = network.loss(x_test, t_test_one_hot)
    test_acc = network.accuracy(x_test, t_test_one_hot)

    y_pred = np.argmax(network.predict(x_test), axis=1)
    recall = recall_score(t_test, y_pred)

    y_pred_prob = network.predict(x_test)
    rmse = np.sqrt(mean_squared_error(t_test_one_hot, y_pred_prob))

    # print(f"Validation Loss: {val_loss:.4f}")
    # print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"RMSE: {rmse:.4f}")

    return val_loss, val_acc, test_loss, test_acc, recall, rmse


# 训练函数
def train(epochs=20, batch_size=32, learning_rate=0.001):
    (x_train, t_train), (x_val, t_val), (x_test, t_test) = load_catdog()
    t_train = to_one_hot(t_train)
    t_val = to_one_hot(t_val)
    t_test = to_one_hot(t_test)

    network = SimpleConvNet(input_dim=(3, 32, 32),
                            conv_param={'filter_num': 32, 'filter_size': 3, 'pad': 1, 'stride': 1},
                            hidden_size=128, output_size=2, weight_init_std=0.01)

    train_size = x_train.shape[0]

    train_loss_list = []
    train_acc_list = []
    val_acc_list = []
    val_loss_list = []
    test_acc_list = []

    iter_per_epoch = max(train_size // batch_size, 1)
    iters_num = int(epochs * iter_per_epoch)

    for i in range(iters_num):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        grad = network.gradient(x_batch, t_batch)
        optimizer = Adam(lr=learning_rate)
        # optimizer = SGD(lr=learning_rate)
        optimizer.update(network.params, grad)

        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)

        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train, batch_size=batch_size)
            val_acc = network.accuracy(x_val, t_val, batch_size=batch_size)
            test_acc = network.accuracy(x_test, t_test, batch_size=batch_size)
            train_loss = network.loss(x_train, t_train)
            val_loss = network.loss(x_val, t_val)
            train_acc_list.append(train_acc)
            val_acc_list.append(val_acc)
            val_loss_list.append(val_loss)
            train_loss_list.append(train_loss)
            test_acc_list.append(test_acc)
            print(f"Epoch {i // iter_per_epoch + 1}: Train Acc = {train_acc:.4f}, Train Loss = {train_loss:.4f}"
                  f" Val Acc = {val_acc:.4f}, Val Loss = {val_loss:.4f}"
                  f" Test Acc = {test_acc:.4f}")

    network.save_params("catdog_cnn_params.pkl")
    print("Model parameters saved to catdog_cnn_params.pkl")

    plt.figure(figsize=(15, 5))
    plt.subplot(2, 2, 1)
    plt.plot(np.arange(len(train_loss_list)), train_loss_list, label='Train Loss')
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    # plt.savefig('CatDog_CNN_Training_Loss.png', dpi=300)
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(np.arange(len(train_acc_list)), train_acc_list, label='Train Acc', marker='o')
    plt.plot(np.arange(len(val_acc_list)), val_acc_list, label='Val Acc', linestyle='--', marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    # plt.savefig('CatDog_CNN_Training_With_Val_Acc.png', dpi=300)
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(np.arange(len(val_loss_list)), val_loss_list, label='Val Loss', marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Loss")
    plt.legend()
    plt.tight_layout()
    # plt.savefig('CatDog_CNN_Training_With_Val.png', dpi=300)

    x = np.arange(len(train_acc_list))
    plt.subplot(2, 2, 4)
    plt.plot(x, train_acc_list, label='train acc', marker='o')
    plt.plot(x, test_acc_list, label='test acc', linestyle='--', marker='s')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')
    plt.title('CatDog CNN Training')
    plt.savefig('CatDog_CNN.png', dpi=300)
    plt.close()

    return network


if __name__ == "__main__":
    epochs = 20
    batch_size = 32
    learning_rate = 0.001
    train(epochs=20, batch_size=32, learning_rate=0.001)
    val_loss, val_acc, test_loss, test_acc, recall, rmse = \
        evaluate(file_name="catdog_cnn_params.pkl")
