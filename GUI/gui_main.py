import sys, os
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image, ImageQt
from BPNN import BPNetwork
import gzip,pickle

from qt.layout import Ui_MainWindow
from qt.paintboard import PaintBoard

from PyQt5.QtWidgets import QMainWindow, QDesktopWidget, QApplication
from PyQt5.QtWidgets import QLabel, QMessageBox, QPushButton, QFrame
from PyQt5.QtGui import QPainter, QPen, QPixmap, QColor, QImage
from PyQt5.QtCore import Qt, QPoint, QSize

# from simple_convnet import SimpleConvNet
# from common.functions import softmax
# from deep_convnet import DeepConvNet

MODE_MNIST = 1  # MNIST随机抽取
MODE_WRITE = 2  # 手写输入

Thresh = 0.5  # 识别结果置信度阈值

# 读取MNIST数据集
(_, _), (x_test, _) = load_mnist(normalize=True, flatten=False, one_hot_label=False)

# 初始化网络

# 网络：BP network网络
f = gzip.open('mnist.pkl.gz', 'rb')
tset, vset, teset = pickle.load(f, encoding='latin1')
f.close()

# Just use the first 9000 images for training
tread = 9000
train_in = tset[0][:tread, :]
# This is a little bit of work -- 1 of N encoding
# Make sure you understand how it does it
train_tgt = np.zeros((tread, 10))
for i in range(tread):
    train_tgt[i, tset[1][i]] = 1
sizes = [784, 32, 64, 10]
network = BPNetwork.MLP(sizes)
network.train(train_in, train_tgt, 0.01, 1000)

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        # 初始化参数
        self.mode = MODE_MNIST
        self.result = [0, 0]

        # 初始化UI
        self.setupUi(self)
        self.center()

        # 初始化画板
        self.paintBoard = PaintBoard(self, Size=QSize(224, 224), Fill=QColor(0, 0, 0, 0))
        self.paintBoard.setPenColor(QColor(0, 0, 0, 0))
        self.dArea_Layout.addWidget(self.paintBoard)

        self.clearDataArea()

    # 窗口居中
    def center(self):
        # 获得窗口
        framePos = self.frameGeometry()
        # 获得屏幕中心点
        scPos = QDesktopWidget().availableGeometry().center()
        # 显示到屏幕中心
        framePos.moveCenter(scPos)
        self.move(framePos.topLeft())


    # 清除数据待输入区
    def clearDataArea(self):
        self.paintBoard.Clear()
        self.lbDataArea.clear()
        self.lbResult.clear()
        # self.lbCofidence.clear()
        self.result = [0, 0]

    # 模式下拉列表回调
    def cbBox_Mode_Callback(self, text):
        if text == '1：Random image from MNIST':
            self.mode = MODE_MNIST
            self.clearDataArea()
            self.pbtGetMnist.setEnabled(True)

            self.paintBoard.setBoardFill(QColor(0, 0, 0, 0))
            self.paintBoard.setPenColor(QColor(0, 0, 0, 0))

        elif text == '2：Write digit by mouse':
            self.mode = MODE_WRITE
            self.clearDataArea()
            self.pbtGetMnist.setEnabled(False)

            # 更改背景
            self.paintBoard.setBoardFill(QColor(0, 0, 0, 255))
            self.paintBoard.setPenColor(QColor(255, 255, 255, 255))

    # 数据清除
    def pbtClear_Callback(self):
        self.clearDataArea()

    # 识别
    def pbtPredict_Callback(self):
        __img, img_array = [], []  # 将图像统一从qimage->pil image -> np.array [1, 1, 28, 28]

        # 获取qimage格式图像
        if self.mode == MODE_MNIST:
            __img = self.lbDataArea.pixmap()  # label内若无图像返回None
            if __img == None:  # 无图像则用纯黑代替
                # __img = QImage(224, 224, QImage.Format_Grayscale8)
                __img = ImageQt.ImageQt(Image.fromarray(np.uint8(np.zeros([224, 224]))))
            else:
                __img = __img.toImage()
        elif self.mode == MODE_WRITE:
            __img = self.paintBoard.getContentAsQImage()

        # 转换成pil image类型处理
        pil_img = ImageQt.fromqimage(__img)
        pil_img = pil_img.resize((28, 28), Image.ANTIALIAS)

        img_array = np.array(pil_img.convert('L')).reshape(784,) / 255.0

        self.result[0] = network.test_single(img_array)  # 预测的数字
        # self.result[1] = 1  # 置信度

        self.lbResult.setText("%d" % (self.result[0]))
        # self.lbCofidence.setText("%.8f" % (self.result[1]))

    # 随机抽取
    def pbtGetMnist_Callback(self):
        self.clearDataArea()

        # 随机抽取一张测试集图片，放大后显示
        img = x_test[np.random.randint(0, 9999)]  # shape:[1,28,28]
        img = img.reshape(28, 28)  # shape:[28,28]

        img = img * 0xff  # 恢复灰度值大小
        pil_img = Image.fromarray(np.uint8(img))
        pil_img = pil_img.resize((224, 224))  # 图像放大显示

        # 将pil图像转换成qimage类型
        qimage = ImageQt.ImageQt(pil_img)

        # 将qimage类型图像显示在label
        pix = QPixmap.fromImage(qimage)
        self.lbDataArea.setPixmap(pix)

 # 窗口关闭事件
    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Message',
                                     "Quit the system?", QMessageBox.Yes |
                                     QMessageBox.No, QMessageBox.Yes)

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    Gui = MainWindow()
    Gui.show()

    sys.exit(app.exec_())