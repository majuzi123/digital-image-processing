from django.shortcuts import render,HttpResponse
import numpy as np
from io import BytesIO
import cv2
import sys
import matplotlib.pyplot as plt
import numpy
# Create your views here.
def home(request):

    return render(request,'home.html') #render用来打开html
def login(request):
   if request.method=='GET':
        return render(request,'login.html')
   else:
     uname=request.POST.get('username')
     pwd = request.POST.get('password')
     if uname=='root' and pwd=='123':
        return render(request, 'home.html')
     else:
         return HttpResponse('用户名或密码错误！')

'''def login2(request):

    return render(request,'home.html')'''
def photo(request):
    file_object=request.FILES.get("pic")
    print(file_object.name)
    f=open('a1.jpg',mode='wb')
    for chunk in file_object.chunks():
        f.write(chunk)
    f.close()
    return render(request,'function.html')
def index(request):

    return render(request,'index.html')
def f1_b(request):
    img = cv2.imread('./a1.jpg')
    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]
    cv2.imwrite('./static/abc.jpg', b)
    return render(request,'index.html',{'image':'/static/abc.jpg'})
def f1_g(request):
    img = cv2.imread('./a1.jpg')
    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]
    cv2.imwrite('./static/abc.jpg', g)
    return render(request,'index.html',{'image':'/static/abc.jpg'})
def f1_r(request):
    img = cv2.imread('./a1.jpg')
    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]
    cv2.imwrite('./static/abc.jpg', r)
    return render(request,'index.html',{'image':'/static/abc.jpg'})
def f2_h(request):
    img = cv2.imread('./a1.jpg')
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    cv2.imwrite('./static/abc.jpg', h)
    return render(request,'index.html',{'image':'/static/abc.jpg'})
def f2_s(request):
    img = cv2.imread('./a1.jpg')
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    cv2.imwrite('./static/abc.jpg', s)
    return render(request,'index.html',{'image':'/static/abc.jpg'})
def f2_v(request):
    img = cv2.imread('./a1.jpg')
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    cv2.imwrite('./static/abc.jpg', v)
    return render(request,'index.html',{'image':'/static/abc.jpg'})
def f3(request):
    img = cv2.imread('./a1.jpg')
    # l, w, h = img.shape
    # 放大图像至原来的两倍，使用双线性插值法
    img = cv2.resize(img, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    height, width, channel = img.shape

    # 构建移动矩阵,x轴左移 30 个像素，y轴下移 60 个像素
    M = np.float32([[1, 0, 30], [0, 1, 60]])

    img = cv2.warpAffine(img, M, (width, height))

    # 构建矩阵，旋转中心坐标为处理后图片长宽的一半，旋转角度为45度，缩放因子为1

    M = cv2.getRotationMatrix2D((width / 2, height / 2), 45, 1)

    dst = cv2.warpAffine(img, M, (width, height))
    cv2.imwrite('./static/abc.jpg', dst)
    return render(request,'index.html',{'image':'/static/abc.jpg'})
def f4_1(request):
    img = cv2.imread('./a1.jpg')
    src = cv2.resize(img, (256, 256))
    # 水平镜像
    horizontal = cv2.flip(src, 1, dst=None)
    # 垂直镜像
    vertical = cv2.flip(src, 0, dst=None)
    # 对角镜像 ，并保存
    cross = cv2.flip(src, -1, dst=None)
    cv2.imwrite('./static/abc.jpg', horizontal)
    return render(request,'index.html',{'image':'/static/abc.jpg'})
def f4_2(request):
    img = cv2.imread('./a1.jpg')
    src = cv2.resize(img, (256, 256))
    # 水平镜像
    horizontal = cv2.flip(src, 1, dst=None)
    # 垂直镜像
    vertical = cv2.flip(src, 0, dst=None)
    # 对角镜像 ，并保存
    cross = cv2.flip(src, -1, dst=None)
    cv2.imwrite('./static/abc.jpg', vertical)
    return render(request,'index.html',{'image':'/static/abc.jpg'})
def f4_3(request):
    img = cv2.imread('./a1.jpg')
    src = cv2.resize(img, (256, 256))
    # 水平镜像
    horizontal = cv2.flip(src, 1, dst=None)
    # 垂直镜像
    vertical = cv2.flip(src, 0, dst=None)
    # 对角镜像 ，并保存
    cross = cv2.flip(src, -1, dst=None)
    cv2.imwrite('./static/abc.jpg', cross)
    return render(request,'index.html',{'image':'/static/abc.jpg'})
def f5(request):
    img = cv2.imread('./a1.jpg')
    src = cv2.resize(img, (256, 256))
    # 获取图像shape
    rows, cols = src.shape[: 2]

    # 设置图像仿射变化矩阵
    post1 = np.float32([[50, 50], [200, 50], [50, 200]])
    post2 = np.float32([[10, 100], [200, 50], [100, 250]])
    M = cv2.getAffineTransform(post1, post2)

    # 图像仿射变换，及保存
    result = cv2.warpAffine(src, M, (rows, cols))
    cv2.imwrite('./static/abc.jpg', result)
    return render(request,'index.html',{'image':'/static/abc.jpg'})

def f6(request):
    def histCover(img, fileName):
        plt.figure(fileName, figsize=(16, 8))
        # 展示输入图像
        plt.subplot(121)
        plt.imshow(img, "gray")
        # 展示直方图
        plt.subplot(122)
        """
        利用cv.calcHist()内置函数进行画灰度图像直方图，该函数的返回值是hist
        """
        hist = cv2.calcHist([img], [0], None, [256], [0, 255])
        plt.plot(hist)
        plt.xlim([0, 255])
        plt.savefig(fileName)
        plt.show()

    # 主函数的定义，定义图片路径
    def main_func(argv):
        img_plt = './static/abc.jpg'
        """
        读入图像，并转化为灰度值，数据路径为img_path
        """
        img_gray = cv2.imread('./a1.jpg', cv2.IMREAD_GRAYSCALE)
        # img= cv.imread(img_path)
        # img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        """
        调用histCover函数绘制直方图，看清楚该函数是无返回值的哦
        """

        histCover(img_gray, img_plt)

    main_func(sys.argv)
    return render(request,'index.html',{'image':'/static/abc.jpg'})
def f7(request):
    def histCover(img, fileName):
        color = ["r", "g", "b"]
        # 展示原始图像
        """
        展示原始图像,因为用到cv2函数读取，要用matplotlib库函数，所以应该转BGR格式为RGB格式
        """
        b, g, r = cv2.split(img)
        img = cv2.merge([r, g, b])

        plt.subplot(121)
        plt.imshow(img)
        plt.subplot(122)
        # 绘制彩色直方图，需要对每个通道进行遍历，并且找到最大值和最小值
        for index, c in enumerate(color):

            """
            对每个通道进行直方图的计算和绘制，需要调用cv2的计算直方图函数，返回值为hist
            """
            hist = cv2.calcHist([img], [index], None, [256], [0, 255])


            plt.plot(hist, color=c)
            plt.xlim([0, 255])
        plt.savefig(fileName)
        plt.show()

    # 主函数的定义，定义图片路径
    def main_func(argv):
        img_path = './a1.jpg'
        img_plt = './static/abc.jpg'


        """
        根据给出的图像路径，加载图像，图像数据路径为img_path，返回值为imgOri1
        """
        imgOri1 = cv2.imread(img_path)


        histCover(imgOri1, img_plt)

    main_func(sys.argv)

    return render(request, 'index.html', {'image': '/static/abc.jpg'})
def f8(request):
    img = cv2.imread('./a1.jpg')  # 根据路径读取一张图片
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])

    plt.savefig('./static/abc.jpg')
    return render(request, 'index.html', {'image': '/static/abc.jpg'})
def f9(request):
    def grayHist(img, filename):
        plt.figure(filename, figsize=(16, 8))
        # 展示输入图像
        plt.subplot(121)
        plt.imshow(img, 'gray')
        # 展示直方图
        plt.subplot(122)
        h, w = img.shape[:2]


        """
        任务1.将二维图像矩阵reshape()为一维数组，返回值为pixelSequence
        """
        pixelSequence = img.reshape(h * w, 1)


        # 将二维图像矩阵reshape()为一维数组
        numberBins = 256

        """
        调用hist()的方法进行直方图的绘制，返回值有histogram, bins, patch
        """
        histogram, bins, patch = plt.hist(pixelSequence, numberBins)


        print(max(histogram))
        plt.xlabel("gray label")
        plt.ylabel("number of pixels")
        plt.axis([0, 255, 0, np.max(histogram)])
        # 打印输出峰值
        plt.savefig(filename)
        plt.show()


    # 定义图像数据的路径
    img_path = './a1.jpg'
    #out_path = 'D:/Mike/PycharmProjects/pythonProject/venv/abc.jpg'
    out2_path = './static/abc.jpg'

    img = cv2.imread(img_path, 0)
    h, w = img.shape[:2]
    out = np.zeros(img.shape, np.uint8)

    """
        通过遍历对不同像素范围内进行分段线性变化，在这里分三段函数进行分段线性变化,主要还是考察for循环的应用
        #y=0.5*x(x<50)
        #y=3.6*x-310(50<=x<150)
        #y=0.238*x+194(x>=150)
    """
    for i in range(h):
            for j in range(w):
                pix = img[i][j]
                if pix < 50:
                    out[i][j] = 0.5 * pix
                elif pix < 150:
                    out[i][j] = 3.6 * pix - 310
                else:
                    out[i][j] = 0.238 * pix + 194

    out = np.around(out)
    out = out.astype(np.uint8)
    #grayHist(img, out_path)
    grayHist(out, out2_path)

    return render(request, 'index.html', {'image': '/static/abc.jpg'})
def f10(request):
    img = cv2.imread('./a1.jpg')  # 根据路径读取一张图片

    img = cv2.GaussianBlur(img, (3, 3), 0)
    edges = cv2.Canny(img, 50, 150, apertureSize=3)

    # 使用HoughLines算法
    # rho为1
    # theta为np.pi/2
    # threshold为 118
    # 其他的默认

    lines = cv2.HoughLines(edges, 1, np.pi / 2, 118)


    result = img.copy()
    for i_line in lines:
        for line in i_line:
            rho = line[0]
            theta = line[1]
            if (theta < (np.pi / 4.)) or (theta > (3. * np.pi / 4.0)):  # 垂直直线
                pt1 = (int(rho / np.cos(theta)), 0)
                pt2 = (int((rho - result.shape[0] * np.sin(theta)) / np.cos(theta)), result.shape[0])
                cv2.line(result, pt1, pt2, (0, 0, 255))
            else:
                pt1 = (0, int(rho / np.sin(theta)))
                pt2 = (result.shape[1], int((rho - result.shape[1] * np.cos(theta)) / np.sin(theta)))
                cv2.line(result, pt1, pt2, (0, 0, 255), 1)

    minLineLength = 200
    maxLineGap = 15

    # 使用HoughLinesP算法
    # rho为1
    # theta为np.pi/2
    # threshold为 118
    # 并设置上面提供的minLineLength和maxLineGap
    # 其他的默认

    linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 80, minLineLength, maxLineGap)



    result_P = img.copy()
    for i_P in linesP:
        for x1, y1, x2, y2 in i_P:
            cv2.line(result_P, (x1, y1), (x2, y2), (0, 255, 0), 3)

    cv2.imwrite('./static/abc.jpg', result)
    #cv2.imwrite('D:/Mike/PycharmProjects/pythonProject/venv/def.jpg', result_P)
    return render(request, 'index.html', {'image': '/static/abc.jpg'})
def f11(request):
    img = cv2.imread('./a1.jpg')  # 根据路径读取一张图片

    img = cv2.GaussianBlur(img, (3, 3), 0)
    edges = cv2.Canny(img, 50, 150, apertureSize=3)

    # 使用HoughLines算法
    # rho为1
    # theta为np.pi/2
    # threshold为 118
    # 其他的默认

    lines = cv2.HoughLines(edges, 1, np.pi / 2, 118)


    result = img.copy()
    for i_line in lines:
        for line in i_line:
            rho = line[0]
            theta = line[1]
            if (theta < (np.pi / 4.)) or (theta > (3. * np.pi / 4.0)):  # 垂直直线
                pt1 = (int(rho / np.cos(theta)), 0)
                pt2 = (int((rho - result.shape[0] * np.sin(theta)) / np.cos(theta)), result.shape[0])
                cv2.line(result, pt1, pt2, (0, 0, 255))
            else:
                pt1 = (0, int(rho / np.sin(theta)))
                pt2 = (result.shape[1], int((rho - result.shape[1] * np.cos(theta)) / np.sin(theta)))
                cv2.line(result, pt1, pt2, (0, 0, 255), 1)

    minLineLength = 200
    maxLineGap = 15

    # 使用HoughLinesP算法
    # rho为1
    # theta为np.pi/2
    # threshold为 118
    # 并设置上面提供的minLineLength和maxLineGap
    # 其他的默认

    linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 80, minLineLength, maxLineGap)



    result_P = img.copy()
    for i_P in linesP:
        for x1, y1, x2, y2 in i_P:
            cv2.line(result_P, (x1, y1), (x2, y2), (0, 255, 0), 3)

    #cv2.imwrite('D:/Mike/PycharmProjects/djangoProject/static/abc.jpg', result)
    cv2.imwrite('./static/abc.jpg', result_P)
    return render(request, 'index.html', {'image': '/static/abc.jpg'})
def f12(request):
    filename = './a1.jpg'

    # 1. 灰度模式读取图像，图像名为CRH
    CRH = cv2.imread(filename, 0)
    # 2. 计算图像梯度。首先要对读取的图像进行数据变换，因为使用了
    # numpy对梯度进行数值计算，所以要使用
    # CRH.astype('float')进行数据格式变换。
    row, column = CRH.shape
    CRH_f = np.copy(CRH)
    gradient = np.zeros((row, column))
    CRH = CRH.astype('float')
    for x in range(row - 1):
        for y in range(column - 1):
            gx = abs(CRH[x + 1, y] - CRH[x, y])
            gy = abs(CRH[x, y + 1] - CRH[x, y])
            gradient[x, y] = gx + gy
            # 3. 对图像进行增强，增强后的图像变量名为sharp
    sharp = CRH_f + gradient

    sharp = np.where(sharp > 255, 255, sharp)
    sharp = np.where(sharp < 0, 0, sharp)

    # 数据类型变换
    gradient = gradient.astype('uint8')
    sharp = sharp.astype('uint8')

    # 保存图像
    filepath = './static/abc.jpg'
    cv2.imwrite(filepath, gradient)

    return render(request, 'index.html', {'image': '/static/abc.jpg'})
def f13_op(request):
    img = cv2.imread('./a1.jpg', cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    # 2. 定义十字形结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (10, 10))
    # 3. 对二值图进行开运算和闭运算操作
    im_op = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    #im_cl = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite('./static/abc.jpg', im_op)
    #cv2.imwrite('D:/Mike/PycharmProjects/pythonProject/venv/def.jpg', im_cl)
    return render(request, 'index.html', {'image': '/static/abc.jpg'})
def f13_cl(request):
    img = cv2.imread('./a1.jpg', cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    # 2. 定义十字形结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (10, 10))
    # 3. 对二值图进行开运算和闭运算操作
    #im_op = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    im_cl = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite('./static/abc.jpg', im_cl)
    #cv2.imwrite('D:/Mike/PycharmProjects/pythonProject/venv/def.jpg', im_cl)
    return render(request, 'index.html', {'image': '/static/abc.jpg'})
def f14(request):
    filename = './a1.jpg'
    # 读取图像
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    # 待输出的图片
    output = np.zeros(image.shape, np.uint8)
    # 遍历图像，获取叠加噪声后的图像
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j] < 100:

                # 添加食盐噪声
                output[i][j] = 255

                # 添加胡椒噪声
            elif image[i][j] > 200:
                output[i][j] = 0
                # 不添加噪声
            else:
                output[i][j] = image[i][j]
    cv2.imwrite('./static/abc.jpg', output)
    return render(request, 'index.html', {'image': '/static/abc.jpg'})
def f15(request):
    filename = './a1.jpg'
    # 读取图像
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    # 待输出的图片
    output = np.zeros(image.shape, np.uint8)
    # 遍历图像，进行均值滤波
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # 计算均值,完成对图片src的几何均值滤波
            ji = 1.0
            # 遍历滤波器内的像素值
            for m in range(-1, 2):
                # for n in range(-1, 2):
                # 防止越界
                if 0 <= j + m < image.shape[1]:
                    ji *= image[i][j + m]

            # 滤波器的大小为1*3
            output[i][j] = pow(ji, 1 / 3)
    cv2.imwrite('./static/abc.jpg', output)
    return render(request, 'index.html', {'image': '/static/abc.jpg'})
