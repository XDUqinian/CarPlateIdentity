# CarPlateIdentity
Design and development of an intelligent license plate recognition system.

## 大致思路

看了一些博客，在 GitHub 上找到一份源代码，对整个项目有了一定的了解。说白了就是给一张图片，然后找车牌位置，识别车牌，识别字符。其中识别字符要训练两个模型分别识别中文和数字，然后主程序会调用这两个模型对车牌进行识别。

一开始很奇怪，训练的模型要用什么样的方式储存呢？看了下源码发现模型是以dat为后缀的文件，存放在项目的一个文件夹下。就是说模型训练程序将训练好的模型存下来，主程序再去调用这个模型就可以了。

但是这个项目最麻烦的部分其实是如何找到车牌的位置，需要用到图像处理，而且不同的算法准确度会不一样。至于训练模型有函数可以用就还好。

## SVM

本项目使用 SVM 训练识别车牌中文和数字的模型。

支持向量机，是一种二分类模型，本质上是特征空间中最大化间隔的分类器。

说白了就是把要识别的东西提取出特征向量，每个特征向量映射为超维空间的一个点，具有相似特征的同类事物就会在这个超维空间有相近的坐标，形式上就聚集在一起。SVM 的目的就是画出一个比较好的线性分界，将不同类事物隔绝开。

支持向量是对识别问题起关键作用的向量，是离分类超平面（线性分界）最近的坐标点。

而最大化可以理解为公平原则，就是线性分界到两个区域的支持向量的距离相同。

训练的过程其实就是这个分界的绘制过程，训练好后，根据要识别的事物的特征向量对应的坐标点落在超维空间的哪个区域，就能确定该事物属于哪个分类。

OpenCV 自带的 SVM模型有两个重要参数 C 和 gamma。

C 是惩罚系数（对误差的宽容度）。C 越高说明越不能容忍出现误差，容易过拟合；C 越小，容易欠拟合；C 过大或过小，泛化能力变差。

gamma 是选择 RBF 函数作为 kernel 后（本项目使用 RBF 作为核函数），该函数自带的一个参数，隐含决定了数据映射到新的特征空间后的分布。gamma 越大，支持向量越少；gamma 越小，支持向量越多。支持向量的个数影响训练与预测的速度。

因为训练量比较小所以识别准确度不高，后面会再训练。

## 源码解读

[GitHub 上找到的一份源码](https://github.com/yinghualuowu/Python_VLPR)

### 一些概念

- 高斯滤波：高斯滤波用于消除图像中的噪声。[传送门](https://blog.csdn.net/weixin_51571728/article/details/121527964?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522166386049016782427459387%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=166386049016782427459387&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-2-121527964-null-null.142^v50^control,201^v3^add_ask&utm_term=%E9%AB%98%E6%96%AF%E6%BB%A4%E6%B3%A2&spm=1018.2226.3001.4187)
- 开运算：对图像进行腐蚀后膨胀，用于去除图像中仅存的小块像素。[传送门](https://blog.csdn.net/hhaowang/article/details/102296876?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522166386193916782412578786%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=166386193916782412578786&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-2-102296876-null-null.142^v50^control,201^v3^add_ask&utm_term=morphologyEx%28%29%20&spm=1018.2226.3001.4187)
- 大津二值化算法：二值化有利于找到图像边缘，大津法可以自己得到阈值。[传送门](https://blog.csdn.net/qq_40243750/article/details/117433179?ops_request_misc=&request_id=&biz_id=102&utm_term=%E5%A4%A7%E6%B4%A5%E6%B3%95%E4%BA%8C%E5%80%BC%E5%8C%96&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-0-117433179.142^v50^control,201^v3^add_ask&spm=1018.2226.3001.4187)

### 程序流程

- 读入一张图片，对图片做预处理。img_first_pre 函数返回处理好的图像文件和原图像文件。
  - 缩小图片：如果图片宽度大于最大宽度，将图片缩小。
  - 高斯滤波：对图片进行高斯滤波，高斯核长宽为 blur=5，标准差为0。
  - 转换为灰度图像：将高斯滤波后的图片转为灰度图像，并保留高斯滤波后的图片副本。
  - 开运算：用长宽为20的卷积核对图像进行开操作，消除前景噪声。
  - 合并：采用图像叠加（灰度图-开操作图）突显字符等部分。
  - 大津法二值化：对灰度图像二值化，得到黑白图像。第一个 ret 是大津法得到的阈值。
  - 获取边缘：利用Canny算子进行边缘检测，利用闭运算及开运算使图像边缘成为一个整体。
- get_imgtk
  - 把 img 从bgr 格式 转换为 rgb 格式。
  - 从 img array 获得 Image im。
  - 用 ImageTk.PhotoImage 给 tkinter 添加图片，显示选择的图片。
  - 如果图片太大，就按比例缩小一下。
  - 返回 imgtk
- from_pic
  - 获取图片路径
  - 路径存在则读图
  - 对图像进行 first_pre 预处理
  - 给 tkinter 添加图像
  - 然后开了两个线程，执行函数 img_color_contours 和 img_only_color，并传入对应的参数。
- img_color_contours
  - 获取预边缘图像的高和宽
  - 调用 img_findContours 获取疑似车牌轮廓
  - 调用 img_Transform 进行矩形矫正
  - 调用 img_color 获取符合车牌颜色的疑似车牌
  - 根据车牌不同颜色对图片做不同处理，调用训练好的模型对字符进行判断。
  - 返回值：识别到的字符、定位的车牌图像、车牌颜色
- img_only_color
  - 这个函数其实和前面 img_first_pre 得到的图片没关系，它内部又仅从颜色角度获得了边缘图像。
  - 剩下的操作就和 img_color_contours 差不多了。
- img_findContours
  - 用 cv2 自带的轮廓检测函数  [cv2.findContours()](https://blog.csdn.net/weixin_40522801/article/details/106496507?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522166391820616782425174752%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=166391820616782425174752&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-106496507-null-null.142^v50^control,201^v3^add_ask&utm_term=cv2.findContours&spm=1018.2226.3001.4187) 检测所有轮廓，建立完整的层次结构，建立网状轮廓结构；轮廓的近似算法，压缩水平方向，垂直方向，对角线方向的元素，值保留该方向的重点坐标，如果一个矩形轮廓只需4个点来保存轮廓信息。返回值 hierarchy 表示每条轮廓的属性。
  - 排除小面积的轮廓，最小面积设为了 2000。
  - 求出点集下的最小面积矩形 [cv2.minAreaRect()](https://blog.csdn.net/weixin_38640670/article/details/119414974?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522166391866716782412570912%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=166391866716782412570912&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-1-119414974-null-null.142^v50^control,201^v3^add_ask&utm_term=cv2.minAreaRect%28cnt%29&spm=1018.2226.3001.4187) ，也就是轮廓的面积。根据长宽比例进一步排除不是车牌的轮廓。[box = cv2.boxPoints(ant)](https://blog.csdn.net/Maisie_Nan/article/details/105833892?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522166391899416782427494131%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=166391899416782427494131&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-105833892-null-null.142^v50^control,201^v3^add_ask&utm_term=cv2.boxPoints&spm=1018.2226.3001.4187)  获取该矩形的四个顶点坐标，不过没用上。

## 一些问题

- 边缘识别已经识别到车牌，但在图像校正后车牌位置却不见了。后续再看看图像校正函数的具体实现。
