# 1.30 图像的叠加
import cv2
import matplotlib.pyplot as plt
 # 读取彩色图像(BGR)
img1=cv2.imread("csr_0001_aligned_aligned.png")
img2 = cv2.imread("example(4).png")  # 读取 CV Logo

x, y = (90, 60)  # 图像叠加位置
W1, H1 = img1.shape[1::-1]
W2, H2 = img2.shape[1::-1]
if (x + W2) > W1: x = W1 - W2
if (y + H2) > H1: y = H1 - H2
print(W1, H1, W2, H2, x, y)
imgROI = img1[y:y + H2, x:x + W2]  # 从背景图像裁剪出叠加区域图像

img2Gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)  # img2: 转换为灰度图像
ret, mask = cv2.threshold(img2Gray, 175, 255, cv2.THRESH_BINARY)  # 转换为二值图像，生成遮罩，LOGO 区域黑色遮盖
maskInv = cv2.bitwise_not(mask)  # 按位非(黑白转置)，生成逆遮罩，LOGO 区域白色开窗，LOGO 以外区域黑色

# mask 黑色遮盖区域输出为黑色，mask 白色开窗区域与运算（原图像素不变）
img1Bg = cv2.bitwise_and(imgROI, imgROI, mask=mask)  # 生成背景，imgROI 的遮罩区域输出黑色
img2Fg = cv2.bitwise_and(img2, img2, mask=maskInv)  # 生成前景，LOGO 的逆遮罩区域输出黑色
# img1Bg = cv2.bitwise_or(imgROI, imgROI, mask=mask)  # 生成背景，与 cv2.bitwise_and 效果相同
# img2Fg = cv2.bitwise_or(img2, img2, mask=maskInv)  # 生成前景，与 cv2.bitwise_and 效果相同
# img1Bg = cv2.add(imgROI, np.zeros(np.shape(img2), dtype=np.uint8), mask=mask)  # 生成背景，与 cv2.bitwise 效果相同
# img2Fg = cv2.add(img2, np.zeros(np.shape(img2), dtype=np.uint8), mask=maskInv)  # 生成背景，与 cv2.bitwise 效果相同
imgROIAdd = cv2.add(img1Bg, img2Fg)  # 前景与背景合成，得到裁剪部分的叠加图像
imgAdd = img1.copy()
imgAdd[y:y + H2, x:x + W2] = imgROIAdd  # 用叠加图像替换背景图像中的叠加位置，得到叠加 Logo 合成图像

plt.figure(figsize=(9, 6))
titleList = ["1. imgGray", "2. imgMask", "3. MaskInv", "4. img2FG", "5. img1BG", "6. imgROIAdd"]
imageList = [img2Gray, mask, maskInv, img2Fg, img1Bg, imgROIAdd]
for i in range(6):
    plt.subplot(2, 3, i + 1), plt.title(titleList[i]), plt.axis('off')
    if (imageList[i].ndim == 3):  # 彩色图像 ndim=3
        plt.imshow(cv2.cvtColor(imageList[i], cv2.COLOR_BGR2RGB))  # 彩色图像需要转换为 RGB 格式
    else:  # 灰度图像 ndim=2
        plt.imshow(imageList[i], 'gray')
plt.show()
cv2.imshow("imgAdd", imgAdd)  # 显示叠加图像 imgAdd
key = cv2.waitKey(0)  # 等待按键命令
