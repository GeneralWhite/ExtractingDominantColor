import numpy as np
import cv2


def gasuss_noise(image, mu=0.0, sigma=0.1):
    """
     添加高斯噪声
    :param image: 输入的图像
    :param mu: 均值
    :param sigma: 标准差
    :return: 含有高斯噪声的图像
    """
    image = np.array(image / 255, dtype=float)
    noise = np.random.normal(mu, sigma, image.shape)
    gauss_noise = image + noise
    if gauss_noise.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    gauss_noise = np.clip(gauss_noise, low_clip, 1.0)
    gauss_noise = np.uint8(gauss_noise * 255)
    return gauss_noise


if __name__ == '__main__':

    # ----------------------读取图片-----------------------------
    img = cv2.imread("Image/c2fe2c2bd913ba9dfe1c55b69fbcb24.jpg")
    # --------------------添加高斯噪声---------------------------
    out2 = gasuss_noise(img, mu=0.0, sigma=0.05)
    # ----------------------显示结果-----------------------------
    cv2.imshow('origion_pic', img)
    cv2.imshow('gauss_noise', out2)
    cv2.waitKey(0)

