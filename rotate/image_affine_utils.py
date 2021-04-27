import cv2
import numpy as np


def get_random_scale_size(img_w, img_h, use_scale_augment_prob, min_scale_size=0.9, max_scale_size=1.1, scale_size_step=0.1):
    """
    获取随机裁剪尺寸
    [in] img_w, 原图宽
    [in] img_h, 原图高
    [in] use_scale_augment_prob, 使用scale增强的概率，0为不做，1为必做
    [in] min_scale_size, 最小尺度
    [in] max_scale_size, 最大尺度
    [in] scale_size_step, 选取尺度步长
    [out] [scale_size,scale_size], 返回的随机裁剪尺寸
    """
    max_wh = max(img_w, img_h)
    if np.random.random() < use_scale_augment_prob:
        random_size = max_wh * \
            np.random.choice(
                np.arange(min_scale_size, max_scale_size, scale_size_step))
        return [random_size, random_size]
    else:
        return [max_wh, max_wh]


def Rotation_Transform(src_point, degree):
    radian = np.pi * degree / 180
    R_matrix = [[np.cos(radian), -np.sin(radian)],
                [np.sin(radian), np.cos(radian)]]
    R_matrix = np.asarray(R_matrix, dtype=np.float32)
    R_pts = np.matmul(R_matrix, src_point)
    return R_pts


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def create_affine_matrix(crop_center, crop_size, dst_size, inverse=False, use_random_angle=True, do_rotation_prob=0.5, fixed_angle=45):
    """
    创建仿射变换矩阵

    [in] crop_center, 要裁剪的中心
    [in] crop_size, 要裁剪的尺寸，小于原图尺寸为裁剪，否则为外扩
    [in] dst_size, 输出的尺寸
    [in] inverse, 输入输出是否相反
    [in] use_random_angle, 是否使用随机角度
    [in] do_rotation_prob, 做旋转的概率， 如果use_random_angle为False，这个参数不会用到。0为不做，1为必做。
    [in] fixed_angle, 固定角度旋转，如果use_random_angle为True，这个参数不会用到。注意逆时针旋转角度为正，参数范围是[-180，180]。
    [out] M,random_angle  返回的仿射变换矩阵，旋转角度
    """
    dst_center = np.array([dst_size[0]//2, dst_size[1]//2], dtype=np.float32)
    if use_random_angle:
        if np.random.rand(1) < do_rotation_prob:
            random_angle = np.random.rand(1)[0]*180-90
        else:
            random_angle = 0.
    else:
        random_angle = fixed_angle
    # print(random_angle)
    src_1 = crop_center
    src_2 = crop_center + \
        Rotation_Transform([0, -crop_size[0]//2], degree=random_angle)
    src_3 = get_3rd_point(src_1, src_2)
    src = np.asarray([src_1, src_2, src_3], np.float32)

    dst_1 = dst_center
    dst_2 = dst_center + [0, -dst_center[0]]
    dst_3 = get_3rd_point(dst_1, dst_2)
    dst = np.asarray([dst_1, dst_2, dst_3], np.float32)
    if inverse:
        M = cv2.getAffineTransform(dst, src)
    else:
        M = cv2.getAffineTransform(src, dst)
    return M, random_angle

"""
随机仿射变换
"""
if __name__ == "__main__":
    try:
        image = cv2.imread("./debug_in_img.jpg")
        img_h, img_w, img_c = image.shape
        dst_size = (768, 512)
        scale_size = get_random_scale_size(
            img_w, img_h, 1, min_scale_size=0.9, max_scale_size=1.2, scale_size_step=0.1)
        crop_center = np.asarray(
            [float(img_w)/2, float(img_h)/2], dtype=np.float32)

        # 测试1 不旋转
        # M,rotate_angle=create_affine_matrix(crop_center,scale_size,dst_size,inverse=False, use_random_angle=False, do_rotation_prob=1, fixed_angle=0)
        # 测试2 随机旋转
        M, rotate_angle = create_affine_matrix(
            crop_center, scale_size, dst_size, inverse=False, use_random_angle=True, do_rotation_prob=1, fixed_angle=90)
        # 测试3 固定角度旋转
        # M,rotate_angle=create_affine_matrix(crop_center,scale_size,dst_size,inverse=False, use_random_angle=False, do_rotation_prob=1, fixed_angle=90)

        print("rotate angle is {}".format(rotate_angle))
        new_image = cv2.warpAffine(
            src=image, M=M, dsize=dst_size, flags=cv2.INTER_LINEAR)
        cv2.imwrite("./test_out.jpg", new_image)
    except:
        pass