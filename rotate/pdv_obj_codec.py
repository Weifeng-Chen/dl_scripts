import torch
import cv2
import numpy as np
import math

def rotate_point(xc,yc, xp,yp, theta):
    xoff = xp-xc
    yoff = yp-yc

    cosTheta = math.cos(theta)
    sinTheta = math.sin(theta)
    pResx = cosTheta * xoff + sinTheta * yoff
    pResy = - sinTheta * xoff + cosTheta * yoff
    return xc+pResx,yc+pResy

def convert_cxywha_to_vertexs(cx,cy,w,h,angle,return_int=False):
    p0x,p0y = rotate_point(cx,cy, cx - w/2, cy - h/2, -angle)
    p1x,p1y = rotate_point(cx,cy, cx + w/2, cy - h/2, -angle)
    p2x,p2y = rotate_point(cx,cy, cx + w/2, cy + h/2, -angle)
    p3x,p3y = rotate_point(cx,cy, cx - w/2, cy + h/2, -angle)
    if return_int:
        points = [(int(p0x), int(p0y)), (int(p1x), int(p1y)), (int(p2x), int(p2y)), (int(p3x), int(p3y))]
    else:
        points = [(p0x, p0y), (p1x, p1y), (p2x, p2y), (p3x, p3y)]
    return points

def gaussian_radius(det_size, min_overlap=0.7):
    """
    计算高斯核尺寸
    """
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
    """
    计算高斯核
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    """
    热度图打点
    """
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius +
                               bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def heatmap_nms(heatmap, kernel_size=3):
    hmax = torch.nn.functional.max_pool2d(
        heatmap, (kernel_size, kernel_size), stride=1, padding=(kernel_size - 1) // 2)
    keep = (hmax == heatmap).float()
    return heatmap * keep


def object_encode(image, label_data, opts):
    """
    训练目标编码

    [in] image, 输入图像，opencv读入格式，经由 data_augmentation@PDVDataset 得到
    [in] label_data, 输入标注数据，经由 data_augmentation@PDVDataset 得到
    [in] opts, 输入参数
    [out] gt_encode, 返回的编码后的真值训练目标
    """
    img_h, img_w, img_c = image.shape
    obj_num = label_data["label_cxywha"].shape[0]
    down_ratio = opts["down_ratio"]
    cat_num = 3  # 固定，0：人头；1：头肩；2：全身
    max_obj_num = 100 * cat_num
    # max_obj_num = min(max_obj_num, obj_num)
    hm_h = img_h//down_ratio
    hm_w = img_w//down_ratio

    # 图像数据数据归一化
    image_norm = np.asarray(np.clip(image, a_min=0., a_max=255.), np.float32)
    image_norm = np.transpose(image_norm / 255. - 0.5, (2, 0, 1))

    # 初始化
    hm = np.zeros((cat_num, hm_h, hm_w), dtype=np.float32)
    wh = np.zeros((max_obj_num, 2), dtype=np.float32)
    angle = np.zeros((max_obj_num, 1), dtype=np.float32)
    reg = np.zeros((max_obj_num, 2), dtype=np.float32)
    reg_mask = np.zeros((max_obj_num), dtype=np.uint8)
    ind = np.zeros((max_obj_num), dtype=np.int64)
    cat_mask = np.zeros((max_obj_num), dtype=np.uint8)

    for k in range(obj_num):
        src_pts = label_data["label_cxywha"][k]
        src_cat = int(label_data["cat"][k])

        xy_down_ratio_fp32=src_pts[0:2]/down_ratio
        xy_down_ratio_int32=xy_down_ratio_fp32.astype(np.int32)
        wh_down_ratio_fp32=src_pts[2:4]/down_ratio
        wh_down_ratio_int32=wh_down_ratio_fp32.astype(np.int32)

        # hm 打点
        radius = gaussian_radius(
            (math.ceil(wh_down_ratio_int32[0]), math.ceil(wh_down_ratio_int32[1])))
        radius = max(0, int(radius))
        draw_umich_gaussian(hm[src_cat], xy_down_ratio_int32, radius)

        # wh 赋值
        wh[k] = wh_down_ratio_fp32
        # angle 赋值
        angle[k] = src_pts[4]*180/math.pi
        # reg 和 reg_mask 赋值
        reg[k] = wh_down_ratio_fp32-wh_down_ratio_int32
        reg_mask[k] = 1
        # ind 赋值
        ind[k] = xy_down_ratio_int32[1]*hm_w+xy_down_ratio_int32[0]
        assert(ind[k] >= 0)
        # cat_mask 赋值
        cat_mask[k] = src_cat

    # # 绘图检查
    # # 只在debug时开启测试
    # hm_draw=hm*255
    # hm_draw = np.clip(hm_draw, a_min=0., a_max=255.).astype(np.uint8)
    # cv2.imwrite("./head_hm.jpg",hm_draw[0])
    # cv2.imwrite("./hbody_hm.jpg",hm_draw[1])
    # cv2.imwrite("./fbody_hm.jpg",hm_draw[2])
    # pts_int32 = label_data["pts"].astype(np.int32)
    # draw_obj_num = pts_int32.shape[0]

    # for i in range(draw_obj_num):
    #     draw_color = (0, 0, 0)
    #     if label_data["cat"][i] == 0:
    #         draw_color = (255, 0, 0)
    #     elif label_data["cat"][i] == 1:
    #         draw_color = (128, 0, 128)
    #     elif label_data["cat"][i] == 2:
    #         draw_color = (0, 255, 0)

    #     for k in range(4):
    #         cv2.putText(image,str(k+1),tuple(pts_int32[i, k+1]),cv2.FONT_HERSHEY_SIMPLEX,0.5,draw_color)

    #     cv2.line(image, tuple(pts_int32[i, 1]),
    #                 tuple(pts_int32[i, 2]), draw_color, 1)
    #     cv2.line(image, tuple(pts_int32[i, 1]),
    #                 tuple(pts_int32[i, 4]), draw_color, 1)
    #     cv2.line(image, tuple(pts_int32[i, 3]),
    #                 tuple(pts_int32[i, 2]), draw_color, 1)
    #     cv2.line(image, tuple(pts_int32[i, 3]),
    #                 tuple(pts_int32[i, 4]), draw_color, 1)
    # cv2.imwrite("./test_out.jpg", image)

    # 赋值
    gt_encode = {}
    gt_encode["input"] = image_norm
    gt_encode["hm"] = hm
    gt_encode["wh"] = wh
    gt_encode["angle"] = angle
    gt_encode["reg"] = reg
    gt_encode["reg_mask"] = reg_mask
    gt_encode["ind"] = ind
    gt_encode["cat_mask"] = cat_mask
    return gt_encode


def object_decode(predict, opts):
    """
    推理结果解码
    [in] image, opencv格式
    [in] predict, 预测结果
    [in] opts, 参数
    """

    """
    predict 输出格式, 参看PDVUNBox@pdv_un_net实现，是一个字典
    hm,  热度图，shape为[batch_size, num_class,h,w]，num_class为类别数，当前版本（v0.1）为3
    wh,  宽高预测，shape为[batch_size,2,h,w] ，维度0的数据循序是pred_w,pred_h
    angle, 角度预测，shape为[batch_size,num_class,h,w]，预测的是(0，360)的值
    reg, 偏移预测，shape为[batch_size,2,h,w] ，维度0的数据循序是offset_w,offset_h
    注意，上述的h和w是经过down_ratio(对应opts["down_ratio"]),转回原图时，要乘down_ratio
    """

    # step-1 按类别找出hm的top-k
    hm = predict["hm"].clone()
    wh = predict["wh"].clone()
    reg = predict["reg"].clone()
    angle = predict["angle"].clone()
    batch_size, num_class, hm_h, hm_w = hm.size()
    assert batch_size == 1
    hm = hm.squeeze(0)
    wh = wh.squeeze(0)
    reg = reg.squeeze(0)
    angle = angle.squeeze(0)

    hm_max = heatmap_nms(hm)
    hm_size = hm_h*hm_w
    max_topk = min(opts["decode_topk"], hm_size)
    top_scores, top_inds = torch.topk(hm_max.view(num_class, -1), max_topk)

    # step-2 根据过滤后的top-k 索引，在wh，angle，reg中收集对应的数据
    reg = reg.reshape(2, -1).transpose(1, 0)
    wh = wh.reshape(2, -1).transpose(1, 0)
    angle = angle.reshape(1, -1).transpose(1, 0)
    predict_objs={}
    for i in range(num_class):
        objs_cat_i=[]
        top_inds_cat_i = top_inds[i, :][top_scores[i, :]
                                        > opts["decode_confidence_th"]]
        top_scores_cat_i=top_scores[i, :][top_scores[i, :]> opts["decode_confidence_th"]]
        top_num = top_inds_cat_i.shape[0]
        reg_gather_idx = top_inds_cat_i.unsqueeze(1).expand((top_num, 2))
        angle_gather_idx = top_inds_cat_i.unsqueeze(1).expand((top_num, 1))
        top_reg = reg.gather(0, reg_gather_idx)
        top_wh = wh.gather(0, reg_gather_idx)
        top_angle = angle.gather(0, angle_gather_idx)

        for k in range(top_num):
            obj_info={}
            obj_info["cx"]=top_inds_cat_i[k].item()%(int(opts["input_width"]/opts["down_ratio"]))
            obj_info["cy"]=int(top_inds_cat_i[k].item()/(int(opts["input_width"]/opts["down_ratio"])))
            obj_info["w"]=top_wh[k][0].item()
            obj_info["h"]=top_wh[k][1].item()
            obj_info["offset_x"]=top_reg[k][0].item()
            obj_info["offset_y"]=top_reg[k][1].item()
            obj_info["angle"]=top_angle[k].item()
            obj_info["score"]=top_scores_cat_i[k].item()
            objs_cat_i.append(obj_info)
        predict_objs[str(i)]=objs_cat_i

    return predict_objs


def convert_gt_encode_to_predict_data(gt_encode):
    """
    用gt编码后的数据生成模型预测数据，用于测试
    """
    predict_data = {}
    hm_c, hm_h, hm_w = gt_encode["hm"].shape
    wh = np.zeros((2, hm_h, hm_w), dtype=np.float32)
    reg = np.zeros((2, hm_h, hm_w), dtype=np.float32)
    angle = np.zeros((1, hm_h, hm_w), dtype=np.float32)

    real_obj_idx = gt_encode["ind"][gt_encode["reg_mask"].astype(np.bool)]
    gt_wh = gt_encode["wh"][gt_encode["reg_mask"].astype(np.bool)]
    gt_reg = gt_encode["reg"][gt_encode["reg_mask"].astype(np.bool)]
    gt_angle = gt_encode["angle"][gt_encode["reg_mask"].astype(np.bool)]
    wh.reshape(2, -1).transpose(1, 0)[real_obj_idx, :] = gt_wh
    reg.reshape(2, -1).transpose(1, 0)[real_obj_idx, :] = gt_reg
    angle.reshape(1, -1).transpose(1, 0)[real_obj_idx, :] = gt_angle

    predict_data["hm"] = torch.from_numpy(gt_encode["hm"])
    predict_data["wh"] = torch.from_numpy(wh)
    predict_data["reg"] = torch.from_numpy(reg)
    predict_data["angle"] = torch.from_numpy(angle)

    predict_data["hm"] = predict_data["hm"].unsqueeze(0)
    predict_data["wh"] = predict_data["wh"].unsqueeze(0)
    predict_data["reg"] = predict_data["reg"].unsqueeze(0)
    predict_data["angle"] = predict_data["angle"].unsqueeze(0)

    return predict_data


if __name__ == "__main__":
    try:
        import sys
        sys.path.append('../')
        from pdv_labels_xml import PDVLabel, PDVObjHead, PDVObjHbody, PDVObjFbody
        from image_affine_augment import convert_raw_label_to_vertex_pts, random_filp, random_affine, label_filter, convert_vertex_pts_to_cxywha

        image = cv2.imread("./Round_0-part2-review_192.168.16.27_image1_a00.jpg")
        pdv_label = PDVLabel()
        pdv_label.load_xml("./Round_0-part2-review_192.168.16.27_image1_a00.xml")
        raw_object_dict = pdv_label.get_objects()

        # 格式转换
        img_h, img_w, img_c = image.shape
        label_data = convert_raw_label_to_vertex_pts(image, raw_object_dict)

        # 随机翻转
        image_aug, label_data_aug = random_filp(image, label_data)
        image_aug = image_aug.astype(np.uint8)

        # 随机仿射变换
        opts = {}
        opts["train_type"] = "only_rbbox"
        opts["input_width"] = 768
        opts["input_height"] = 512
        opts["scale_augment_prob"] = 0.9
        opts["scale_augment_min_scale_size"] = 0.9
        opts["scale_augment_max_scale_size"] = 1.2
        opts["scale_augment_scale_size_step"] = 0.1
        opts["do_rotation_prob"] = 1.0
        opts["down_ratio"] = 4
        opts["decode_topk"] = 100
        opts["decode_confidence_th"] = 0.5
        image_aug, label_data_aug = random_affine(
            image_aug, label_data_aug, opts)

        # 过滤由变换造成的不完整标注
        label_data_aug = label_filter(image_aug, label_data_aug)

        # 标注数据格式转换
        label_data_aug = convert_vertex_pts_to_cxywha(label_data_aug)

        # 真值数据编码
        gt_encode = object_encode(image_aug, label_data_aug, opts)

        # 解码
        predict_data = convert_gt_encode_to_predict_data(gt_encode)
        predict_objs = object_decode(predict_data, opts)

        # 绘图
        img=gt_encode["input"]
        img=(img+0.5)*255
        img = np.asarray(np.clip(img, a_min=0., a_max=255.), np.float32)
        img=img.astype(np.uint8)
        img=img.transpose(1,2,0)
        img_draw=img.copy()
        head_objs_list=predict_objs["0"]
        hbody_objs_list=predict_objs["1"]
        fbody_objs_list=predict_objs["2"]
        for head_obj in head_objs_list:
            cx=int(head_obj["cx"]*opts["down_ratio"])
            cy=int(head_obj["cy"]*opts["down_ratio"])
            w=int(head_obj["w"]*opts["down_ratio"])
            h=int(head_obj["h"]*opts["down_ratio"])
            angle=float(head_obj["angle"])*math.pi/180
            vertexs=convert_cxywha_to_vertexs(cx,cy,w,h,angle,return_int=True)
            cv2.line(img_draw,vertexs[0],vertexs[1],(0,0,255),1)
            cv2.line(img_draw,vertexs[0],vertexs[3],(0,0,255),1)
            cv2.line(img_draw,vertexs[2],vertexs[1],(0,0,255),1)
            cv2.line(img_draw,vertexs[2],vertexs[3],(0,0,255),1)
            cv2.circle(img_draw,(cx,cy),3,(0,0,255),-1)

        for hbody_obj in hbody_objs_list:
            cx=int(hbody_obj["cx"]*opts["down_ratio"])
            cy=int(hbody_obj["cy"]*opts["down_ratio"])
            w=int(hbody_obj["w"]*opts["down_ratio"])
            h=int(hbody_obj["h"]*opts["down_ratio"])
            angle=float(hbody_obj["angle"])*math.pi/180
            vertexs=convert_cxywha_to_vertexs(cx,cy,w,h,angle,return_int=True)
            cv2.line(img_draw,vertexs[0],vertexs[1],(64,128,255),1)
            cv2.line(img_draw,vertexs[0],vertexs[3],(64,128,255),1)
            cv2.line(img_draw,vertexs[2],vertexs[1],(64,128,255),1)
            cv2.line(img_draw,vertexs[2],vertexs[3],(64,128,255),1)
            cv2.circle(img_draw,(cx,cy),3,(64,128,255),-1)

        for fbody_obj in fbody_objs_list:
            cx=int(fbody_obj["cx"]*opts["down_ratio"])
            cy=int(fbody_obj["cy"]*opts["down_ratio"])
            w=int(fbody_obj["w"]*opts["down_ratio"])
            h=int(fbody_obj["h"]*opts["down_ratio"])
            angle=float(fbody_obj["angle"])*math.pi/180
            vertexs=convert_cxywha_to_vertexs(cx,cy,w,h,angle,return_int=True)
            cv2.line(img_draw,vertexs[0],vertexs[1],(0,255,0),1)
            cv2.line(img_draw,vertexs[0],vertexs[3],(0,255,0),1)
            cv2.line(img_draw,vertexs[2],vertexs[1],(0,255,0),1)
            cv2.line(img_draw,vertexs[2],vertexs[3],(0,255,0),1)
            cv2.circle(img_draw,(cx,cy),5,(128,0,128),-1)

        cv2.imwrite("./test_out.jpg",img_draw)
    except:
        pass
