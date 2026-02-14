import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt # 用于画图
from tools.ch13 import set_figsize, set_title

torch.set_printoptions(2)  # 精简输出精度

'''生成多个锚框'''
#@save
# data 是输入张量
# sizes, ratios 是人为选取的缩放比、宽高比张量
def multibox_prior(data, sizes, ratios):
    """生成以每个像素为中心具有不同形状的锚框"""
    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    # 由于直接在 GPU 上计算，所以缩放比、宽高比直接进入 GPU
    size_tensor = torch.tensor(sizes, device=device)
    ratio_tensor = torch.tensor(ratios, device=device)

    # 为了将锚点移动到像素的中心，需要设置偏移量。
    # 因为一个像素的位置是 1，但是我们需要将锚点坐标移到物理中心
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height
    steps_w = 1.0 / in_width
    # 将锚框坐标归一化到 [0,1] 范围，一个像素的步长为 1.0 / in_height
    
    # 生成锚框的所有中心点（每个像素都是中心）
    # range 也要返回一个序列，因此也要指定设备
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
    shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing='ij')
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)
    # 将二维网格展平为一维数组，方便后续批量计算

    # 生成 “boxes_per_pixel” 个高和宽，
    # 之后用于创建锚框的四角坐标 (xmin, xmax, ymin, ymax)
    # 锚框面积固定为 size²，宽高比 r = w/h → 推导得：w = size * √r，h = size / √r
    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),
                   sizes[0] * torch.sqrt(ratio_tensor[1:]))) * in_height / in_width  
    # w0 * in_height / in_width 是因为 w0 h0 都是归一化到 [0, 1]。为了让宽高比保持一致，必须这样修正！
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]),
                   sizes[0] / torch.sqrt(ratio_tensor[1:])))
    
    # 除以 2 来获得半高和半宽（因为这是相对于锚框中心点的偏移）
    # 前面 w h 都是一维张量 (boxes_per_pixel)
    # 拼接后，转置前 anchor_manipulations (4, boxes_per_pixel)
    # 转置后 (boxes_per_pixel, 4)
    # repeat(a, b) 表示：第一维重复 a 次，第二维重复 b 次。这里第二维不重复
    # 重复后 (boxes_per_pixel * 像素总数, 4) ⇒ 相当于为每个像素复制一份所有锚框的偏移量
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(
                                        in_height * in_width, 1) / 2
    # ⇒ 从上面代码可以看出，这里锚框的表示法是【左上+右下】；而且坐标归一化。

    # 每个中心点都将有 “boxes_per_pixel” 个锚框，
    # 所以生成含所有锚框中心的网格，重复了 “boxes_per_pixel” 次
    # stack([shift_x, shift_y, shift_x, shift_y])：将每个中心点 (x,y) 扩展为 
    # (x,y,x,y)（对应锚框的(xmin, ymin, xmax, ymax)初始值）
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y],
                dim=1).repeat_interleave(boxes_per_pixel, dim=0)
    output = out_grid + anchor_manipulations # out_grid 是锚框中心点；anchor_manipulations 是偏移。
    return output.unsqueeze(0)
    # output.unsqueeze(0) 的张量形状是 (1, boxes_per_pixel * 像素总数, 4) 即 (目标总数, 锚框总数, 4)
    # ⇒ 也就是说：【画锚框本身】不关心【通道数】

img = d2l.plt.imread('data/img/catdog.jpg')
h, w = img.shape[:2] # 看来 plt.imread 返回的张量形状没有 channels

print(f'h, w: {h}, {w}')
X = torch.rand(size=(1, 3, h, w))
Y = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
print(f'Y.shape: {Y.shape}')

# 因为 Y 的形状是 (1, boxes_per_pixel * h * w, 4)
# 在本例子中 boxes_per_pixel = 3 + 3 - 1 = 5
boxes = Y.reshape(h, w, 5, 4)
# 访问像素点 (250, 250) 处的其中一个锚框坐标表示
print(f'boxes[250, 250, 0, :]: {boxes[250, 250, 0, :]}')

#@save
def show_bboxes(axes, bboxes, labels=None, colors=None):
    """显示所有边界框（绘图代码）"""
    def _make_list(obj, default_values=None):
        if obj is None:  # 如果参数里面没给 obj，就用后面 default_values 指定的，例如下面的 color 就是这个意思
            obj = default_values
        # 这边 if isinstance 用来判断 obj 的类型是不是我们制定的
        # 补充：list 是列表，tuple 是元组 (元组是有序列表，并且一旦初始化就不可以再修改了)
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj

    labels = _make_list(labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = d2l.bbox_to_rect(bbox.detach().numpy(), color)
        axes.add_patch(rect)
        # 下面的是用来进入 text 文字描述的
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va='center', ha='center', fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0))
            
set_figsize()
bbox_scale = torch.tensor((w, h, w, h))

plt.figure(1)
fig = d2l.plt.imshow(img)
drawFig = plt.gcf() # d2l.plt.imshow() 和 plt.gcf() 返回类型不一样！
set_title(drawFig, '第一步：在 (250, 250) 像素点处生成多个锚框')

show_bboxes(fig.axes, boxes[250, 250, :, :] * bbox_scale,
            ['s=0.75, r=1', 's=0.5, r=1', 's=0.25, r=1', 's=0.75, r=2',
             's=0.75, r=0.5'])


'''计算交并比'''
#@save
def box_iou(boxes1, boxes2):
    """计算两个锚框或边界框列表中成对的交并比"""
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                              (boxes[:, 3] - boxes[:, 1]))
    # boxes1,boxes2,areas1,areas2 的形状:
    #（其实 boxes1 和 boxes2 一个是锚框，一个是边界框；）
    #（确切地说，这里没有办法分清：谁是锚框，谁是边界框）
    # boxes1：(boxes1的数量,4),
    # boxes2：(boxes2的数量,4),
    # areas1：(boxes1的数量,),
    # areas2：(boxes2的数量,)
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    # 张量索引里的 None 等价于 torch.unsqueeze()，作用是给张量 “插入” 一个长度为 1 的新维度
    # boxes1[:, None, :2]把(N,2)变成了(N, 1, 2)（插入了一个长度为 1 的中间维度），此时：
    # 张量 A：(N, 1, 2)（boxes1 扩展后）
    # 张量 B：(M, 2)（boxes2 未扩展）
    # PyTorch 的广播机制会自动把这两个张量 “扩展” 成相同形状(N, M, 2)
    '''这里一维变二维的代码写的是真简洁！'''
    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
    # 计算交集面积，同时处理无交集的情况
    # inter_areasandunion_areas 的形状:(boxes1的数量, boxes2的数量)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    # inters[:, :, 0]：
    # : 表示取第 0 维的所有元素（所有 N 个框）；
    # 第二个 : 表示取第 1 维的所有元素（所有 M 个框）；
    # 0 表示只取第 2 维的第 0 个元素（仅宽度）。
    # → 这个操作会「去掉」第 2 维，结果形状是 (N, M)
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas # 返回二维张量 (boxes1的数量, boxes2的数量)


'''
在训练集中标注锚框。
【两步分配原则】：在目标检测中，锚框（anchors）数量远多于真实边界框（ground_truth），分配的核心诉求是：
1. 高质量匹配的锚框要优先分配到对应真实框（保证正样本质量）；
2. 每个真实框必须有至少一个锚框对应（避免真实框无锚框学习，导致训练漏检）。
'''
#@save
def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """将最接近的真实边界框分配给锚框"""
    # ground_truth (真实边界框的数量, 4); anchors (锚框的数量, 4)
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # 位于第 i 行和第 j 列的元素 x_ij 是锚框 i 和真实边界框 j 的 IoU
    # jaccard 张量形状 (num_anchors, num_gt_boxes)
    jaccard = box_iou(anchors, ground_truth)
    # 对于每个锚框，分配的真实边界框的张量
    # 每定义一个新的数据，就要指定它的 device
    # 其实这里的 torch.full 方法是用于初始化的！
    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long,
                                  device=device)
    '''这里的思路是：求出每一行的最大值；然后分别求出该最大值的行、列索引；然后让行索引对应到真实的边缘框。
    整个过程只有 jaccard 是二维张量，其余全都是一维张量！'''
    # 对于每个 锚框 行，求出 iou 值最大的 锚框
    max_ious, indices = torch.max(jaccard, dim=1)
    # max_ious 和 indices 形状都是 (num_anchors) 因为 keepdim=False
    # 注意：这里的 (max_ious >= iou_threshold) 整体是个布尔张量！不会有负数！
    anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1) # 超出阈值的锚框索引
    box_j = indices[max_ious >= iou_threshold] # 超出阈值的真实框（注意：不是真实框索引！）
    # indices[i] 表示 “第 i 个锚框” 匹配到的 “最大 IoU 对应的真实边缘框索引”
    # 布尔张量索引 的本质是：用一个和目标张量形状完全相同的布尔张量，
    # ⇒ box_j 用于筛选出目标张量中 “布尔值为 True” 的位置对应的所有元素
    anchors_bbox_map[anc_i] = box_j # (num_anchors, 1) 让锚框索引到真实边缘框
    col_discard = torch.full((num_anchors,), -1)
    row_discard = torch.full((num_gt_boxes,), -1)
    for _ in range(num_gt_boxes):
        max_idx = torch.argmax(jaccard) # 不指定维度，这里 max_idx 是一个实数！
        anc_idx = (max_idx / num_gt_boxes).long() # 整除得到行数（一行有 num_gt_boxes 个，也就是锚框索引）
        box_idx = (max_idx % num_gt_boxes).long() # 取模得到列数
        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
    '''
    分 2 步匹配：
    第一步：保证 “高质量匹配” （IoU≥阈值）的锚框能分配到真实框；
    因为第一步中，可能有的行 IoU 最大值也低于阈值，
    第二步：兜底，确保每个真实框都有至少一个锚框对应（哪怕 IoU 低于阈值），避免训练时部分真实框无锚框学习。
    '''
    return anchors_bbox_map
    # anchors_bbox_map 是一维张量；索引序号是锚框序号；值是真实边缘框序号 ✓

#@save
# 计算锚框偏移量（并不简单，是有特殊公式计算的）
def offset_boxes(anchors, assigned_bb, eps=1e-6):
    """对锚框偏移量的转换"""
    c_anc = d2l.box_corner_to_center(anchors)
    c_assigned_bb = d2l.box_corner_to_center(assigned_bb)
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    offset_wh = 5 * torch.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    offset = torch.cat([offset_xy, offset_wh], axis=1)
    '''
    >>> a1 = torch.tensor(range(3))
    >>> b1 = torch.tensor(range(6, 9))
    >>> torch.cat([a1, b1])
    tensor([0, 1, 2, 6, 7, 8])
    >>> torch.cat((a1, b1))
    tensor([0, 1, 2, 6, 7, 8])
    ⇒ 这里返回的仍然是二维张量 (num_anchors, 4)
    '''
    return offset

#@save
def multibox_target(anchors, labels):
    """使用真实边界框标记锚框"""
    # anchors 张量原本形状是 (1, num_anchors, 4) (目标数量, num_anchors, 4)
    # ⇒ 压缩后，anchors 张量形状是 (num_anchors, 4)
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]
    for i in range(batch_size):
        label = labels[i, :, :] 
        # ⇒ labels 张量形状 (batch_size, num_lables, 5)
        # 5 = 1 + 4，其中 1 是标签，添加到 class_labels 列表中！
        anchors_bbox_map = assign_anchor_to_bbox(
            label[:, 1:], anchors, device)
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(
            1, 4)
        # bbox_mask 张量形状：(num_anchors) → (num_anchors, 4)
        # bbox_mask（掩码）作用：仅让「正锚框（匹配到真实框的锚框）」参与边界框偏移量的损失计算，
        # 负锚框的偏移量损失被置零。
        # 将类标签和分配的边界框坐标初始化为零
        class_labels = torch.zeros(num_anchors, dtype=torch.long,
                                   device=device)
        assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32,
                                  device=device)
        # 使用真实边界框来标记锚框的类别。
        # 如果一个锚框没有被分配，标记其为背景（值为零）
        indices_true = torch.nonzero(anchors_bbox_map >= 0)
        # 如果锚框没有被标记，那么 anchors_bbox_map[i] = -1
        # assign_anchor_to_bbox 只保证每个真实框有对应的锚框，但反之则未必！
        bb_idx = anchors_bbox_map[indices_true] 
        # indices_true 是锚框索引；bb_idx 是真实框索引
        class_labels[indices_true] = label[bb_idx, 0].long() + 1
        assigned_bb[indices_true] = label[bb_idx, 1:]
        # 上面实际上已经完成了【对锚框增添标签】的操作！
        # 偏移量转换
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1)) # 张量形状 (batch_size, num_anchors * 4)
        batch_class_labels.append(class_labels) # 张量形状 (batch_size, num_gt_boxes)
    # 将列表转换为张量
    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)

'''一个具体例子'''
ground_truth = torch.tensor([[0, 0.1, 0.08, 0.52, 0.92],
                         [1, 0.55, 0.2, 0.9, 0.88]])
anchors = torch.tensor([[0, 0.1, 0.2, 0.3], [0.15, 0.2, 0.4, 0.4],
                    [0.63, 0.05, 0.88, 0.98], [0.66, 0.45, 0.8, 0.8],
                    [0.57, 0.3, 0.92, 0.9]])

plt.figure(2)
fig = d2l.plt.imshow(img)
drawFig = plt.gcf()
set_title(drawFig, '第三步：在训练集中标注锚框')

show_bboxes(fig.axes, ground_truth[:, 1:] * bbox_scale, ['dog', 'cat'], 'k')
show_bboxes(fig.axes, anchors * bbox_scale, ['0', '1', '2', '3', '4'])

labels = multibox_target(anchors.unsqueeze(dim=0),
                         ground_truth.unsqueeze(dim=0))

print(f'labels[2]: {labels[2]}')
print(f'labels[1]: {labels[1]}')
print(f'labels[0]: {labels[0]}')



'''使用非极大值抑制预测边界框'''
#@save
# inverse 本来是【反】的意思；这里应该是【还原】的意思
def offset_inverse(anchors, offset_preds):
    """根据带有预测偏移量的锚框来预测边界框"""
    # 张量形状 anchors 和 offset_preds 都是 (num_anchors, 4)
    # 锚框格式转换：左上+右下 → 中心坐标+宽高
    anc = d2l.box_corner_to_center(anchors)
    # 偏移量x和y → 除以归一化系数10 → 乘以锚框宽高（相对锚框尺寸）→ 加锚框原始中心
    pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
    # 偏移量w和h → 除以归一化系数5 → 指数还原（训练时做了对数变换）→ 乘锚框原始宽高
    pred_bbox_wh = torch.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]
    pred_bbox = torch.cat((pred_bbox_xy, pred_bbox_wh), axis=1)
    predicted_bbox = d2l.box_center_to_corner(pred_bbox)
    return predicted_bbox # 张量形状 predicted_bbox (num_anchors, 4)
    # /10和/5是训练阶段为了稳定训练设置的归一化系数，预测时需要反向还原；
    # torch.exp是因为训练时对宽高的偏移做了对数变换（避免宽高为负）

#@save
def nms(boxes, scores, iou_threshold):
    """对预测边界框的置信度进行排序（综合 offset_inverse 和 nms）"""
    # boxes 其实是【预测边界框】(num_anchors, 4)
    # scores 是一维张量 (num_anchors)；对一维张量降序后，得到一维张量 B (num_anchors)
    B = torch.argsort(scores, dim=-1, descending=True)
    keep = []  # 保留预测边界框的指标
    while B.numel() > 0:
        i = B[0]
        keep.append(i)
        if B.numel() == 1: break # 如果只剩最后一个锚框，那就不用计算 IoU 值了
        # boxes[i, :].reshape(-1, 4)：将单个框转为 (1,4) 格式（符合box_iou输入要求）
        # boxes[B[1:], :].reshape(-1, 4)：剩余框转为 (n,4) 格式
        iou = box_iou(boxes[i, :].reshape(-1, 4),
                      boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
        # ⇒ iou 是一维张量 (len(B[1:]))
        # 筛选出与当前框IoU ≤ 阈值的框（重叠度低，需要保留继续处理）
        # 难道当前框 > 阈值的框就丢掉吗？
        # 并不是当前框 > 阈值的框直接丢掉！
        # 而是当前框 > 阈值，说明和第一个框选的是同一个目标。
        # 所以就是一个目标只选一个最好的框，把其他圈到本目标的框直接丢掉！
        # 如果当前框 IoU 值太小，可能是其他目标
        inds = torch.nonzero(iou <= iou_threshold).reshape(-1)
        B = B[inds + 1] # 更新待处理框列表B：只保留重叠度低的框
    return torch.tensor(keep, device=boxes.device)

#@save
def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5,
                       pos_threshold=0.009999999):
    """使用非极大值抑制来预测边界框"""
    # class_probs 是 classes_probability（类别概率）
    device, batch_size = cls_probs.device, cls_probs.shape[0]
    # 去除冗余的维度 anchors (1, num_anchors, 4) → (num_anchors, 4)
    anchors = anchors.squeeze(0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        # 张量形状 cls_probs (batch_size, num_classes, num_anchors)
        '''这里的样本到底是什么？为什么 anchors 张量第 0 维度一直为 1，
        而 cls_probs 第 0 维度却是 batch_size？因为锚框与批量无关，但为什么无关呢？
        
        答：这里的样本就是图片。
        至于为什么锚框与批量无关，是因为所有图片的锚框位置、大小完全一致
        （比如第 1 张和第 10 张图片的第 100 个锚框，坐标都是一样的）；
        不需要为每个样本单独存储锚框，只需要 1 份锚框（第 0 维度为 1）即可供整个批次复用；'''
        # 步骤1：提取当前样本的类别概率、偏移量。
        # 这一步 offset_pred (batch_size, num_anchors * 4) → (batch_size, num_anchors, 4)
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        # 步骤2：取每个锚框的最高置信度（排除背景类cls_prob[1:]）和对应类别 ID
        conf, class_id = torch.max(cls_prob[1:], 0) 
        # 反应知识：这是指定维度的 max 函数，会同时返回索引；而且 keepdim=False
        # 步骤3：还原预测边界框
        predicted_bb = offset_inverse(anchors, offset_pred)
        # 步骤4：执行NMS，筛选保留的边界框索引
        # 另外，从下面代码可看出 nms 返回的 keep 张量是一维张量
        keep = nms(predicted_bb, conf, nms_threshold)

        # 步骤5：标记非保留的锚框为背景（class_id=-1）
        all_idx = torch.arange(num_anchors, dtype=torch.long, device=device)
        combined = torch.cat((keep, all_idx)) # 把要 keep 的 idx 和 所有的 idx 拼接在一起！
        uniques, counts = combined.unique(return_counts=True)
        non_keep = uniques[counts == 1] # counts == 1 是布尔张量
        all_id_sorted = torch.cat((keep, non_keep)) # keep, non_keep, all_id_sorted 都是一维张量
        class_id[non_keep] = -1
        class_id = class_id[all_id_sorted]
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
        
        # 步骤6：过滤低置信度框（低于pos_threshold也标记为背景）
        below_min_idx = (conf < pos_threshold)
        class_id[below_min_idx] = -1 # 又是布尔张量索引！
        conf[below_min_idx] = 1 - conf[below_min_idx]

        # 步骤7：拼接结果（类别 ID + 置信度 + 边界框坐标）
        pred_info = torch.cat((class_id.unsqueeze(1), # 未增加维度前，class_id 是一维张量 (num_anchors)
                               conf.unsqueeze(1),     # 未增加维度前，conf 是一维张量 (num_anchors) 因为 keepdim=False
                               predicted_bb), dim=1)  # predicted_bb (num_anchors, 4)
        out.append(pred_info)
    return torch.stack(out)

anchors = torch.tensor([[0.1, 0.08, 0.52, 0.92], [0.08, 0.2, 0.56, 0.95],
                      [0.15, 0.3, 0.62, 0.91], [0.55, 0.2, 0.9, 0.88]])
offset_preds = torch.tensor([0] * anchors.numel())
cls_probs = torch.tensor([[0] * 4,  # 背景的预测概率
                      [0.9, 0.8, 0.7, 0.1],  # 狗的预测概率
                      [0.1, 0.2, 0.3, 0.9]])  # 猫的预测概率

plt.figure(3)
fig = d2l.plt.imshow(img)
drawFig = plt.gcf()
set_title(drawFig, '第四步：加入预测概率，但没有 nms 限制的锚框')

show_bboxes(fig.axes, anchors * bbox_scale,
            ['dog=0.9', 'dog=0.8', 'dog=0.7', 'cat=0.9'])


output = multibox_detection(cls_probs.unsqueeze(dim=0),
                            offset_preds.unsqueeze(dim=0),
                            anchors.unsqueeze(dim=0),
                            nms_threshold=0.5)
print(f'output: {output}')


plt.figure(4)
fig = d2l.plt.imshow(img)
drawFig = plt.gcf()
set_title(drawFig, '第五步：加入预测概率，并且有 nms 限制的锚框')


for i in output[0].detach().numpy():
    if i[0] == -1:
        continue
    label = ('dog=', 'cat=')[int(i[0])] + str(i[1])
    show_bboxes(fig.axes, [torch.tensor(i[2:]) * bbox_scale], label)

plt.show()