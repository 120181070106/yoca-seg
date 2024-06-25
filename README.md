```
#---------------------------(predict.ipynb)------------------------#
    if mode == "predict":
        image = Image.open('4.jpg')#先自动对目录下的4.jpg文件实施基线预测
        #此外还提供的基准图片有：45是基线目标，67是大目标，89是小目标，cd是难目标
        r_image = yolo.detect_image(image, crop = crop, count=count)
        r_image.show()
        while True:
            img = input('Input image filename:')
            try:#这样自动叠加后缀就只需要输入文件名
                image = Image.open(img+'.jpg')
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image(image, crop = crop, count=count)
                r_image.show()
#---------------------------(yolo.py)------------------------#
        "model_path"        : 'model_data/b基础633.pth',#原yolov8_s换为自训的基线权
        "classes_path"      : 'model_data/voc_classes.txt',#只含0到6七类，分别分行
        "phi"               : 'n',#版本从s换为更易训、内存更小的n 
        "cuda"              : False,#cuda换为否方便推理时切无卡模式用cpu更省钱
#---------------------------(utils_fit.py)------------------------#
    if local_rank == 0:#去掉开训和完训，以及验证全程的显示
        # print('Start Train')
    if local_rank == 0:
        pbar.close()
        # print('Finish Train')
        # print('Start Validation')
        # pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    if local_rank == 0:
        pbar.close()
        # print('Finish Validation')
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(save_state_dict, os.path.join(save_dir, "p%03d.pth" % (epoch + 1)))
            # torch.save(save_state_dict, os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):#关掉最优权的保存提示，将定期权重名改为p030三个数的形式，忽略具体损失，最后精简best_epoch_weights为b，last_epoch_weights为l
            torch.save(save_state_dict, os.path.join(save_dir, "p%03d.pth" % (epoch + 1)))
            # torch.save(save_state_dict, os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))
            # print('Save best model to best_epoch_weights.pth')
            torch.save(save_state_dict, os.path.join(save_dir, "b.pth"))
        #     torch.save(save_state_dict, os.path.join(save_dir, "best_epoch_weights.pth"))
        # torch.save(save_state_dict, os.path.join(save_dir, "last_epoch_weights.pth"))
        torch.save(save_state_dict, os.path.join(save_dir, "l.pth"))
#---------------------------(callbacks.py)------------------------#
            # print("Calculate Map.")
            # print("Get map done.") #关掉算map始末的提示
#---------------------------(train.ipynb)------------------------#
if __name__ == "__main__": #精简参数行，去除多余注释
    Cuda            = True #服务器训练只能用gpu，无卡模式cpu训不了
    seed            = 11
    distributed     = False
    sync_bn         = False
    fp16            = True #设true更快些
    classes_path    = 'model_data/voc_classes.txt'
    model_path      = 'model_data/b基础633.pth' #原为'model_data/yolov8_s.pth'改成咱们自训的
    input_shape     = [640, 640]
    phi             = 'n' # 原's'改更小更高效
    pretrained      = False #有权重就不用预训练
    mosaic              = True
    mosaic_prob         = 0.5
    mixup               = True
    mixup_prob          = 0.5
    special_aug_ratio   = 0.7
    label_smoothing     = 0
    Init_Epoch          = 0
    Freeze_Epoch        = 50
    Freeze_batch_size   = 2 #原32改小
    UnFreeze_Epoch      = 300
    Unfreeze_batch_size = 4 #原16改小
    Freeze_Train        = False #预冻结前50的骨网权重，在前置网需要同时训练故设False
    Init_lr             = 1e-2
    Min_lr              = Init_lr * 0.01
    optimizer_type      = "adam"
    momentum            = 0.937
    weight_decay        = 5e-4
    lr_decay_type       = "cos"
    save_period         = 30 #每隔30轮保存下权重，整个只需10个文件，减少原10的冗余
    save_dir            = 'logs'
    eval_flag           = True
    eval_period         = 10
    num_workers         = 4
```
分割加入
```
#----------------------(backbone.py)---------------------#
import os,torchvision
from torchvision.utils import save_image

class Seg(nn.Module):#分割块
    def __init__(self, c=16):
        super(Seg, self).__init__()
        self.一=Conv(c,2*c,3,1); self.二=Conv(2*c,2*c,3,1)
        self.三=Conv(2*c,2*c,3,1)
        self.出=nn.Sequential(Conv(4*c,2*c,3,1))
        self.出0=nn.Sequential(Conv(4*c,2*c,3,1))
        self.出5=nn.Sequential(Conv(2*c,2,3,1))
    def forward(self, x):
        x1=self.一(x); x2=self.二(x1); x3=self.三(x2)#; x4=self.四(x3)
        x0=self.出0(torch.cat([x2,x3],1))
        xx=self.出(torch.cat([x1,x0],1))
        return self.出5(xx)

class Backbone(nn.Module):
    self.seg = Seg() #代替原四个dark
    self.dark2 = Tb(16); self.dark3 = Tb(32)
    self.dark4 = Tb(64); self.dark5 = Tb(128)
    def forward(self, x):
        # x = self.stem(x)
        pr = self.seg(x)
        png = nn.functional.interpolate(pr,size=(640,640),mode='bilinear')
        pr = nn.functional.softmax(pr,dim = 1).argmax(axis=1).unsqueeze(1)
        image = torchvision.transforms.ToPILImage()(pr[0].byte()*255)
        image.save('logs/pr.png') #上面连着插入#x=self.dark2(x)代替首dark里的内容
        return feat1, feat2, feat3, png#额外输出png算损

#----------------------(yolo.py)---------------------#
class YoloBody(nn.Module):
    self.stride     = torch.tensor([256 / x.shape[-2] for x in self.backbone.forward(torch.zeros(1, 3, 256, 256))])
    self.stride = self.stride[:3]
    def forward(self, x): #也就是补充", png"的内容
        feat1, feat2, feat3, png = self.backbone.forward(x)
        return dbox, cls, x, self.anchors.to(dbox.device), self.strides.to(dbox.device),png

#----------------------(utils_bbox.py，可以不用)---------------------#
class DecodeBox():
    def decode_box(self, inputs):
        dbox, cls, origin_cls, anchors, strides, png = inputs[:6] #添加png但不用

#----------------------(yolo.py)---------------------#
    "model_path"        : 'b外割756.pth', #自训的权重

#----------------------(train.py)---------------------#
    model_path      = 'b外割756.pth.pth'#'model_data/yolov8_s.pth'
    Freeze_Train        = False

#----------------------(utils_fit.py)---------------------#
import torch
import torch.nn as nn
import torch.nn.functional as F #连续插入
def Focal_Loss(inputs, target, num_classes=2, alpha=0.5, gamma=2):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)
    logpt  = -nn.CrossEntropyLoss(ignore_index=num_classes, reduction='none')(temp_inputs, temp_target.to(torch.int64))
    pt = torch.exp(logpt)
    if alpha is not None:
        logpt *= alpha
    loss = -((1 - pt) ** gamma) * logpt
    loss = loss.mean()
    return loss
def fit_one_epoch():
    # val_loss    = 0
    aux_loss    = 0
    images, bboxes, pngs = batch
        #----前向传播----------#
        dbox, cls, origin_cls, anchors, strides, png= model_train(images)
        outputs = (dbox, cls, origin_cls, anchors, strides)
        loss_seg=Focal_Loss(png,pngs.to(png.device),2)
        loss_value = loss_seg*300+yolo_loss(outputs, bboxes)
        # loss += loss_value.item()
        aux_loss += loss_seg.item()
        if local_rank == 0:
            pbar.set_postfix(**{'损':loss/(iteration+1),'辅':aux_loss/(iteration+1)*300,'lr':get_lr(optimizer)})

#----------------------(dataloader.py)---------------------#
class YoloDataset(Dataset):
#         if self.mosaic and self.rand() < self.mosaic_prob and self.epoch_now < self.epoch_length * self.special_aug_ratio:
#             lines = sample(self.annotation_lines, 3)
#             lines.append(self.annotation_lines[index])
#             shuffle(lines)
#             image, box  = self.get_random_data_with_Mosaic(lines, self.input_shape)
            
#             if self.mixup and self.rand() < self.mixup_prob:
#                 lines           = sample(self.annotation_lines, 1)
#                 image_2, box_2  = self.get_random_data(lines[0], self.input_shape, random = self.train)
#                 image, box      = self.get_random_data_with_MixUp(image, box, image_2, box_2)
#         else:
        image, png, box      = self.get_random_data(self.annotation_lines[index], self.input_shape, random = self.train) #取消马赛克增强环节，此句加入标签
        # box         = np.array(box, dtype=np.float32)
        png         = np.array(png)
        png[png >= self.num_classes] = self.num_classes
        return image, png, labels_out
    def get_random_data():
        # image   = cvtColor(image)
        label = Image.open(line[0].replace("JPEGImages", "SegmentationClass").replace(".jpg", ".png"))
        label   = Image.fromarray(np.array(label))
        # if not random:
        #     image_data  = np.array(new_image, np.float32)
            label       = label.resize((nw,nh), Image.NEAREST)
            new_label   = Image.new('L', [w, h], (0))
            new_label.paste(label, ((w-nw)//2, (h-nh)//2))
            return image_data, new_label, box #将分割掩码作为标签
        # image = image.resize((nw,nh), Image.BICUBIC)
        label = label.resize((nw,nh), Image.NEAREST)
        #   将图像多余的部分加上灰条       -------以此句为参照
        # new_image = Image.new('RGB', (w,h), (128,128,128))
        new_label = Image.new('L', (w,h), (0))
        # new_image.paste(image, (dx, dy))
        new_label.paste(label, (dx, dy))
        # image = new_image
        label = new_label   #图片的处理后均跟上对掩码的处理
        # if flip: 
        #     image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        return image_data, label, box

def yolo_dataset_collate(batch):
    images  = []
    pngs    = [] #加入了掩码标签的分批，建议全部替换原函数
    bboxes  = []
    for i, (img, png, box) in enumerate(batch):
        # print(png.shape)
        images.append(img)
        pngs.append(png)
        box[:, 0] = i
        bboxes.append(box)
    images  = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    pngs  = torch.from_numpy(np.array(pngs)).type(torch.FloatTensor)
    bboxes  = torch.from_numpy(np.concatenate(bboxes, 0)).type(torch.FloatTensor)
    # print(images.shape,bboxes.shape,pngs.shape)
    return images, bboxes, pngs
```