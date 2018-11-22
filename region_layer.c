/**********************参考文章链接如下*********************************/
//https://www.jianshu.com/p/e06ff630accf
//http://www.mamicode.com/info-detail-1974310.html
//https://blog.csdn.net/xueyingxue001/article/details/72831551
//http://www.voidcn.com/article/p-upajeqpz-brz.html
#include "region_layer.h"
#include "activations.h"
#include "blas.h"
#include "box.h"
#include "cuda.h"
#include "utils.h"

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

layer make_region_layer(int batch, int w, int h, int n, int classes, int coords)
{
    layer l = {0};//初始化层
    l.type = REGION;//说明这是 REGION 层

    l.n = n;//一个 cell 提出 n 个box
    l.batch = batch;
    l.h = h;//图片被分为多少个cell=最后特征图大小->例如：13*13，l.h=l.w=13
    l.w = w;
    l.c = n*(classes + coords + 1);// 有 n * (classes + 4 + 1) 个channel
    l.out_w = l.w;
    l.out_h = l.h;
    l.out_c = l.c;
    l.classes = classes;//有多少物体类别
    l.coords = coords;
    l.cost = calloc(1, sizeof(float));//calloc(n,size):在内存的动态存储区中分配n个长度为size的连续空间，函数返回一个指向分配起始地址的指针
    l.biases = calloc(n*2, sizeof(float));
    l.bias_updates = calloc(n*2, sizeof(float));
    l.outputs = h*w*n*(classes + coords + 1);//输出 feature map 大小,e.g. whole boxes 13*13*5*(20+4+1) 
    l.inputs = l.outputs;
    l.truths = 30*(l.coords + 1);// 一张图片含有的truth box参数的个数（30表示一张图片最多有30个ground truth box，写死的）
    l.delta = calloc(batch*l.outputs, sizeof(float));//batch*outputs
    l.output = calloc(batch*l.outputs, sizeof(float));
    int i;
    for(i = 0; i < n*2; ++i){
        l.biases[i] = .5;
    }//biases是预先设定的长宽先验值，从聚类中得到,若不提供先验值，则全部初始化为0.5。

    l.forward = forward_region_layer;
    l.backward = backward_region_layer;
#ifdef GPU
    l.forward_gpu = forward_region_layer_gpu;
    l.backward_gpu = backward_region_layer_gpu;
    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
    l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
#endif

    fprintf(stderr, "detection\n");
    srand(0);

    return l;
}

void resize_region_layer(layer *l, int w, int h)
{//将 feature map resize，调整其宽长以及宽长所影响的其它参数，目测只能往大里resize，不然数据丢失
    l->w = w;//新宽度
    l->h = h;//新高度
    //更改w，h影响到的其它变量值改变
    l->outputs = h*w*l->n*(l->classes + l->coords + 1);
    l->inputs = l->outputs;

    l->output = realloc(l->output, l->batch*l->outputs*sizeof(float));
    l->delta = realloc(l->delta, l->batch*l->outputs*sizeof(float));

#ifdef GPU
    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);

    l->delta_gpu =     cuda_make_array(l->delta, l->batch*l->outputs);
    l->output_gpu =    cuda_make_array(l->output, l->batch*l->outputs);
#endif
}

box get_region_box(float *x, float *biases, int n, int index, int i, int j, int w, int h, int stride)
{
    /*
    bx=σ(tx)+cx
    by=σ(ty)+cy
    bw=pw*e^tw
    bh=ph*e^th
    bx,by,bw,bh是预测边框的中心和宽高
    cx,cy 是当前网格左上角到图像左上角的距离，要先将网格大小归一化，即令一个网格的宽=1，高=1。pw,ph是先验框的宽和高。
    σ是sigmoid函数。
    tx,ty,tw,th,to是要学习的参数，分别用于预测边框的中心和宽高，以及置信度
    */
    box b;
    b.x = (i + x[index + 0*stride]) / w;//占整体的比例，此处w，h为最后一层feature的宽高
    b.y = (j + x[index + 1*stride]) / h;
    b.w = exp(x[index + 2*stride]) * biases[2*n]   / w;
    b.h = exp(x[index + 3*stride]) * biases[2*n+1] / h;
    return b;
}
/***********************计算box loss的梯度*****************************/
float delta_region_box(box truth, float *x, float *biases, int n, int index, int i, int j, int w, int h, float *delta, float scale, int stride)
{
    box pred = get_region_box(x, biases, n, index, i, j, w, h, stride);
    float iou = box_iou(pred, truth);

    float tx = (truth.x*w - i);
    float ty = (truth.y*h - j);
    float tw = log(truth.w*w / biases[2*n]);
    float th = log(truth.h*h / biases[2*n + 1]);

    delta[index + 0*stride] = scale * (tx - x[index + 0*stride]);
    delta[index + 1*stride] = scale * (ty - x[index + 1*stride]);
    delta[index + 2*stride] = scale * (tw - x[index + 2*stride]);
    delta[index + 3*stride] = scale * (th - x[index + 3*stride]);
    return iou;
}
/***********************计算class loss的梯度*****************************/
void delta_region_class(float *output, float *delta, int index, int class, int classes, tree *hier, float scale, int stride, float *avg_cat)
{
    int i, n;
    if(hier){// word tree类型的loss
        float pred = 1;
        while(class >= 0){
            pred *= output[index + stride*class];
            int g = hier->group[class];
            int offset = hier->group_offset[g];
            for(i = 0; i < hier->group_size[g]; ++i){
                delta[index + stride*(offset + i)] = scale * (0 - output[index + stride*(offset + i)]);
            }
            delta[index + stride*class] = scale * (1 - output[index + stride*class]);

            class = hier->parent[class];
        }
        *avg_cat += pred;
    } else {//这就是普通的loss
        for(n = 0; n < classes; ++n){
	    // 把所有 class 的预测概率与真实 class 的 0/1 的差 * scale，然后存入 l.delta 里相应 class 序号的位置
            delta[index + stride*n] = scale * (((n == class)?1 : 0) - output[index + stride*n]);
            if(n == class) *avg_cat += output[index + stride*n];
        }
    }
}

float logit(float x)
{
    return log(x/(1.-x));
}

float tisnan(float x)
{
    return (x != x);
}

int entry_index(layer l, int batch, int location, int entry)
{//索引定位函数，给出第几个batch，第几个box，和要索引的是哪个特征元素，即可得其坐标位置
    //l.output的数据组织方式如下：（一维）
    //把所有的grid cell预测的若干个box来作为分割段：
    //第一个数据段是所有grid cell的第0个box;
    //第二个数据段是所有grid cell的第1个box;
    //依次类推;
    //在每个数据段内，数据的排列又是这样的：
    //先排所有box的x数据，然后是y，接着是w和h和confidence。
    //最后的样子,假设output feature map 是 2x2的，每个cell预测2个box：
    //xxxxyyyywwwwhhhhccccxxxxyyyywwwwhhhhcccc
    /*最后的feature map是一个很多层的立方体，长宽分别为l.w*l.h，深度为: 
        depth=l.n*(l.coords+l.classes+1) 
        l.n是每个grid cell预测的box的数量; 
        l.coords是坐标数量，为4。（源码中还有不是4的if语句，未深究） 
        l.classes就是要检测的物体的类别数量; 
        数字1就是表示confidence的数值。
        相当于把这个深度为depth的立方体，切成depth个面，然后把这个面，拉成一条；一条接一条就成了l.output输出的数据形式了
*/
    int n =   location / (l.w*l.h);
    int loc = location % (l.w*l.h);
    return batch*l.outputs + n*l.w*l.h*(l.coords+l.classes+1) + entry*l.w*l.h + loc;
}
/*****************************计算loss******************************************/
/*
源码中计算loss的步骤：

1.计算不包含目标的anchors的iou损失

2.12800样本之前计算未预测到target的anchors的梯度

3.针对于每一个target，计算最接近的anchor的coord梯度

4.计算包含目标的anchors的iou损失

5.计算类别预测的损失和梯度
*/
void forward_region_layer(const layer l, network net)
{//这个函数就是前馈计算损失函数的
    int i,j,b,t,n;
    //参见network.c，每次执行前馈时，都把 net.input 指向当前层的输出，memcpy(void *dest, void *src, unsigned int count);
    memcpy(l.output, net.input, l.outputs*l.batch*sizeof(float));//把最后的卷积结果拷贝过来

#ifndef GPU
    //把所有的x，y，confidence使用逻辑斯蒂函数映射到（0,1）
    for (b = 0; b < l.batch; ++b){//batch
        for(n = 0; n < l.n; ++n){//boxes
            int index = entry_index(l, b, n*l.w*l.h, 0);//定位bth batch nth box's x,y 的数组索引
            activate_array(l.output + index, 2*l.w*l.h, LOGISTIC);
            index = entry_index(l, b, n*l.w*l.h, 4);
            if(!l.background) activate_array(l.output + index,   l.w*l.h, LOGISTIC);
        }
    }
    //下面是计算class概率的两种方式，根据设置选其中一个
    if (l.softmax_tree){
        int i;
        int count = 5;
        for (i = 0; i < l.softmax_tree->groups; ++i) {
            int group_size = l.softmax_tree->group_size[i];
            softmax_cpu(net.input + count, group_size, l.batch, l.inputs, l.n*l.w*l.h, 1, l.n*l.w*l.h, l.temperature, l.output + count);
            count += group_size;
        }
    } else if (l.softmax){
        int index = entry_index(l, 0, 0, l.coords + !l.background);//apply softmax to n classes scores
        //通过softmax_cpu函数(in blas.c),计算softmax to each batches' each boxes' position's softmax
        softmax_cpu(net.input + index, l.classes + l.background, l.batch*l.n, l.inputs/l.n, l.w*l.h, 1, l.w*l.h, 1, l.output + index);
    }
#endif
    memset(l.delta, 0, l.outputs * l.batch * sizeof(float));//分配内存并初始化梯度,梯度清零
    //（memset将l.delta指向内存中的后l.outputs * l.batch * sizeof(float)个字节的内容全部设置为0）
    if(!net.train) return;
    float avg_iou = 0;//平均 IOU
    float recall = 0;//平均召回率
    float avg_cat = 0;// 平均的类别辨识率
    float avg_obj = 0;//有物体的 predict平均
    float avg_anyobj = 0;//所有predict 平均 indicate all boxes having objects' probability
    int count = 0; // 该batch内检测的target数
    int class_count = 0;
    *(l.cost) = 0;
    for (b = 0; b < l.batch; ++b) {// 遍历batch内数据
        if(l.softmax_tree){//计算softmax_tree形式下的 loss-begin
            int onlyclass = 0;
            for(t = 0; t < 30; ++t){
                box truth = float_to_box(net.truth + t*(l.coords + 1) + b*l.truths, 1);
                if(!truth.x) break;
                int class = net.truth[t*(l.coords + 1) + b*l.truths + 4];
                float maxp = 0;
                int maxi = 0;
                if(truth.x > 100000 && truth.y > 100000){
                    for(n = 0; n < l.n*l.w*l.h; ++n){
                        int class_index = entry_index(l, b, n, 5);
                        int obj_index = entry_index(l, b, n, 4);
                        float scale =  l.output[obj_index];
                        l.delta[obj_index] = l.noobject_scale * (0 - l.output[obj_index]);
                        float p = scale*get_hierarchy_probability(l.output + class_index, l.softmax_tree, class, l.w*l.h);
                        if(p > maxp){
                            maxp = p;
                            maxi = n;
                        }
                    }
                    int class_index = entry_index(l, b, maxi, 5);
                    int obj_index = entry_index(l, b, maxi, 4);
                    delta_region_class(l.output, l.delta, class_index, class, l.classes, l.softmax_tree, l.class_scale, l.w*l.h, &avg_cat);
                    if(l.output[obj_index] < .3) l.delta[obj_index] = l.object_scale * (.3 - l.output[obj_index]);
                    else  l.delta[obj_index] = 0;
                    ++class_count;
                    onlyclass = 1;
                    break;
                }
            }
            if(onlyclass) continue;
        }//计算softmax_tree形式下的loss-over
        /**************1.计算noobj时的confidence loss(遍历那 13*13 个格子后判断当期格子有无物体，然后计算 loss)**********************/
        /*
        // 实际是计算没有物体的 box 的 confidence 的 loss
		// 1， 遍历所有格子以及每个格子的 box，计算每个 box 与真实 box 的 best_iou
		// 2， 先不管三七二十一，把该 box 当成没有目标来算 confidence 的 loss 
		// 3， 如果当前 box 的 best_iou > 阈值，则说明该 box 是有物体的，于是上面哪行计算的 loss 就不算数，因此把刚才计算的 confidence 的 loss 清零。
		// 假设图片被分成了 13 * 13 个格子，那 l.h 和 l.w 就为 13
		// 于是要遍历所有的格子，因此下面就要循环 13 * 13 次*/
        for (j = 0; j < l.h; ++j) {
            for (i = 0; i < l.w; ++i) {
                for (n = 0; n < l.n; ++n) {// 每个格子会预测 5 个 boxes，因此这里要循环 5 次
                    //对每个box，找其所对应的 ground truth box
                    int box_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 0); //带入 entry_index, 由output tensor的存储格式可以知道这里是第n个anchor在(i,j)上对应box的首地址
                    //box类型里面存的都是x,y,w,h；l.output数组里面存的是tx,ty,th,tw
                    box pred = get_region_box(l.output, l.biases, n, box_index, i, j, l.w, l.h, l.w*l.h);// 在cell（i，j）上相对于anchor n的预测结果， 相对于feature map的值
                    float best_iou = 0;
                    //假设一张图片中最多包含 30 个物体，于是对每一个物体求iou
                    for(t = 0; t < 30; ++t){
                        //net.truth存放的是真实数据,其存储格式：x,y,w,h,c,x,y,w,h,c,....
                        //读取一个真实目标框，get truth_box's x, y, w, h
                        box truth = float_to_box(net.truth + t*5 + b*l.truths, 1);
                        if(!truth.x) break;//遍历完所有真实box则跳出循环
                        float iou = box_iou(pred, truth);//计算iou
                        if (iou > best_iou) {
                            best_iou = iou;//找到与当前预测box的最大iou
                        }
                    }
                    int obj_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 4);// 获得预测结果中保存 confidence 的 index
                    avg_anyobj += l.output[obj_index];// 有目标的概率
                     // 所有的predict box先不管三七二十一，直接都当做noobject来计算其损失梯度，主要是为了计算速度考虑
                    //confidence_loss
                    l.delta[obj_index] = l.noobject_scale * (0 - l.output[obj_index]);
                    if(l.background) l.delta[obj_index] = l.noobject_scale * (1 - l.output[obj_index]);//未执行
                    // 然后再做个判断，如果当box计算的best_iou > 阈值的话，则说明该 box 是有物体的，于是上面哪行计算的 loss 就不算数，因此清零
                    if (best_iou > l.thresh) {
                        l.delta[obj_index] = 0;
                    }
                    /********************2.region loss***********************/
                    //net.seen 已训练样本的个数，记录网络看了多少张图片了
                    //如果当前cell没有目标物体（即在这一块没有ground truth落入），将当前anchor的位置和大小当作“ground truth”-a
                    //将网络预测出的预测位置和a进行相减求误差，配以scale=0.01的权重计算损失，主要目的是为了在模型训练的前期更加稳定
                    /***Also, in every image many grid cells do not contain any object. 
                    This pushes the donfidence scores of thos cells towards zero, 
                    ofthen overpowering the gradient from cells that do contain objects. 
                    This can lead to model instability, causing training to diverge early on.**/
                    if(*(net.seen) < 12800){
                        // 单纯的获取“以当前格子中心”为 x, y的box作为ground truth
                        box truth = {0};// 当前cell为中心对应的第n个anchor的box
                        truth.x = (i + .5)/l.w;// cell的中点 // 对应tx=0.5
                        truth.y = (j + .5)/l.h; //ty=0.5
                        truth.w = l.biases[2*n]/l.w;//相对于feature map的大小 // tw=0
                        truth.h = l.biases[2*n+1]/l.h;//th=0
                         //将预测的 tx, ty, tw, th 和 实际box计算得出的 tx',ty', tw', th' 的差存入 l.delta
                        delta_region_box(truth, l.output, l.biases, n, box_index, i, j, l.w, l.h, l.delta, .01, l.w*l.h);
                    }
                }
            }
        }
        /*********************3.region box loss********************/
        //因此下面是“直接遍历一张图片中的所有已标记的物体的中心所在的格子，然后计算 loss”，而不是“遍历那 13*13 个格子后判断当期格子有无物体，然后计算 loss”
        /*首先遍历ground truth box，然后从所有已标记的物体（ground truth box）中心点所在的那个cell的n个pred boxes中找到IOU最大的box来计算loss*/
        for(t = 0; t < 30; ++t){
            //读取一个真实目标框，get truth_box's x, y, w, h，归一化的值
            box truth = float_to_box(net.truth + t*5 + b*l.truths, 1);

            if(!truth.x) break;
            float best_iou = 0;
            int best_n = 0;
            i = (truth.x * l.w);// 类型的强制转换，计算该truth所在的cell的i,j坐标，i为int类型
            j = (truth.y * l.h);
            //printf("%d %f %d %f\n", i, truth.x*l.w, j, truth.y*l.h);
            box truth_shift = truth;
            truth_shift.x = 0;
            truth_shift.y = 0;
            //printf("index %d %d\n",i, j);
            for(n = 0; n < l.n; ++n){// 遍历对应的cell预测出的n个anchor
                 // 即通过该cell对应的anchors与truth的iou来判断使用哪一个anchor产生的predict来回归
                int box_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);
                // 预测box，归一化的值
                box pred = get_region_box(l.output, l.biases, n, box_index, i, j, l.w, l.h, l.w*l.h);
                 //下面这几句是将truth与anchor中心对齐后，计算anchor与truch的iou
                /*bias_match标志位用来确定由anchor还是anchor对应的prediction来确定用哪个anchor产生的prediction来回归。
                如果bias_match=1,即cfg中设置，那么先用anchor与truth box的iou来选择每个cell使用哪个anchor的预测框计算损失。
                如果bias_match=0，使用每个anchor的预测框与truth box的iou选择使用哪一个anchor的预测框计算损失，
                这里我刚开始纳闷，bias_match=0计算的iou和后面rescore=1里面用的iou不是一样了吗，那delta就一直为0啊？
                其实这里在选择anchor时计算iou是在中心对齐的情况下计算的，所以和后面rescore计算的iou还是不一样的。*/
                if(l.bias_match){
                    pred.w = l.biases[2*n]/l.w;//用anchor的长、宽作为计算损失
                    pred.h = l.biases[2*n+1]/l.h;
                }
                //printf("pred: (%f, %f) %f x %f\n", pred.x, pred.y, pred.w, pred.h);
                pred.x = 0;
                pred.y = 0;//为了方便计算 iou，上面把 ground truth box 和 pred box 都平移到中心坐标为（0,0）来计算 ，也即去除中心偏移带来的影响
                float iou = box_iou(pred, truth_shift);
                if (iou > best_iou){
                    best_iou = iou;
                    best_n = n;// 最优iou对应的anchor索引，然后使用该anchor预测的predict box计算与真实box的误差
                }
            }
            //printf("%d %f (%f, %f) %f x %f\n", best_n, best_iou, truth.x, truth.y, truth.w, truth.h);
            // 根据上面的 best_n 找出 box 的 index
            int box_index = entry_index(l, b, best_n*l.w*l.h + j*l.w + i, 0);
            /*
	    在计算boxes的 w 和 h 误差时，YOLOv1中采用的是平方根以降低boxes的大小对误差的影响，而YOLOv2是直接计算，
	    但是根据ground truth的大小对权重系数进行修正：l.coord_scale * (2 - truth.w*truth.h)（这里w和h都归一化到(0,1))，
	    这样对于尺度较小的boxes其权重系数会更大一些，可以放大误差，起到和YOLOv1计算平方根相似的效果
	    */
            float iou = delta_region_box(truth, l.output, l.biases, best_n, box_index, i, j, l.w, l.h, l.delta, l.coord_scale *  (2 - truth.w*truth.h), l.w*l.h);
            if(iou > .5) recall += 1;// 如果iou> 0.5, 认为找到该目标，召回数+1
            avg_iou += iou;
            /************4.计算有object的框框的confidence loss*****************/
            //l.delta[best_index + 4] = iou - l.output[best_index + 4];
            // 根据 best_n 找出 confidence 的 index
            int obj_index = entry_index(l, b, best_n*l.w*l.h + j*l.w + i, l.coords);
            
            avg_obj += l.output[obj_index];
            // 因为运行到这里意味着该格子中有物体中心，所以该格子的confidence就是1， 而预测的confidence是l.output[obj_index]，所以根据公式有下式
            l.delta[obj_index] = l.object_scale * (1 - l.output[obj_index]);
            if (l.rescore) { //控制参数rescore，当其为1时，target取best_n的预测框与ground truth的真实IOU值（cfg文件中默认采用这种方式）
		    /*如果这个栅格中不存在一个 object，则confidence score应该为0；
		    否则的话，confidence score则为 predicted bounding box与 ground truth box之间的 IOU*/
                l.delta[obj_index] = l.object_scale * (iou - l.output[obj_index]);
            }
            if(l.background){//不执行
                l.delta[obj_index] = l.object_scale * (0 - l.output[obj_index]);
            }
            /*******************5.类别回归的 loss************************/
            int class = net.truth[t*(l.coords + 1) + b*l.truths + l.coords];// 真实类别
            if (l.map) class = l.map[class];//不执行
            int class_index = entry_index(l, b, best_n*l.w*l.h + j*l.w + i, l.coords + 1);//预测的class向量首地址
	    // 把所有 class 的预测概率与真实class的0/1的差* scale，然后存入l.delta里相应class序号的位置
            delta_region_class(l.output, l.delta, class_index, class, l.classes, l.softmax_tree, l.class_scale, l.w*l.h, &avg_cat);
            ++count;
            ++class_count;
        }
    }
    //printf("\n");
	/**********************终端输出*****************************/
    //现在，l.delta 中的每一个位置都存放了 class、confidence、x, y, w, h 的差，于是通过 mag_array 遍历所有位置，计算每个位置的平方的和后开根
    // 然后利用 pow 函数求平方
    *(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);
    printf("Region Avg IOU: %f, Class: %f, Obj: %f, No Obj: %f, Avg Recall: %f,  count: %d\n", avg_iou/count, avg_cat/class_count, avg_obj/count, avg_anyobj/(l.w*l.h*l.n*l.batch), recall/count, count);
}

void backward_region_layer(const layer l, network net)
{
    /*
       int b;
       int size = l.coords + l.classes + 1;
       for (b = 0; b < l.batch*l.n; ++b){
       int index = (b*size + 4)*l.w*l.h;
       gradient_array(l.output + index, l.w*l.h, LOGISTIC, l.delta + index);
       }
       axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, net.delta, 1);
     */
}

void correct_region_boxes(box *boxes, int n, int w, int h, int netw, int neth, int relative)
{
    int i;
    int new_w=0;
    int new_h=0;
    if (((float)netw/w) < ((float)neth/h)) {
        new_w = netw;
        new_h = (h * netw)/w;
    } else {
        new_h = neth;
        new_w = (w * neth)/h;
    }
    for (i = 0; i < n; ++i){
        box b = boxes[i];
        b.x =  (b.x - (netw - new_w)/2./netw) / ((float)new_w/netw); 
        b.y =  (b.y - (neth - new_h)/2./neth) / ((float)new_h/neth); 
        b.w *= (float)netw/new_w;
        b.h *= (float)neth/new_h;
        if(!relative){
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }
        boxes[i] = b;
    }
}
/****************获取检测到的boxes********************/
void get_region_boxes(layer l, int w, int h, int netw, int neth, float thresh, float **probs, box *boxes, int only_objectness, int *map, float tree_thresh, int relative)
{
    int i,j,n,z;
    float *predictions = l.output;
    if (l.batch == 2) {
        float *flip = l.output + l.outputs;
        for (j = 0; j < l.h; ++j) {
            for (i = 0; i < l.w/2; ++i) {
                for (n = 0; n < l.n; ++n) {
                    for(z = 0; z < l.classes + 5; ++z){
                        int i1 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + i;
                        int i2 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + (l.w - i - 1);
                        float swap = flip[i1];
                        flip[i1] = flip[i2];
                        flip[i2] = swap;
                        if(z == 0){
                            flip[i1] = -flip[i1];
                            flip[i2] = -flip[i2];
                        }
                    }
                }
            }
        }
        for(i = 0; i < l.outputs; ++i){
            l.output[i] = (l.output[i] + flip[i])/2.;
        }
    }
    for (i = 0; i < l.w*l.h; ++i){
        int row = i / l.w;
        int col = i % l.w;
        for(n = 0; n < l.n; ++n){
            int index = n*l.w*l.h + i;
            for(j = 0; j < l.classes; ++j){
                probs[index][j] = 0;
            }
            int obj_index = entry_index(l, 0, n*l.w*l.h + i, 4);
            int box_index = entry_index(l, 0, n*l.w*l.h + i, 0);
            float scale = l.background ? 1 : predictions[obj_index];//是否是物体的概率
            boxes[index] = get_region_box(predictions, l.biases, n, box_index, col, row, l.w, l.h, l.w*l.h);//计算出归一化的bbox

            int class_index = entry_index(l, 0, n*l.w*l.h + i, l.coords + !l.background);
            if(l.softmax_tree){//不考虑这个

                hierarchy_predictions(predictions + class_index, l.classes, l.softmax_tree, 0, l.w*l.h);
                if(map){
                    for(j = 0; j < 200; ++j){
                        int class_index = entry_index(l, 0, n*l.w*l.h + i, 5 + map[j]);
                        float prob = scale*predictions[class_index];
                        probs[index][j] = (prob > thresh) ? prob : 0;
                    }
                } else {
                    int j =  hierarchy_top_prediction(predictions + class_index, l.softmax_tree, tree_thresh, l.w*l.h);
                    probs[index][j] = (scale > thresh) ? scale : 0;
                    probs[index][l.classes] = scale;
                }
            } else {
                float max = 0;
                for(j = 0; j < l.classes; ++j){
                    int class_index = entry_index(l, 0, n*l.w*l.h + i, 5 + j);
                    float prob = scale*predictions[class_index];//类别概率乘以是否是物体的概率
                    probs[index][j] = (prob > thresh) ? prob : 0;
                    if(prob > max) max = prob;
                    // TODO REMOVE
                    // if (j == 56 ) probs[index][j] = 0; 
                    /*
                       if (j != 0) probs[index][j] = 0; 
                       int blacklist[] = {121, 497, 482, 504, 122, 518,481, 418, 542, 491, 914, 478, 120, 510,500};
                       int bb;
                       for (bb = 0; bb < sizeof(blacklist)/sizeof(int); ++bb){
                       if(index == blacklist[bb]) probs[index][j] = 0;
                       }
                     */
                }
                probs[index][l.classes] = max;
            }
            if(only_objectness){
                probs[index][0] = scale;
            }
        }
    }
    correct_region_boxes(boxes, l.w*l.h*l.n, w, h, netw, neth, relative);//将归一化bbox尺度映射到原始图片的尺度
}

#ifdef GPU

void forward_region_layer_gpu(const layer l, network net)
{
    copy_ongpu(l.batch*l.inputs, net.input_gpu, 1, l.output_gpu, 1);
    int b, n;
    for (b = 0; b < l.batch; ++b){
        for(n = 0; n < l.n; ++n){
            int index = entry_index(l, b, n*l.w*l.h, 0);
            activate_array_ongpu(l.output_gpu + index, 2*l.w*l.h, LOGISTIC);
            index = entry_index(l, b, n*l.w*l.h, l.coords);
            if(!l.background) activate_array_ongpu(l.output_gpu + index,   l.w*l.h, LOGISTIC);
        }
    }
    if (l.softmax_tree){
        int index = entry_index(l, 0, 0, 5);
        softmax_tree(net.input_gpu + index, l.w*l.h, l.batch*l.n, l.inputs/l.n, 1, l.output_gpu + index, *l.softmax_tree);
        /*
        // TIMING CODE
        int zz;
        int number = 1000;
        int count = 0;
        int i;
        for (i = 0; i < l.softmax_tree->groups; ++i) {
            int group_size = l.softmax_tree->group_size[i];
            count += group_size;
        }
        printf("%d %d\n", l.softmax_tree->groups, count);
        {
            double then = what_time_is_it_now();
            for(zz = 0; zz < number; ++zz){
                int index = entry_index(l, 0, 0, 5);
                softmax_tree(net.input_gpu + index, l.w*l.h, l.batch*l.n, l.inputs/l.n, 1, l.output_gpu + index, *l.softmax_tree);
            }
            cudaDeviceSynchronize();
            printf("Good GPU Timing: %f\n", what_time_is_it_now() - then);
        } 
        {
            double then = what_time_is_it_now();
            for(zz = 0; zz < number; ++zz){
                int i;
                int count = 5;
                for (i = 0; i < l.softmax_tree->groups; ++i) {
                    int group_size = l.softmax_tree->group_size[i];
                    int index = entry_index(l, 0, 0, count);
                    softmax_gpu(net.input_gpu + index, group_size, l.batch*l.n, l.inputs/l.n, l.w*l.h, 1, l.w*l.h, 1, l.output_gpu + index);
                    count += group_size;
                }
            }
            cudaDeviceSynchronize();
            printf("Bad GPU Timing: %f\n", what_time_is_it_now() - then);
        }
        {
            double then = what_time_is_it_now();
            for(zz = 0; zz < number; ++zz){
                int i;
                int count = 5;
                for (i = 0; i < l.softmax_tree->groups; ++i) {
                    int group_size = l.softmax_tree->group_size[i];
                    softmax_cpu(net.input + count, group_size, l.batch, l.inputs, l.n*l.w*l.h, 1, l.n*l.w*l.h, l.temperature, l.output + count);
                    count += group_size;
                }
            }
            cudaDeviceSynchronize();
            printf("CPU Timing: %f\n", what_time_is_it_now() - then);
        }
        */
        /*
           int i;
           int count = 5;
           for (i = 0; i < l.softmax_tree->groups; ++i) {
           int group_size = l.softmax_tree->group_size[i];
           int index = entry_index(l, 0, 0, count);
           softmax_gpu(net.input_gpu + index, group_size, l.batch*l.n, l.inputs/l.n, l.w*l.h, 1, l.w*l.h, 1, l.output_gpu + index);
           count += group_size;
           }
         */
    } else if (l.softmax) {
        int index = entry_index(l, 0, 0, l.coords + !l.background);
        //printf("%d\n", index);
        softmax_gpu(net.input_gpu + index, l.classes + l.background, l.batch*l.n, l.inputs/l.n, l.w*l.h, 1, l.w*l.h, 1, l.output_gpu + index);
    }
    if(!net.train || l.onlyforward){
        cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
        return;
    }

    float *truth_cpu = 0;
    if(net.truth_gpu){
        int num_truth = l.batch*l.truths;
        truth_cpu = calloc(num_truth, sizeof(float));
        cuda_pull_array(net.truth_gpu, truth_cpu, num_truth);
    }
    cuda_pull_array(l.output_gpu, net.input, l.batch*l.inputs);
    forward_region_layer(l, net);
    //cuda_push_array(l.output_gpu, l.output, l.batch*l.outputs);
    if(!net.train) return;
    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);
}

void backward_region_layer_gpu(const layer l, network net)
{
    int b, n;
    for (b = 0; b < l.batch; ++b){
        for(n = 0; n < l.n; ++n){
            int index = entry_index(l, b, n*l.w*l.h, 0);
            gradient_array_ongpu(l.output_gpu + index, 2*l.w*l.h, LOGISTIC, l.delta_gpu + index);
            index = entry_index(l, b, n*l.w*l.h, l.coords);
            if(!l.background) gradient_array_ongpu(l.output_gpu + index,   l.w*l.h, LOGISTIC, l.delta_gpu + index);
        }
    }
    axpy_ongpu(l.batch*l.inputs, 1, l.delta_gpu, 1, net.delta_gpu, 1);
}
#endif
/***这个函数在图像风格转换中用于把 object confidence 置为0****/
void zero_objectness(layer l)
{
    int i, n;
    for (i = 0; i < l.w*l.h; ++i){
        for(n = 0; n < l.n; ++n){
            int obj_index = entry_index(l, 0, n*l.w*l.h + i, 4);
            l.output[obj_index] = 0;
        }
    }
}
