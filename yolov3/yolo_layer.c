/**********************************************yolov3损失函数************************************************************/
//参考连接 https://blog.csdn.net/hfq0219/article/details/90141698
#include "yolo_layer.h"
#include "activations.h"
#include "blas.h"
#include "box.h"
#include "dark_cuda.h"
#include "utils.h"

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

layer make_yolo_layer(int batch, int w, int h, int n, int total, int *mask, int classes, int max_boxes)
{
    int i;
    layer l = { (LAYER_TYPE)0 };
    l.type = YOLO;

    l.n = n;// 这一层用的anchor数量，该层每个 grid 预测的框的个数，yolov3.cfg 为3
    l.total = total;// 所有的anchor数量
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = n*(classes + 4 + 1);///输入和输出相等，yolo 层是最后一层，不需要把输出传递到下一层。
    l.out_w = l.w;
    l.out_h = l.h;
    l.out_c = l.c;
    l.classes = classes;
    l.cost = (float*)calloc(1, sizeof(float));
    l.biases = (float*)calloc(total * 2, sizeof(float));// anchor的具体值
    // below 具体使用的那几个anchor
    //mask和用到第几个先验框有关，取值为0-9。比如，mask_n=1,先验框的宽和高是（biases[2*mask_n],biases[2*mask_n+1]）
    if(mask) l.mask = mask;//l.mask 里保存了 [yolo] 配置里 “mask = 0,1,2” 的数值
    //只有当cfg中没有定义mask的数值时才执行下面的else中语句 
    else{
        l.mask = (int*)calloc(n, sizeof(int));
        for(i = 0; i < n; ++i){
            l.mask[i] = i;
        }
    }
    l.bias_updates = (float*)calloc(n * 2, sizeof(float));
    l.outputs = h*w*n*(classes + 4 + 1);
    l.inputs = l.outputs;
    l.max_boxes = max_boxes;
    l.truths = l.max_boxes*(4 + 1);    // 90*(4 + 1);
    l.delta = (float*)calloc(batch * l.outputs, sizeof(float));// MSE的差
    l.output = (float*)calloc(batch * l.outputs, sizeof(float));
    for(i = 0; i < total*2; ++i){
        l.biases[i] = .5;//如果未指定 anchors，默认设置为0.5，否则在 ./src/parser.c 里会把 l.biases 的值设为 anchors 的大小
    }

    l.forward = forward_yolo_layer;
    l.backward = backward_yolo_layer;
#ifdef GPU
    l.forward_gpu = forward_yolo_layer_gpu;
    l.backward_gpu = backward_yolo_layer_gpu;
    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
    l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);

    free(l.output);
    if (cudaSuccess == cudaHostAlloc(&l.output, batch*l.outputs*sizeof(float), cudaHostRegisterMapped)) l.output_pinned = 1;
    else {
        cudaGetLastError(); // reset CUDA-error
        l.output = (float*)calloc(batch * l.outputs, sizeof(float));
    }

    free(l.delta);
    if (cudaSuccess == cudaHostAlloc(&l.delta, batch*l.outputs*sizeof(float), cudaHostRegisterMapped)) l.delta_pinned = 1;
    else {
        cudaGetLastError(); // reset CUDA-error
        l.delta = (float*)calloc(batch * l.outputs, sizeof(float));
    }
#endif

    fprintf(stderr, "yolo\n");
    srand(time(0));

    return l;
}

void resize_yolo_layer(layer *l, int w, int h)
{
    l->w = w;
    l->h = h;

    l->outputs = h*w*l->n*(l->classes + 4 + 1);
    l->inputs = l->outputs;

    if (!l->output_pinned) l->output = (float*)realloc(l->output, l->batch*l->outputs * sizeof(float));
    if (!l->delta_pinned) l->delta = (float*)realloc(l->delta, l->batch*l->outputs*sizeof(float));

#ifdef GPU
    if (l->output_pinned) {
        cudaFreeHost(l->output);
        if (cudaSuccess != cudaHostAlloc(&l->output, l->batch*l->outputs * sizeof(float), cudaHostRegisterMapped)) {
            cudaGetLastError(); // reset CUDA-error
            l->output = (float*)realloc(l->output, l->batch * l->outputs * sizeof(float));
            l->output_pinned = 0;
        }
    }

    if (l->delta_pinned) {
        cudaFreeHost(l->delta);
        if (cudaSuccess != cudaHostAlloc(&l->delta, l->batch*l->outputs * sizeof(float), cudaHostRegisterMapped)) {
            cudaGetLastError(); // reset CUDA-error
            l->delta = (float*)realloc(l->delta, l->batch * l->outputs * sizeof(float));
            l->delta_pinned = 0;
        }
    }

    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);

    l->delta_gpu =     cuda_make_array(l->delta, l->batch*l->outputs);
    l->output_gpu =    cuda_make_array(l->output, l->batch*l->outputs);
#endif
}
//获得预测的边界框
// box pred = get_yolo_box(l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, state.net.w, state.net.h, l.w*l.h)
box get_yolo_box(float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride)
{
    box b;
    // ln - natural logarithm (base = e)
    // x` = t.x * lw - i;   // x = ln(x`/(1-x`))   // x - output of previous conv-layer
    // y` = t.y * lh - i;   // y = ln(y`/(1-y`))   // y - output of previous conv-layer
                            // w = ln(t.w * net.w / anchors_w); // w - output of previous conv-layer
                            // h = ln(t.h * net.h / anchors_h); // h - output of previous conv-layer
    //(w,h) 输入图片尺寸 (lw,lh)当前特征图尺寸  b.x, b.y 为全图相对尺寸
    ////把结果分别利用feature map宽高和输入图宽高做了归一化，这就对应了我刚刚谈到的公式了
    //虽然b.w和b.h是除以输入图片大小 如416，但这是因为下面的函数中的tw和th用的是w,h=416，x,y都是针对feature map大小的
    b.x = (i + x[index + 0*stride]) / lw;//x[index + 0*stride]相当于tx
    b.y = (j + x[index + 1*stride]) / lh;
    b.w = exp(x[index + 2*stride]) * biases[2*n]   / w;//b.w b.h 将显示在结果图像上的目标宽度和高度
    b.h = exp(x[index + 3*stride]) * biases[2*n+1] / h;
    return b;
}
//计算boundbox的loss
ious delta_yolo_box(box truth, float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, float *delta, float scale, int stride, float iou_normalizer, IOU_LOSS iou_loss)
{
    ious all_ious = { 0 };
    // i - step in layer width
    // j - step in layer height
    //  Returns a box in absolute coordinates
    box pred = get_yolo_box(x, biases, n, index, i, j, lw, lh, w, h, stride);
    all_ious.iou = box_iou(pred, truth);// 计算IOU
    all_ious.giou = box_giou(pred, truth);
    // avoid nan in dx_box_iou
    if (pred.w == 0) { pred.w = 1.0; }
    if (pred.h == 0) { pred.h = 1.0; }
    if (iou_loss == MSE)    // old loss
    {
        float tx = (truth.x*lw - i);// groud truth 相对于框左上角坐标
        float ty = (truth.y*lh - j);
        float tw = log(truth.w*w / biases[2 * n]);// 因为bw = pw*exp tw.所以 tw = log(bw/w)
        float th = log(truth.h*h / biases[2 * n + 1]);
        // square error的求导
         // scale = 2 - truth.w * truth.h 
        delta[index + 0 * stride] = scale * (tx - x[index + 0 * stride]);
        delta[index + 1 * stride] = scale * (ty - x[index + 1 * stride]);
        delta[index + 2 * stride] = scale * (tw - x[index + 2 * stride]);
        delta[index + 3 * stride] = scale * (th - x[index + 3 * stride]);
    }
    else {
        // https://github.com/generalized-iou/g-darknet
        // https://arxiv.org/abs/1902.09630v2
        // https://giou.stanford.edu/
        all_ious.dx_iou = dx_box_iou(pred, truth, iou_loss);

        // jacobian^t (transpose)
        delta[index + 0 * stride] = (all_ious.dx_iou.dl + all_ious.dx_iou.dr);
        delta[index + 1 * stride] = (all_ious.dx_iou.dt + all_ious.dx_iou.db);
        delta[index + 2 * stride] = ((-0.5 * all_ious.dx_iou.dl) + (0.5 * all_ious.dx_iou.dr));
        delta[index + 3 * stride] = ((-0.5 * all_ious.dx_iou.dt) + (0.5 * all_ious.dx_iou.db));

        // predict exponential, apply gradient of e^delta_t ONLY for w,h
        delta[index + 2 * stride] *= exp(x[index + 2 * stride]);
        delta[index + 3 * stride] *= exp(x[index + 3 * stride]);

        // normalize iou weight
        delta[index + 0 * stride] *= iou_normalizer;
        delta[index + 1 * stride] *= iou_normalizer;
        delta[index + 2 * stride] *= iou_normalizer;
        delta[index + 3 * stride] *= iou_normalizer;
    }

    return all_ious;
}
// 计算分类loss
void delta_yolo_class(float *output, float *delta, int index, int class_id, int classes, int stride, float *avg_cat, int focal_loss)
{
    int n;
    // 参考公式（6）公式在https://blog.csdn.net/jmu201521121021/article/details/86658163中，已经有梯度，就只计算此类
    if (delta[index + stride*class_id]){
        /*我们知道，在YOLO_v3中类别损失函数使用的是sigmoid-loss，而不是使用softmax-loss。
        分类时使用sigmoid损失函数时，由于在使用真值框的中心点计算得到的最后一层feature map上的点位置存在量化误差，
        feature map上的点只能为整型，因此可能会存在两个靠的很近的真值框中心点计算出的位置在feature map上的坐标点位置是一样的，
        出现这种情况时，对应的class梯度已经在前一个真值框计算时计算过，而新的真值框计算class梯度时，
        没有必要将原来的class_delta全部覆盖掉，只需要更新对应class label对应的sigmoid梯度即可，
        因此这样的操作方式可能导致一个目标框的几个类别概率都比较大（即多label）。
        
        //当然，如果计算分类损失时使用softmax-loss就没必要这样做了。因为softmax计算出的类别概率是互斥的，
        不像使用sigmoid计算分类损失，因为每个类别都使用一个sigmoid计算其分类损失，他们的类别不是互斥的，
       因此可以使用代码中描述的操作方式，使用softmax-loss计算分类损失梯度时，第一部分代码可以直接忽略，让新的目标框类别梯度覆盖原来的即可。*/
        delta[index + stride*class_id] = 1 - output[index + stride*class_id];
        if(avg_cat) *avg_cat += output[index + stride*class_id];
        return;
    }
    // Focal loss
    if (focal_loss) {
        // Focal Loss
        float alpha = 0.5;    // 0.25 or 0.5
        //float gamma = 2;    // hardcoded in many places of the grad-formula

        int ti = index + stride*class_id;
        float pt = output[ti] + 0.000000000000001F;
        // http://fooplot.com/#W3sidHlwZSI6MCwiZXEiOiItKDEteCkqKDIqeCpsb2coeCkreC0xKSIsImNvbG9yIjoiIzAwMDAwMCJ9LHsidHlwZSI6MTAwMH1d
        float grad = -(1 - pt) * (2 * pt*logf(pt) + pt - 1);    // http://blog.csdn.net/linmingan/article/details/77885832
        //float grad = (1 - pt) * (2 * pt*logf(pt) + pt - 1);    // https://github.com/unsky/focal-loss

        for (n = 0; n < classes; ++n) {
            delta[index + stride*n] = (((n == class_id) ? 1 : 0) - output[index + stride*n]);

            delta[index + stride*n] *= alpha*grad;

            if (n == class_id) *avg_cat += output[index + stride*n];
        }
    }
    else {
        // default
        for (n = 0; n < classes; ++n) {//对所有类别，如果预测正确，则误差为 1-predict，否则为 0-predict
            delta[index + stride*n] = ((n == class_id) ? 1 : 0) - output[index + stride*n];// 公式（6）
            if (n == class_id && avg_cat) *avg_cat += output[index + stride*n];
        }
    }
}

static int entry_index(layer l, int batch, int location, int entry)
{
    int n =   location / (l.w*l.h);//第几个框，v3每个 grid 有3个框
    int loc = location % (l.w*l.h);//第几个 grid
    return batch*l.outputs + n*l.w*l.h*(4+l.classes+1) + entry*l.w*l.h + loc;
}

static box float_to_box_stride(float *f, int stride)
{
    box b = { 0 };
    b.x = f[0];
    b.y = f[1 * stride];
    b.w = f[2 * stride];
    b.h = f[3 * stride];
    return b;
}
/****前向
*****两个循环。


*****首先，网络的每个输出的bbox都对比groudtruth，如果IOU > ignore则不参与训练，进一步的，大于truth则计算loss，参与训练，但是cfg文件中这个值设置的是1,所以应该就是忽略后面这个进一步的了。

*****第二个循环，对每个目标，查找最合适的anchor，如果本层负责这个尺寸的anchor，就计算对应的各loss。否则忽略。*/
void forward_yolo_layer(const layer l, network_state state)
{
    int i, j, b, t, n;
    memcpy(l.output, state.input, l.outputs*l.batch * sizeof(float));//将层输入直接拷贝到层输出

#ifndef GPU
    // x,y ,confidence class通过激活函数logistic，公式(2)计算
    for (b = 0; b < l.batch; ++b) {
        for (n = 0; n < l.n; ++n) {//l.n 为一个cell中预测多少个box v3为3个
            int index = entry_index(l, b, n*l.w*l.h, 0);
            // 对 tx, ty进行logistic变换
            activate_array(l.output + index, 2 * l.w*l.h, LOGISTIC);        // x,y,
             // l.scale_x_y意义暂时不明 https://github.com/AlexeyAB/darknet/issues/3293中有提及
	    scal_add_cpu(2 * l.w*l.h, l.scale_x_y, -0.5*(l.scale_x_y - 1), l.output + index, 1);   
            //第b张图片中第1个cell中第n+1个box中第5个参数（confidence）的位置索引
	    index = entry_index(l, b, n*l.w*l.h, 4);
            // 对confidence和类别进行logistic变换
            activate_array(l.output + index, (1 + l.classes)*l.w*l.h, LOGISTIC);
        }
    }
#endif
    // 初始化梯度
    memset(l.delta, 0, l.outputs * l.batch * sizeof(float));
    if (!state.train) return;//不做训练时,直接退出,以下都是训练代码
    //float avg_iou = 0;
    float tot_iou = 0;
    float tot_giou = 0;
    float tot_iou_loss = 0;
    float tot_giou_loss = 0;
    float recall = 0;
    float recall75 = 0;
    float avg_cat = 0;
    float avg_obj = 0;
    float avg_anyobj = 0;
    int count = 0;
    int class_count = 0;
    *(l.cost) = 0;
    // 下面四个for循环是依次取n个预测的box的 x，y, w,h,confidence,class，然后依次和所有groud true 计算IOU，取IOU最大的groud true.
    for (b = 0; b < l.batch; ++b) {
        for (j = 0; j < l.h; ++j) {
            for (i = 0; i < l.w; ++i) {
                for (n = 0; n < l.n; ++n) {//n为anchor数量 v3为3
                    //内存布局: batch-anchor-xoffset-yoffset-w-h-objectness-classid
                    //xoffset,yoffset,bw,bh,objectness,classid的尺寸都是l.w * l.h
                    int box_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);//获得第b张图中i行j列的cell中第n+1个预测box的起始位置，即 box.x 的位置
                    //(i,j) 对应的第l.mask[n]个anchor的预测结果//获取pre box的x,y confidence , class //l.mask[n]得到的是选用的预设anchor的编号
                    box pred = get_yolo_box(l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, state.net.w, state.net.h, l.w*l.h);
                    float best_iou = 0;
                    int best_t = 0;// 和pre有最大IOU的groud truth的索引
                    //遍历图中所有groundtruth object ,找出和pred重合度最高的一个object
                    for (t = 0; t < l.max_boxes; ++t) {//max_boxes: cfg中配置,默认90,单个图片中目标个数最大值
			    //获得第b张图片第t个标注框的xywh
                        box truth = float_to_box_stride(state.truth + t*(4 + 1) + b*l.truths, 1);
                        int class_id = state.truth[t*(4 + 1) + b*l.truths + 4]; //获得第t个groundtruth objec的类别
                        if (class_id >= l.classes) {
                            printf(" Warning: in txt-labels class_id=%d >= classes=%d in cfg-file. In txt-labels class_id should be [from 0 to %d] \n", class_id, l.classes, l.classes - 1);
                            printf(" truth.x = %f, truth.y = %f, truth.w = %f, truth.h = %f, class_id = %d \n", truth.x, truth.y, truth.w, truth.h, class_id);
                            getchar();
                            continue; // if label contains class_id more than number of classes in the cfg-file
                        }
                        if (!truth.x) break;  // continue;//不允许groundtruth的中心点为0!!
                        float iou = box_iou(pred, truth);//计算iou,注意(box.x,box.y)是中心点位置
                        if (iou > best_iou) {
                            best_iou = iou;
                            best_t = t;
                        }
                    }
                   
                    int obj_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 4);//读取预测的objectness索引
                    avg_anyobj += l.output[obj_index];//读取预测的objectness值,avg_anyobj训练状态检测量
                    /*
                    objectness 学习
                    iou < l.ignore_thresh: 作为负样本
                    iou > l.truth_thresh: 作为正样本
                    其他:不参与训练
                    */
                     // 计算 confidence的偏差
                    l.delta[obj_index] = l.cls_normalizer * (0 - l.output[obj_index]);//默认作为负样本,目标objectness=0,误差设置为0 - l.output[obj_index]
                   // 大于IOU设置的阈值 confidence梯度设为0
                    if (best_iou > l.ignore_thresh) {
                        l.delta[obj_index] = 0;//iou较大,不能作为负样本, 清除误差
                    }
                    // yolov3这段代码不会执行，因为 l.truth_thresh值为1
                    if (best_iou > l.truth_thresh) {
                        l.delta[obj_index] = l.cls_normalizer * (1 - l.output[obj_index]);//iou足够大,是正样本,目标是objectness=1, 梯度设置为1 - l.output[obj_index]

                        int class_id = state.truth[best_t*(4 + 1) + b*l.truths + 4];//groundtruth classid
                        if (l.map) class_id = l.map[class_id];
                        int class_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 4 + 1);//预测的classid
                        //计算classid的误差(one-hot-encoding,计算方法和objectness类似,但这里还支持focal_loss)
                        delta_yolo_class(l.output, l.delta, class_index, class_id, l.classes, l.w*l.h, 0, l.focal_loss);
                        box truth = float_to_box_stride(state.truth + best_t*(4 + 1) + b*l.truths, 1);
                        //计算box误差 
                        delta_yolo_box(truth, l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, state.net.w, state.net.h, l.delta, (2 - truth.w*truth.h), l.w*l.h, l.iou_normalizer, l.iou_loss);
                    }
                }
            }
        }
        // box,class 的梯度，只计算groud truth对应的预测框的梯： 
        //先计算groud truth和所有anchor iou，然后选最大IOU的索引，若这个索引在mask里，计算梯度和loss.
        for (t = 0; t < l.max_boxes; ++t) {
            box truth = float_to_box_stride(state.truth + t*(4 + 1) + b*l.truths, 1);
            if (truth.x < 0 || truth.y < 0 || truth.x > 1 || truth.y > 1 || truth.w < 0 || truth.h < 0) {
                printf(" Wrong label: truth.x = %f, truth.y = %f, truth.w = %f, truth.h = %f \n", truth.x, truth.y, truth.w, truth.h);
            }
            int class_id = state.truth[t*(4 + 1) + b*l.truths + 4];
            if (class_id >= l.classes) continue; // if label contains class_id more than number of classes in the cfg-file

            if (!truth.x) break;  // continue;
            float best_iou = 0;
            int best_n = 0;
            i = (truth.x * l.w);// pre对应中心坐标y
            j = (truth.y * l.h);// pred 对应中心坐标x
            box truth_shift = truth;
            truth_shift.x = truth_shift.y = 0;
            for (n = 0; n < l.total; ++n) {//所有anchor都参与计算iou
                box pred = { 0 };//中心位置都变成左上角
                pred.w = l.biases[2 * n] / state.net.w;//按网络输入尺寸归一化
                pred.h = l.biases[2 * n + 1] / state.net.h;
                float iou = box_iou(pred, truth_shift);
                if (iou > best_iou) {
                    best_iou = iou;
                    best_n = n;
                }
            }
           // 判断最好AIOU对应的索引best_n 是否在mask里面，若没有，返回-1
           // 如果最合适的anchor由本层负责预测（由mask来决定）执行
            //这个函数判断上面找到的 anchor 是否是该层要预测的
	    //best_n在[0,8]之间, mask_n取值范围为0，1，2 
            int mask_n = int_index(l.mask, best_n, l.n);//只有在mask中指定的anchor进行如下计算
            if (mask_n >= 0) {//如果该 anchor 是该层要预测的
                //获得该 anchor 在层输出中对应的 box 位置
                int box_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 0);
                //和前面的计算一样,但读取了返回值iou // 计算box梯度 //计算 box 与 gt 的误差
                ious all_ious = delta_yolo_box(truth, l.output, l.biases, best_n, box_index, i, j, l.w, l.h, state.net.w, state.net.h, l.delta, (2 - truth.w*truth.h), l.w*l.h, l.iou_normalizer, l.iou_loss);

                // range is 0 <= 1
                tot_iou += all_ious.iou;
                tot_iou_loss += 1 - all_ious.iou;
                // range is -1 <= giou <= 1
                tot_giou += all_ious.giou;
                tot_giou_loss += 1 - all_ious.giou;
                //获得该 box 的 confident 的位置
                int obj_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 4);
                avg_obj += l.output[obj_index];
                // 计算梯度，公式(6)，梯度前面要加个”-“号， 1代表是真实标签 //该位置应该有正样本，所以误差为 1-predict
                l.delta[obj_index] = l.cls_normalizer * (1 - l.output[obj_index]);
                   //获得 gt 的真实类别
                int class_id = state.truth[t*(4 + 1) + b*l.truths + 4];
                if (l.map) class_id = l.map[class_id];//yolo9000
                int class_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 4 + 1);//获得 box 类别的起始位置0（80种类别）
                //和前面的计算一样,但读取了一个状态信息avg_cat
                // 计算类别的梯度
                delta_yolo_class(l.output, l.delta, class_index, class_id, l.classes, l.w*l.h, &avg_cat, l.focal_loss);

                ++count;
                ++class_count;
                //if(iou > .5) recall += 1;
                //if(iou > .75) recall75 += 1;
                //avg_iou += iou;
                if (all_ious.iou > .5) recall += 1;
                if (all_ious.iou > .75) recall75 += 1;
            }
        }
    }
    //*(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);
    //printf("Region %d Avg IOU: %f, Class: %f, Obj: %f, No Obj: %f, .5R: %f, .75R: %f,  count: %d\n", state.index, avg_iou / count, avg_cat / class_count, avg_obj / count, avg_anyobj / (l.w*l.h*l.n*l.batch), recall / count, recall75 / count, count);

    float avg_iou_loss = 0;
    // gIOU loss + MSE (objectness) loss
    if (l.iou_loss == MSE) {
        //计算损失函数，cost=sum(l.delta*l.delta)
        *(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);
    }
    else {
        // Always compute classification loss both for iou + cls loss and for logging with mse loss
        // TODO: remove IOU loss fields before computing MSE on class
        //   probably split into two arrays
        int stride = l.w*l.h;
        float* no_iou_loss_delta = (float *)calloc(l.batch * l.outputs, sizeof(float));
        memcpy(no_iou_loss_delta, l.delta, l.batch * l.outputs * sizeof(float));
        for (b = 0; b < l.batch; ++b) {
            for (j = 0; j < l.h; ++j) {
                for (i = 0; i < l.w; ++i) {
                    for (n = 0; n < l.n; ++n) {
                        int index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);
                        no_iou_loss_delta[index + 0 * stride] = 0;
                        no_iou_loss_delta[index + 1 * stride] = 0;
                        no_iou_loss_delta[index + 2 * stride] = 0;
                        no_iou_loss_delta[index + 3 * stride] = 0;
                    }
                }
            }
        }
        float classification_loss = l.cls_normalizer * pow(mag_array(no_iou_loss_delta, l.outputs * l.batch), 2);
        free(no_iou_loss_delta);

        if (l.iou_loss == GIOU) {
            avg_iou_loss = count > 0 ? l.iou_normalizer * (tot_giou_loss / count) : 0;
        }
        else {
            avg_iou_loss = count > 0 ? l.iou_normalizer * (tot_iou_loss / count) : 0;
        }
        *(l.cost) = avg_iou_loss + classification_loss;
    }
    printf("v3 (%s loss, Normalizer: (iou: %f, cls: %f) Region %d Avg (IOU: %f, GIOU: %f), Class: %f, Obj: %f, No Obj: %f, .5R: %f, .75R: %f, count: %d\n", (l.iou_loss == MSE ? "mse" : (l.iou_loss == GIOU ? "giou" : "iou")), l.iou_normalizer, l.cls_normalizer, state.index, tot_iou / count, tot_giou / count, avg_cat / class_count, avg_obj / count, avg_anyobj / (l.w*l.h*l.n*l.batch), recall / count, recall75 / count, count);
}

void backward_yolo_layer(const layer l, network_state state)
{
   axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, state.delta, 1); //直接把 l.delta 拷贝给上一层的 delta。注意 net.delta 指向 prev_layer.delta。
}
//调整预测 box 中心和大小
//得到除以了W,H后的bx,by,bw,bh，如果将这4个值分别乘以输入网络的图片的宽和高（如416*416）就可以得到bbox相对于坐标系(416*416)位置和大小了。
//但还要将相对于输入网络图片(416x416)的边框属性变换成原图按照纵横比不变进行缩放后的区域的坐标(416*312)。
void correct_yolo_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative, int letter)
{//w 和 h 是输入图片的尺寸，netw 和 neth 是网络输入尺寸
    int i;
    // 此处new_w表示输入图片经压缩后在网络输入大小的letter_box中的width,new_h表示在letter_box中的height,
	// 以1280*720的输入图片为例，在进行letter_box的过程中，原图经resize后的width为416， 那么resize后的对应height为720*416/1280,
	//所以height为234，而超过234的上下空余部分在作为网络输入之前填充了128，new_h=234
    int new_w=0;
    int new_w=0;
    int new_h=0;
    if (letter) {
        if (((float)netw / w) < ((float)neth / h)) {//新图片尺寸
            // 如果w>h说明resize的时候是以width/图像的width为resize比例的，先得到中间图的width,再根据比例得到height
            new_w = netw;
            new_h = (h * netw) / w;
        }
        else {
            new_h = neth;
            new_w = (w * neth) / h;
        }
    }
    else {
        new_w = netw;
        new_h = neth;
    }
    for (i = 0; i < n; ++i){//调整 box 相对新图片尺寸的位置
        box b = dets[i].bbox;
        // 此处的公式很不好理解还是接着上面的例子，现有new_w=416,new_h=234,因为resize是以w为长边压缩的
		// 所以x相对于width的比例不变，而b.y表示y相对于图像高度的比例，在进行这一步的转化之前，b.y表示
		// 的是预测框的y坐标相对于网络height的比值，要转化到相对于letter_box中图像的height的比值时，需要先
		// 计算出y在letter_box中的相对坐标，即(b.y - (neth - new_h)/2./neth)，再除以比例
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
        dets[i].bbox = b;
    }
}
//预测输出中置信度超过阈值的 box 个数
int yolo_num_detections(layer l, float thresh)
{
    int i, n;
    int count = 0;
    for (i = 0; i < l.w*l.h; ++i){
        for(n = 0; n < l.n; ++n){
            int obj_index  = entry_index(l, 0, n*l.w*l.h + i, 4);//获得置信度偏移位置
            if(l.output[obj_index] > thresh){
                ++count;//置信度超过阈值
            }
        }
    }
    return count;
}

void avg_flipped_yolo(layer l)
{
    int i,j,n,z;
    float *flip = l.output + l.outputs;
    for (j = 0; j < l.h; ++j) {
        for (i = 0; i < l.w/2; ++i) {
            for (n = 0; n < l.n; ++n) {
                for(z = 0; z < l.classes + 4 + 1; ++z){
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
//获得预测输出中超过阈值的 box
int get_yolo_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets, int letter)
{
    //printf("\n l.batch = %d, l.w = %d, l.h = %d, l.n = %d \n", l.batch, l.w, l.h, l.n);
    int i,j,n;
    float *predictions = l.output;
    if (l.batch == 2) avg_flipped_yolo(l);
    int count = 0;
    for (i = 0; i < l.w*l.h; ++i){
        int row = i / l.w;
        int col = i % l.w;
        for(n = 0; n < l.n; ++n){
            int obj_index  = entry_index(l, 0, n*l.w*l.h + i, 4);
            float objectness = predictions[obj_index];//置信度
            //if(objectness <= thresh) continue;    // incorrect behavior for Nan values
            if (objectness > thresh) {
                //printf("\n objectness = %f, thresh = %f, i = %d, n = %d \n", objectness, thresh, i, n);
                int box_index = entry_index(l, 0, n*l.w*l.h + i, 0);
                dets[count].bbox = get_yolo_box(predictions, l.biases, l.mask[n], box_index, col, row, l.w, l.h, netw, neth, l.w*l.h);
                dets[count].objectness = objectness;
                dets[count].classes = l.classes;
                for (j = 0; j < l.classes; ++j) {
                    int class_index = entry_index(l, 0, n*l.w*l.h + i, 4 + 1 + j);
                    float prob = objectness*predictions[class_index];
                    dets[count].prob[j] = (prob > thresh) ? prob : 0;
                }
                ++count;
            }
        }
    }
    correct_yolo_boxes(dets, count, w, h, netw, neth, relative, letter);
    return count;
}

#ifdef GPU

void forward_yolo_layer_gpu(const layer l, network_state state)
{
    //copy_ongpu(l.batch*l.inputs, state.input, 1, l.output_gpu, 1);
    simple_copy_ongpu(l.batch*l.inputs, state.input, l.output_gpu);
    int b, n;
    for (b = 0; b < l.batch; ++b){
        for(n = 0; n < l.n; ++n){
            int index = entry_index(l, b, n*l.w*l.h, 0);
            // y = 1./(1. + exp(-x))
            // x = ln(y/(1-y))  // ln - natural logarithm (base = e)
            // if(y->1) x -> inf
            // if(y->0) x -> -inf
            activate_array_ongpu(l.output_gpu + index, 2*l.w*l.h, LOGISTIC);    // x,y
            scal_add_ongpu(2 * l.w*l.h, l.scale_x_y, -0.5*(l.scale_x_y - 1), l.output_gpu + index, 1);      // scale x,y
            index = entry_index(l, b, n*l.w*l.h, 4);
            activate_array_ongpu(l.output_gpu + index, (1+l.classes)*l.w*l.h, LOGISTIC); // classes and objectness
        }
    }
    if(!state.train || l.onlyforward){
        //cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
        cuda_pull_array_async(l.output_gpu, l.output, l.batch*l.outputs);
        CHECK_CUDA(cudaPeekAtLastError());
        return;
    }

    float *in_cpu = (float *)calloc(l.batch*l.inputs, sizeof(float));
    cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
    memcpy(in_cpu, l.output, l.batch*l.outputs*sizeof(float));
    float *truth_cpu = 0;
    if (state.truth) {
        int num_truth = l.batch*l.truths;
        truth_cpu = (float *)calloc(num_truth, sizeof(float));
        cuda_pull_array(state.truth, truth_cpu, num_truth);
    }
    network_state cpu_state = state;
    cpu_state.net = state.net;
    cpu_state.index = state.index;
    cpu_state.train = state.train;
    cpu_state.truth = truth_cpu;
    cpu_state.input = in_cpu;
    forward_yolo_layer(l, cpu_state);
    //forward_yolo_layer(l, state);
    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);
    free(in_cpu);
    if (cpu_state.truth) free(cpu_state.truth);
}

void backward_yolo_layer_gpu(const layer l, network_state state)
{
    axpy_ongpu(l.batch*l.inputs, 1, l.delta_gpu, 1, state.delta, 1);
}
#endif
