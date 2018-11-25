# yolov2源码详解
:space_invader:训练命令： **./darknet detector train cfg/coco.data cfg/yolo.cfg darknet19_448.conv.23**  
:space_invader:测试图片命令： **./darknet detect test cfg/coco.data cfg/yolo.cfg yolo.weights data/dog.jpg**
## 1.训练部分
### 1.1 darknet.c->main()
首先./darknet是程序的名称，主函数应该在darknet.c源文件中，其main()函数中，要求输入的参数不少于2个.  
根据不同命令参数*argv[1]* 进入不同的调用方法  
**在训练过程中：** *argv[1]* 即第二个参数是**“detector”**。 
因此，程序进入 **run_detector(argc, argv);**  ,同时把argn和argv传递过去了。
[代码定位](https://github.com/pjreddie/darknet/blob/56d69e73aba37283ea7b9726b81afd2f79cd1134/examples/darknet.c#L417)
``` c
} else if (0 == strcmp(argv[1], "detector")){
        run_detector(argc, argv);
```
### 1.2 detector.c->run_detector()
**detector.c** 中 **run_detector()** 函数，*argv[2]* 为"train"   
**因为是训练过程，所以进入train_detector()函数。** [代码定位](https://github.com/pjreddie/darknet/blob/56d69e73aba37283ea7b9726b81afd2f79cd1134/examples/detector.c#L655)   
``` c
if(0==strcmp(argv[2], "test")) test_detector(datacfg, cfg, weights, filename, thresh, hier_thresh, outfile, fullscreen);
    else if(0==strcmp(argv[2], "test2")) test_detector2(datacfg, cfg, weights, filename, thresh, hier_thresh, outfile, fullscreen);
    else if(0==strcmp(argv[2], "train")) train_detector(datacfg, cfg, weights, gpus, ngpus, clear);
    else if(0==strcmp(argv[2], "valid")) validate_detector(datacfg, cfg, weights, outfile);
    else if(0==strcmp(argv[2], "valid2")) validate_detector_flip(datacfg, cfg, weights, outfile);
    else if(0==strcmp(argv[2], "recall")) validate_detector_recall(cfg, weights);
    else if(0==strcmp(argv[2], "demo")) {
        list *options = read_data_cfg(datacfg);
        int classes = option_find_int(options, "classes", 20);
        char *name_list = option_find_str(options, "names", "data/names.list");
        char **names = get_labels(name_list);
        demo(cfg, weights, thresh, cam_index, filename, names, classes, frame_skip, prefix, avg, hier_thresh, width, height, fps, fullscreen);
    }
```
### 1.3 detector.c->train_detector()
[代码定位](https://github.com/pjreddie/darknet/blob/56d69e73aba37283ea7b9726b81afd2f79cd1134/examples/detector.c#L5)  
``` c
/*
** 图像检测网络训练函数（针对图像检测的网络训练）
** 输入： datacfg     训练数据描述信息文件路径及名称
**       cfgfile     神经网络结构配置文件路径及名称
**       weightfile  预训练参数文件路径及名称
**       gpus        GPU卡号集合（比如使用1块GPU，那么里面只含0元素，默认使用0卡号GPU；如果不使用GPU，那么为空指针）
**       ngpus       使用GPUS块数，使用一块GPU和不使用GPU时，nqpus都等于1
**       clear       
** 说明：关于预训练参数文件weightfile，
*/
void train_detector(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear)
{
    //读取相应数据文件  
    list *options = read_data_cfg(datacfg);
   /*从options找出训练图片路径信息，如果没找到，默认使用"data/train.list"路径下的图片信息（train.list含有标准的信息格式：<object-class> <x> <y> <width> <height>）。该文件可以由darknet提供的scripts/voc_label.py根据自行在网上下载的voc数据集生成，所以说是默认路径，其实也需要使用者自行调整，也可以任意命名，不一定要为train.list*/
    char *train_images = option_find_str(options, "train", "data/train.list");//train_images将含有训练图片中所有图片的标签以及定位信息
    
    char *backup_directory = option_find_str(options, "backup", "/backup/");
    /*srand函数是随机数发生器的初始化函数。srand和rand()配合使用产生伪随机数序列。
    rand函数在产生随机数前，需要系统提供的生成伪随机数序列的种子，rand根据这个种子的值产生一系列随机数。
    如果系统提供的种子没有变化，每次调用rand函数生成的伪随机数序列都是一样的。*/
    srand(time(0));
    char *base = basecfg(cfgfile); //读取网络配置文件 
    printf("%s\n", base);
    float avg_loss = -1;
    // 构建网络：用多少块GPU，就会构建多少个相同的网络（不使用GPU时，ngpus=1）
    network *nets = calloc(ngpus, sizeof(network));

    srand(time(0));
    int seed = rand();
    int i;
    //for循环次数为ngpus，使用多少块GPU，就循环多少次（不使用GPU时，ngpus=1，也会循环一次），每一次循环都会构建一个相同的神经网络，如果提供了初始训练参数，也会为每个网络导入相同的初始训练参数
    for(i = 0; i < ngpus; ++i){
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[i]);
#endif
        nets[i] = parse_network_cfg(cfgfile);//1.3.1解析网络结构，包括训练参数，层数，各层的类别、参数，各层的输入大小等。
        //如果命令包含权重文件，则装载权重文件
        if(weightfile){
            load_weights(&nets[i], weightfile);//1.3.2
        }
        //清空记录训练次数
        if(clear) *nets[i].seen = 0;
        nets[i].learning_rate *= ngpus;
    }
    srand(time(0));
    network net = nets[0];
    //一次载入到显存的图片数量 
    int imgs = net.batch * net.subdivisions * ngpus;
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    data train, buffer;
    //网络的最后一层，如region层(最后一层的索引号是net.n-1)
    layer l = net.layers[net.n - 1];

    int classes = l.classes;
    //抖动产生额外数据 
    float jitter = l.jitter;
    //得到训练数据路径
    list *plist = get_paths(train_images);
    //int N = plist->size;
    char **paths = (char **)list_to_array(plist);

    load_args args = {0};
    args.w = net.w;
    args.h = net.h;
    args.paths = paths;
    args.n = imgs;//一次加载的数据量
    args.m = plist->size;//m是带训练图片总的数据量
    args.classes = classes;
    args.jitter = jitter;
    args.num_boxes = l.max_boxes;//默认是30个
    args.d = &buffer;
    args.type = DETECTION_DATA;
    args.threads = 8;
   
   //数据扩增，角度，曝光，饱和，灰度
    args.angle = net.angle;
    args.exposure = net.exposure;
    args.saturation = net.saturation;
    args.hue = net.hue;
    /*返回线程ID，其类型为pthread_t，去创建一个执行什么的线程，但是暂时还没有启动这个线程吧。load_data去加载数据的，load_data创建多线程*/
    //n张图片以及图片上的truth box会被加载到buffer.X,buffer.y里面去
    pthread_t load_thread = load_data(args);
    clock_t time;
    int count = 0;
    //while(i*imgs < N*120){
    while(get_current_batch(net) < net.max_batches){
        if(l.random && count++%10 == 0){//l.random决定是否要多尺度训练，如果多尺度训练的话，每10batches改变图片大小
            printf("Resizing\n");
            //根据训练图片的大小调节下面参数，该处图片防缩范围在320-608之间，可以根据自己图片大小更改10，10，32这三个数
            int dim = (rand() % 10 + 10) * 32;
            //最后200次迭代,图片大小为608*608（32的19倍）
            if (get_current_batch(net)+200 > net.max_batches) dim = 608;
            //int dim = (rand() % 4 + 16) * 32;
            printf("%d\n", dim);
            args.w = dim;
            args.h = dim;
            //线程相关 系统方法，可自行百度
            pthread_join(load_thread, 0);//wait for load_thread ternimate
            train = buffer;
            free_data(train);
            load_thread = load_data(args);
            //放缩网络的大小进行训练  
            for(i = 0; i < ngpus; ++i){
                resize_network(nets + i, dim, dim);
            }
            net = nets[0];
        }
        //如果不进行多尺度训练
        time=clock();
        //args.n数量的图像由args.threads个子线程加载完成,该线程会退出
        pthread_join(load_thread, 0);
        //加载完成的args.n张图像会存入到args.d中
        train = buffer;
        load_thread = load_data(args);

        printf("Loaded: %lf seconds\n", sec(clock()-time));

        time=clock();
        float loss = 0;

//训练网络
#ifdef GPU
        if(ngpus == 1){
            loss = train_network(net, train);//1.3.3开始训练
        } else {
            loss = train_networks(nets, ngpus, train, 4);
        }
#else
        loss = train_network(net, train);
#endif
        if (avg_loss < 0) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;

        i = get_current_batch(net);
        printf("%d: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(net), loss, avg_loss, get_current_rate(net), sec(clock()-time), i*imgs);

        if(i%1000==0){
#ifdef GPU
            if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
            char buff[256];
            sprintf(buff, "%s/%s.backup", backup_directory, base);
            save_weights(net, buff);
        }
        //每隔多少次保存权重
        if(i%10000==0 || (i < 1000 && i%100 == 0)){
#ifdef GPU
            if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
            save_weights(net, buff);
        }
        //这里要相当注意,train指针指向的空间来自于buffer,而buffer中的空间来自于load_data函数
	//后续逻辑中动态分配的空间,而在train被赋值为buffer以后,在下一次load_data逻辑中会
        //再次动态分配,这里一定要记得释放前一次分配的,否则指针将脱钩,内存泄漏不可避免
        free_data(train);
    }
//迭代次数达到最大值，保存最后权重
#ifdef GPU
    if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(net, buff);
}
```
进入**src/parser.c** ，解析cfg文件**parse_network_cfg()**   

### 1.3.1 parse.c->parse_network_cfg()
[section结构体定义](https://github.com/pjreddie/darknet/blob/b13f67bfdd87434e141af532cdb5dc1b8369aa3b/src/parser.c#L43)  
[network结构体定义](https://github.com/pjreddie/darknet/blob/61c9d02ec461e30d55762ec7669d6a1d3c356fb2/include/darknet.h#L430)  
[size_params结构体定义](https://github.com/pjreddie/darknet/blob/b13f67bfdd87434e141af532cdb5dc1b8369aa3b/src/parser.c#L121)  
读取每一层的类型和参数，并解析。其中，如果.cfg中某一层为 **[convolutional]** ,调用**parse_convolutional(options, params)** 方法，解析卷积层
``` c
/*
*从神经网络结构参数文件中读入所有神经网络层的结构参数，存储到sections中，
*sections的每个node包含一层神经网络的所有结构参数
*/
network parse_network_cfg(char *filename)
{
    /*
    *sections是一个list，其中包含n（即网络的层数）个section，每个section包含type以及options，每个options又是一个小list。
    *注意，返回的都是空的，只有每个section中的type是各层的类别而已.sections的作用仅仅是提供给一共有多少层
    */
    list *sections = read_cfg(filename);
    
    // 获取sections的第一个节点，可以查看一下cfg/***.cfg文件，其实第一块参数（以[net]开头）不是某层神经网络的参数，
    // 而是关于整个网络的一些通用参数，比如学习率，衰减率，输入图像宽高，batch大小等，
    // 具体的关于某个网络层的参数是从第二块开始的，如[convolutional],[maxpool]...，
    // 这些层并没有编号，只说明了层的属性，但层的参数都是按顺序在文件中排好的，读入时，
    // sections链表上的顺序就是文件中的排列顺序。
    node *n = sections->front;//双向链表，前向和后项都是一个node数据结构
    if(!n) error("Config file has no sections");
    // 创建网络结构并动态分配内存：输入网络层数为sections->size - 1，sections的第一段不是网络层，而是通用网络参数
    network net = make_network(sections->size - 1);//make_network()在network.c里面，产生network这种数据结构
    
    // 所用显卡的卡号（gpu_index在cuda.c中用extern关键字声明）
    // 在调用parse_network_cfg()之前，使用了cuda_set_device()设置了gpu_index的值号为当前活跃GPU卡号
    net.gpu_index = gpu_index;
    // size_params结构体元素不含指针变量
    size_params params;//是包含了网络和训练的参数，size_params定义在parser.c中

    section *s = (section *)n->val;//.cfg文件第一部分参数type和options存储的首地址（n是一个node结构，这个结构中的val是一个void*，所以这里就是将node结构中的val强转为section*）
    list *options = s->options;//第一部分参数的值的首地址
    if(!is_network(s)) error("First section must be [net] or [network]");
    parse_net_options(options, &net);//将options中的参数，如batch,lr,decay等存入net结构体

    params.h = net.h;
    params.w = net.w;
    params.c = net.c;
    params.inputs = net.inputs;
    params.batch = net.batch;
    params.time_steps = net.time_steps;
    params.net = net;

    size_t workspace_size = 0;
    n = n->next;//[net]搞定了，接下来去下一个node
    int count = 0;
    free_section(s);
    fprintf(stderr, "layer     filters    size              input                output\n");
    while(n){
        params.index = count;
        fprintf(stderr, "%5d ", count);
        s = (section *)n->val;//同上
        options = s->options;
        layer l = {0};
        LAYER_TYPE lt = string_to_layer_type(s->type);//返回[...]中的字符串，并且变大写
        if(lt == CONVOLUTIONAL){
            l = parse_convolutional(options, params);//1.3.1.1
        }else if(lt == DECONVOLUTIONAL){
            l = parse_deconvolutional(options, params);
                。
                。
                。
        }
#ifdef GPU
            l.output_gpu = net.layers[count-1].output_gpu;
            l.delta_gpu = net.layers[count-1].delta_gpu;
#endif
        }else{
            fprintf(stderr, "Type not recognized: %s\n", s->type);
        }
        l.truth = option_find_int_quiet(options, "truth", 0);
        l.onlyforward = option_find_int_quiet(options, "onlyforward", 0);
        l.stopbackward = option_find_int_quiet(options, "stopbackward", 0);
        l.dontload = option_find_int_quiet(options, "dontload", 0);
        l.dontloadscales = option_find_int_quiet(options, "dontloadscales", 0);
        l.learning_rate_scale = option_find_float_quiet(options, "learning_rate", 1);
        l.smooth = option_find_float_quiet(options, "smooth", 0);
        option_unused(options);
        net.layers[count] = l;
        if (l.workspace_size > workspace_size) workspace_size = l.workspace_size;
        free_section(s);
        n = n->next;
        ++count;
        if(n){//这边是前面一层的输出作为下一层的输入
            params.h = l.out_h;
            params.w = l.out_w;
            params.c = l.out_c;
            params.inputs = l.outputs;
        }
    }//循环读取cfg文件中每个网络层参数到net中去   
    free_list(sections);
    layer out = get_network_output_layer(net);
    net.outputs = out.outputs;
    net.truths = out.outputs;
    if(net.layers[net.n-1].truths) net.truths = net.layers[net.n-1].truths;
    net.output = out.output;
    net.input = calloc(net.inputs*net.batch, sizeof(float));
    net.truth = calloc(net.truths*net.batch, sizeof(float));
#ifdef GPU
    net.output_gpu = out.output_gpu;
    net.input_gpu = cuda_make_array(net.input, net.inputs*net.batch);
    net.truth_gpu = cuda_make_array(net.truth, net.truths*net.batch);
#endif
    if(workspace_size){
        //printf("%ld\n", workspace_size);
#ifdef GPU
        if(gpu_index >= 0){
            net.workspace = cuda_make_array(0, (workspace_size-1)/sizeof(float)+1);
        }else {
            net.workspace = calloc(1, workspace_size);
        }
#else
	//workspace_size定义为每层中需要的workspace_size的最大值
	//feature存储在workspace中，其值为一个指针
        net.workspace = calloc(1, workspace_size);
#endif
    }
    return net;
}
```
其中，如果.cfg中某一层为 **[convolutional]** ,调用**parse_convolutional(options, params)** 方法，解析卷积层  

#### 1.3.1.1 parse.c->parse_convolutional()
parse_convolutional文件是根据cfg文件找到适宜的参数传给make_convolutional_layer，然后具体的层的开辟需要make_convolutional函数来运行。forward_convolutional_layer是运行具体的层的运算。其中，feature存在net.workspace中
``` c
convolutional_layer parse_convolutional(list *options, size_params params)
{
    int n = option_find_int(options, "filters",1);//卷积核个数
    int size = option_find_int(options, "size",1);//卷积核大小
    int stride = option_find_int(options, "stride",1);//步长
    int pad = option_find_int_quiet(options, "pad",0);//图像周围是否补0
    int padding = option_find_int_quiet(options, "padding",0);//补0的长度
    int groups = option_find_int_quiet(options, "groups", 1);
    if(pad) padding = size/2;

    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);

    int batch,h,w,c;
    h = params.h;//图片的高
    w = params.w;//图片的宽
    c = params.c;//图片的通道数
    batch=params.batch;
    if(!(h && w && c)) error("Layer before convolutional layer must output image.");
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);//BN操作
    int binary = option_find_int_quiet(options, "binary", 0);//权重二值化
    int xnor = option_find_int_quiet(options, "xnor", 0);//权重和输入二值化
    //解析卷积层
    convolutional_layer layer = make_convolutional_layer(batch,h,w,c,n,groups,size,stride,padding,activation, batch_normalize, binary, xnor, params.net->adam);//1.3.1.1.1
    layer.flipped = option_find_int_quiet(options, "flipped", 0);
    layer.dot = option_find_float_quiet(options, "dot", 0);

    return layer;
}
```
其中，会进入**src/convolutional_layer.c**文件，调用**make_convolutional_layer()** 方法，进行解析卷积层

##### 1.3.1.1.1 convolutional_layer.c->make_convolutional_layer()
[layer类型结构体定义](https://github.com/pjreddie/darknet/blob/b13f67bfdd87434e141af532cdb5dc1b8369aa3b/include/darknet.h#L119)
``` c
/* 
**  输入：batch    每个batch含有的图片数
**      h               图片高度（行数）
**      w               图片宽度（列数）
        c               输入图片通道数
        n               卷积核个数
        size            卷积核尺寸
        stride          跨度
        padding         四周补0长度
        activation      激活函数类别
        batch_normalize 是否进行BN(规范化)
        binary          是否对权重进行二值化
        xnor            是否对权重以及输入进行二值化
        adam            使用
*/
convolutional_layer make_convolutional_layer(int batch, int h, int w, int c, int n, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam)
{
    int i;
    // convolutional_layer是使用typedef定义的layer的别名
    convolutional_layer l = {0};
    l.type = CONVOLUTIONAL; // 层属性：卷积层

    l.h = h;                // 输入图像高度
    l.w = w;                // 输入图像宽度
    l.c = c;                // 输入图像通道
    l.n = n;                // 卷积核个数（即滤波器个数）
    l.binary = binary;      // 是否对权重进行二值化
    l.xnor = xnor;          // 是否对权重以及输入进行二值化
    l.batch = batch;        // 每个batch含有的图片数
    l.stride = stride;      // 跨度
    l.size = size;          // 卷积核尺寸
    l.pad = padding;        // 四周补0长度
    l.batch_normalize = batch_normalize;    // 是否进行BN(规范化)

    // 该卷积层总的权重元素（卷积核元素）个数=输入图像通道数*卷积核个数*卷积核尺寸
    // （因为一个卷积核要作用在输入图片的所有通道上，所以说是一个卷积核，实际含有的卷积核参数个数需要乘以输入图片的通道数）
    l.weights = calloc(c*n*size*size, sizeof(float));
    // 
    l.weight_updates = calloc(c*n*size*size, sizeof(float));

    // bias就是Wx+b中的b（上面的weights就是W），有多少个卷积核，就有多少个b（与W的个数一一对应，每个W的元素个数为c*size*size）
    l.biases = calloc(n, sizeof(float));
    l.bias_updates = calloc(n, sizeof(float));

    /// 该卷积层总的权重元素个数（权重元素个数等于输入数据的通道数*卷积核个数*卷积核的二维尺寸，注意因为每一个卷积核是同时作用于输入数据
    /// 的多个通道上的，因此实际上卷积核是三维的，包括两个维度的平面尺寸，以及输入数据通道数这个维度，每个通道上的卷积核参数都是独立的训练参数）
    l.nweights = c*n*size*size;
    l.nbiases = n;

    // float scale = 1./sqrt(size*size*c);
    float scale = sqrt(2./(size*size*c));
    //scale = .02;
    //for(i = 0; i < c*n*size*size; ++i) l.weights[i] = scale*rand_uniform(-1, 1);
    // 初始化权重：缩放因子*标准正态分布随机数，缩放因子等于sqrt(2./(size*size*c))，为什么取这个值呢？？
    // 此处初始化权重为正态分布，而在全连接层make_connected_layer()中初始化权重是均匀分布的。
    // TODO：个人感觉，这里应该加一个if条件语句：if(weightfile)，因为如果导入了预训练权重文件，就没有必要这样初始化了（事实上在detector.c的train_detector()函数中，
    // 紧接着parse_network_cfg()函数之后，就添加了if(weightfile)语句判断是否导入权重系数文件，如果导入了权重系数文件，也许这里初始化的值也会覆盖掉，
    // 总之这里的权重初始化的处理方式还是值得思考的，也许更好的方式是应该设置专门的函数进行权重的初始化，同时偏置也是，不过这里似乎没有考虑偏置的初始化，在make_connected_layer()中倒是有。。。）
    for(i = 0; i < c*n*size*size; ++i) l.weights[i] = scale*rand_normal();
    
    // 根据该层输入图像的尺寸、卷积核尺寸以及跨度计算输出特征图的宽度和高度
    int out_w = convolutional_out_width(l);
    int out_h = convolutional_out_height(l);
    l.out_h = out_h;        // 输出图像高度
    l.out_w = out_w;        // 输出图像宽度
    l.out_c = n;            // 输出图像通道（等于卷积核个数，有多少个卷积核，最终就得到多少张特征图，每张图是一个通道）

    l.outputs = l.out_h * l.out_w * l.out_c;    // 对应每张输入图片的所有输出特征图的总元素个数（每张输入图片会得到n也即l.out_c张特征图）
    l.inputs = l.w * l.h * l.c;                 // mini-batch中每张输入图片的像素元素个数
    // 关于上面两个参数的说明：
    // 一个mini-batch中有多张图片，每张图片可能有多个通道（彩色图有三通道），l.inputs是每张输入图片所有通道的总元素个数，
    // 而每张输入图片会有n个卷积核对其进行卷积操作，因此一张输入图片会输出n张特征图，这n张特征图的总元素个数就为l.outputs

    // l.output为该层所有的输出（包括mini-batch所有输入图片的输出）
    l.output = calloc(l.batch*l.outputs, sizeof(float));
    l.delta  = calloc(l.batch*l.outputs, sizeof(float));

    // 卷积层三种指针函数，对应三种计算：前向，反向，更新
    l.forward = forward_convolutional_layer;
    l.backward = backward_convolutional_layer;
    l.update = update_convolutional_layer;
    if(binary){
        l.binary_weights = calloc(c*n*size*size, sizeof(float));
        l.cweights = calloc(c*n*size*size, sizeof(char));
        l.scales = calloc(n, sizeof(float));
    }
    if(xnor){
        l.binary_weights = calloc(c*n*size*size, sizeof(float));
        l.binary_input = calloc(l.inputs*l.batch, sizeof(float));
    }

    if(batch_normalize){
        l.scales = calloc(n, sizeof(float));
        l.scale_updates = calloc(n, sizeof(float));
        for(i = 0; i < n; ++i){
            l.scales[i] = 1;
        }

        l.mean = calloc(n, sizeof(float));
        l.variance = calloc(n, sizeof(float));

        l.mean_delta = calloc(n, sizeof(float));
        l.variance_delta = calloc(n, sizeof(float));

        l.rolling_mean = calloc(n, sizeof(float));
        l.rolling_variance = calloc(n, sizeof(float));
        l.x = calloc(l.batch*l.outputs, sizeof(float));
        l.x_norm = calloc(l.batch*l.outputs, sizeof(float));
    }
    if(adam){
        l.adam = 1;
        l.m = calloc(c*n*size*size, sizeof(float));
        l.v = calloc(c*n*size*size, sizeof(float));
        l.bias_m = calloc(n, sizeof(float));
        l.scale_m = calloc(n, sizeof(float));
        l.bias_v = calloc(n, sizeof(float));
        l.scale_v = calloc(n, sizeof(float));
    }

#ifdef GPU
    l.forward_gpu = forward_convolutional_layer_gpu;
    l.backward_gpu = backward_convolutional_layer_gpu;
    l.update_gpu = update_convolutional_layer_gpu;

    if(gpu_index >= 0){
        if (adam) {
......
        if(binary){
.....
        }
        if(xnor){
......
        }

        if(batch_normalize){
.......
        }
........
    }
#endif
    l.workspace_size = get_workspace_size(l);
    l.activation = activation;

    fprintf(stderr, "conv  %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", n, size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c);

    return l;
}
```
### 1.3.2 parser.c->load_weights()
返回train_detector()函数中，顺序往下执行至**load_weights()**  
``` c
void load_weights(network *net, char *filename)
{
    //调用load_weights_upto(net, filename, net->n)函数
    load_weights_upto(net, filename, 0, net->n);//1.3.2.1
```
#### 1.3.2.1 parser.c->load_weights_upto()
``` c
void load_weights_upto(network *net, char *filename, int start, int cutoff)
{
#ifdef GPU
    if(net->gpu_index >= 0){
        cuda_set_device(net->gpu_index);
    }
#endif
    fprintf(stderr, "Loading weights from %s...", filename);
    fflush(stdout);
    FILE *fp = fopen(filename, "rb");
    if(!fp) file_error(filename);

    int major;
    int minor;
    int revision;
    fread(&major, sizeof(int), 1, fp);
    fread(&minor, sizeof(int), 1, fp);
    fread(&revision, sizeof(int), 1, fp);
    if ((major*10 + minor) >= 2 && major < 1000 && minor < 1000){
        fread(net->seen, sizeof(size_t), 1, fp);
    } else {
        int iseen = 0;
        fread(&iseen, sizeof(int), 1, fp);
        *net->seen = iseen;
    }
    int transpose = (major > 1000) || (minor > 1000);
    
    int i;
    for(i = start; i < net->n && i < cutoff; ++i){
        //读取各层权重
	layer l = net->layers[i];
        if (l.dontload) continue;
        if(l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL){
            load_convolutional_weights(l, fp);//1.3.2.1.1
        }
        .
	.
	.
#ifdef GPU
            if(gpu_index >= 0){
                push_local_layer(l);
            }
#endif
        }
    }
    fprintf(stderr, "Done!\n");
    fclose(fp);
}
```
#### 1.3.2.1.1 parser.c->load_convolutional_weights()
先读biases，n个卷积核有n个biasies，然后读入weight的个数num=l.nweights  
``` c
/*
*函数原型：size_t fread ( void *buffer, size_t size, size_t count, FILE *stream) ;
*buffer：用于接收数据的内存地址
*size：要读的每个数据项的字节数，单位是字节
*count：要读count个数据项，每个数据项size个字节.
*stream：输入流
*/
void load_convolutional_weights(layer l, FILE *fp)
{
    int num = l.c/l.groups*l.n*l.size*l.size;//卷积层的参数个数，卷积核个数×通道数×卷积核长度×卷积核宽度
    fread(l.biases, sizeof(float), l.n, fp);
    if (l.batch_normalize && (!l.dontloadscales)){
        fread(l.scales, sizeof(float), l.n, fp);
        fread(l.rolling_mean, sizeof(float), l.n, fp);
        fread(l.rolling_variance, sizeof(float), l.n, fp);
	.
	.
	.
    }
    fread(l.weights, sizeof(float), num, fp);
    if (l.flipped) {
    	//转置矩阵
        transpose_matrix(l.weights, l.c*l.size*l.size, l.n);
    }
#ifdef GPU
    if(gpu_index >= 0){
        push_convolutional_layer(l);//函数在src/convolutional_kernels.cu中
    }
#endif
}
```
### 1.3.3 network.c->train_network()
调用train_network(network net, data d) ，开始训练  
``` c
float train_network(network *net, data d)
{
    //调用data的相关参数
    assert(d.X.rows % net->batch == 0);
    int batch = net->batch;
    //注意，n现在表示加载一次数据可以训练几次，其实就是subdivisions
    int n = d.X.rows / batch;

    int i;
    float sum = 0;
    for(i = 0; i < n; ++i){
    //完成数据拷贝
        get_next_batch(d, batch, i*batch, net->input, net->truth);//1.3.3.1
        float err = train_network_datum(net);//1.3.3.2
        sum += err;
    }
    return (float)sum/(n*batch);
}
```
#### 1.3.3.1 data.c->get_next_batch()
进入src/data.c，调用get_next_batch()，完成数据拷贝  
``` c
void get_next_batch(data d, int n, int offset, float *X, float *y)
{
    int j;
    for(j = 0; j < n; ++j){
	//offset就是第几个batch(i*batch)了，j表示的是每个batch中的第几个样本（图像）
        int index = offset + j;
	//void *memcpy(void *dest, const void *src, size_t n);
        //memcpy函数的功能是从源src所指的内存地址的起始位置开始拷贝n个字节到目标dest所指的内存地址的起始位置中
        memcpy(X+j*d.X.cols, d.X.vals[index], d.X.cols*sizeof(float));
        if(y) memcpy(y+j*d.y.cols, d.y.vals[index], d.y.cols*sizeof(float));
    }
}
```
#### 1.3.3.2 network.c->train_network_datum()
``` c
float train_network_datum(network net)
{
#ifdef GPU//使用GPU时训练网络
    if(gpu_index >= 0) return train_network_datum_gpu(net);//1.3.3.2.1
#endif
    *net.seen += net.batch;
    net.train = 1;
    forward_network(net);
    backward_network(net);
    //计算平均损失
    float error = *net.cost;
    //*net.seen是已经训练过的子batch数，
    //((*net.seen)/net.batch)%net.subdivisions的意义正是已经训练过了多少个真正的batch。
    if(((*net.seen)/net.batch)%net.subdivisions == 0) update_network(net);
    return error;
}
```
##### 1.3.3.2.1 network_kernels.cu->train_network_datum_gpu()
``` c
float train_network_datum_gpu(network net)
{
    *net.seen += net.batch;

    int x_size = net.inputs*net.batch;
    int y_size = net.truths*net.batch;
    cuda_push_array(net.input_gpu, net.input, x_size);
    cuda_push_array(net.truth_gpu, net.truth, y_size);

    net.train = 1;
    //前向反向传播
    forward_network_gpu(net);
    backward_network_gpu(net);

    float error = *net.cost;//网络代价 
    if (((*net.seen) / net.batch) % net.subdivisions == 0) update_network_gpu(net);//更新网络  

    return error;
}
```
