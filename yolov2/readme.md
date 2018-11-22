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
**因为是训练过程，所以进入train_detector（）函数。**[代码定位](https://github.com/pjreddie/darknet/blob/56d69e73aba37283ea7b9726b81afd2f79cd1134/examples/detector.c#L655)   
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
###1.3 detector.c->train_detector()
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
    srand(time(0));//srand()与rand()结合产生随机数 
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
        nets[i] = parse_network_cfg(cfgfile);//解析网络结构，包括训练参数，层数，各层的类别、参数，各层的输入大小等。
        //如果命令包含权重文件，则装载权重文件
        if(weightfile){
            load_weights(&nets[i], weightfile);
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
    //
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
    args.m = plist->size;//总的数据量
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
            pthread_join(load_thread, 0);
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
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data(args);
        //可视化数据扩增之后训练样本
        /*
        int k;
        for(k = 0; k < l.max_boxes; ++k){
            box b = float_to_box(train.y.vals[10] + 1 + k*5);
            if(!b.x) break;
            printf("loaded: %f %f %f %f\n", b.x, b.y, b.w, b.h);
        }
        */
        /*
        int zz;
        for(zz = 0; zz < train.X.cols; ++zz){
            image im = float_to_image(net.w, net.h, 3, train.X.vals[zz]);
            int k;
            for(k = 0; k < l.max_boxes; ++k){
                box b = float_to_box(train.y.vals[zz] + k*5);
                printf("%f %f %f %f\n", b.x, b.y, b.w, b.h);
                draw_bbox(im, b, 1, 1,0,0);
            }
            show_image(im, "truth11");
            cvWaitKey(0);
            save_image(im, "truth11");
        }
        */

        printf("Loaded: %lf seconds\n", sec(clock()-time));

        time=clock();
        float loss = 0;

//训练网络
#ifdef GPU
        if(ngpus == 1){
            loss = train_network(net, train);//开始训练
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
