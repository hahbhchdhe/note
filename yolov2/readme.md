# yolov2源码详解
:space_invader:训练命令： **./darknet detector train cfg/coco.data cfg/yolo.cfg darknet19_448.conv.23**  
:space_invader:测试图片命令： **./darknet detect test cfg/coco.data cfg/yolo.cfg yolo.weights data/dog.jpg**
## 1.训练部分
### 1.1
首先./darknet是程序的名称，主函数应该在darknet.c源文件中，其main()函数中，要求输入的参数不少于2个。 
根据不同命令参数*argv[1]* 进入不同的调用方法  
**在训练过程中：** *argv[0]* 即第一个参数是./darknet，是程序的名称。*argv[1]* 即第二个参数是“detector”。  
[代码定位](https://github.com/pjreddie/darknet/blob/56d69e73aba37283ea7b9726b81afd2f79cd1134/examples/darknet.c#L417)
``` c
} else if (0 == strcmp(argv[1], "detector")){
        run_detector(argc, argv);
```
因此，程序进入 **run_detector(argc, argv);**  
### 1.2
**detector.c** 中 **run_detector()** 函数， 开始是对一些参数的缺省值 [代码定位](https://github.com/pjreddie/darknet/blob/56d69e73aba37283ea7b9726b81afd2f79cd1134/examples/detector.c#L655) 
```c
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
