# yolov2源码详解
训练命令： **./darknet detector train cfg/coco.data cfg/yolo.cfg darknet19_448.conv.23**  
测试图片命令： **./darknet detect test cfg/coco.data cfg/yolo.cfg yolo.weights data/dog.jpg**
## 1.训练部分
### 1.1
首先./darknet是程序的名称，主函数应该在darknet.c源文件中，其main()函数中，要求输入的参数不少于2个。  
其中 *argv[0]* 即第一个参数是./darknet，是程序的名称。*argv[1]* 即第二个参数是“detector”。  
根据不同命令参数argv[1]进入不同的调用方法  
[代码定位](https://github.com/pjreddie/darknet/blob/56d69e73aba37283ea7b9726b81afd2f79cd1134/examples/darknet.c#L417)
``` c
} else if (0 == strcmp(argv[1], "detector")){
        run_detector(argc, argv);
```
