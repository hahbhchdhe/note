# yolov2源码详解
训练命令： **./darknet detector train cfg/coco.data cfg/yolo.cfg darknet19_448.conv.23**
测试图片命令： **./darknet detect test cfg/coco.data cfg/yolo.cfg yolo.weights data/dog.jpg**
## 训练部分
首先./darknet是程序的名称，主函数应该在darknet.c源文件中
在main()函数中，要求输入的参数不少于2个，其中argv[0]即第一个参数是./darknet，是程序的名称。argv[1]即第二个参数是detector 
根据不同命令参数argv[1]进入不同的调用方法

``` c

```
