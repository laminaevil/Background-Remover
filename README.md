# Background Remover

These all are for images in Opencv. (Exclude Raw, Bmp files.)

## Getting Started

First, If you have to load the image and start to read C++ Book to know how to programm...


### Prerequisites

The software you need to install

```
Opencv
```

### Installing OpenCV

C++

```
1. 首先到環境變數裡面的PATH裡面增加一個路徑叫做
D:\Warehouse\appData\opencv\build\x64\vc15\bin (這個路徑是你放OPENCV的地方)
2. 完成了這步驟呢 可能會出現兩個問題，1. 你的電腦本身太舊所以要重開機讓他重新讀環境變數檔案
3. 開啟 Visual studio 2017 後 新增一個c++的檔案
4. 在專案中打開專案的屬性並且設置以下幾點
4.1 設置c++中的 
其他的 include目錄 D:\Warehouse\appData\opencv\build\include 路徑是依照你們的路徑
4.2 設置連結器路徑
其他程式庫目錄 D:\Warehouse\appData\opencv\build\x64\vc15\lib
4.3設置連結器的輸入
其他相依姓 opencv_world341d.lib
5. 最後  因為OPENCV 是用X64的 所以要把 86 轉乘64 才能用
```

Python
```
pip install opencv-python
```

## Running the tests

```
Mat image = CV_RawRead(256, 256, "ButterFly");
Mat dst[2], dst_1[2];
CV_DFT(image, dst, "Butterfly", "Butterfly");
CV_ILPF(dst, dst_1, image.rows, image.cols, 50, "Butterfly  ", "Butterfly");
CV_IDFT(dst_1, "Butterfly", "Butterfly",50 ,0);

```


### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Authors

* **NamuhEvil** - *Initial work* - 
[Namuhevil](https://github.com/namuhevil)

## License

This project is licensed under the NamuhEvil

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
