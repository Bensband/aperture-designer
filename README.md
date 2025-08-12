### 程序介绍
这是一套针对菲涅尔衍射区域，集成“正向模拟”+“逆向设计”的光学程序。其逆向衍射设计模块，采用深度学习智能算法，支持在不同的参数下，根据低频目标衍射图像逆向设计对应的掩膜结构。

本程序由python语言构建，由pyinstaller打包，源代码见code文件夹，Windows可执行文件位于dist文件夹

### 声明
本程序的Ui界面基于以下开源技术构建：
- PySide6 (Qt for Python) 
许可证：GNU Lesser General Public License v3
源码：https://pypi.org/project/PySide6/
完整许可证文本：https://www.gnu.org/licenses/lgpl-3.0.html

### 运行配置
核心安装包版本：
1. torch:                 2.6.0+cu126
2. pyside6:               6.9.0

[!]
本程序基于深度学习算法如果没有gpu加速运行时间将延长至数个小时（反之则3-5min）  

torch支持的cuda版本为12.6及以上。

### 如何使用？
进入dist文件夹后，会发现有两个用7-zip分割后的压缩包，只解压第一个，后面会依次解压  

解压完成之后找到aperture_designer.exe即可运行

其中，buffer文件夹用于存储缓存  

result文件夹用于存放结果