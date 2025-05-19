# Model Reproduction

这个仓库用来记录一些开源模型代码浏览过程，每个文件夹对应了一个开源项目，建议在进入子文件夹时配合对应的CSDN博客阅读，因为在编写博客的时候回添加一些图片和说明，由于精力有限，这些说明不会在子文件夹中的 README 中呈现。

* CSDN 专栏链接：[]()

---
# 准备工作（推荐）

在此过程中会大量用到python库引用关系图，这里通过 `pydeps` 库进行绘制，需要安装以下依赖：

```shell
(model) $ sudo apt-get install graphviz # Ubuntu
(model) $ brew install graphviz         # MacOS
(model) $ pip install pydeps
```

这个工具用来绘制文件的库依赖关系生成 `svg` 格式文件，如果你使用的是 vs-code 可以在插件中心中搜索 `Svg Preview` 插件并安装。

常用命令：
```shell
(model) $ pydeps model.py --max-bacon=1
```

---
# 已完成工程

|原始工程|分支|commit|复现仓库|所属领域|难易程度|
|--|--|--|--|--|--|
|[nanoGPT](https://github.com/karpathy/nanoGPT)|master|93a43d9a5c22450bbf06e78da2cb6eeef084b717|[nanoGPT](./nanoGPT/)|LLM|简单|

