# コンピュータビジョンにおけるTransformer

# DETR：DEtection TRansformer (物体検出)

## 論文URL

[End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)

## 概要

Transfomerを応用し物体検出に応用したモデル

Facebook AIによって提案

Vision and Languageタスク向けモデルは物体検出にTransfomerを用いていない（Transformer由来のものを用いていない）

画像を空間的に均等に区切って領域ごとに特徴表現を獲得し、各領域を１つのトークンとみなして処理を行う。

![https://www.ogis-ri.co.jp/otc/hiroba/technical/detr/img/pic202109-005.png](https://www.ogis-ri.co.jp/otc/hiroba/technical/detr/img/pic202109-005.png)

Transfomerに入力する前にCNNで画像の次元削減した特徴マップを作成する

![https://www.ogis-ri.co.jp/otc/hiroba/technical/detr/img/pic202109-006.png](https://www.ogis-ri.co.jp/otc/hiroba/technical/detr/img/pic202109-006.png)

## 参考文献

[Transformerを使った初めての物体検出「DETR」 - 第1回 今までの物体検出との違いとColabでの推論 | オブジェクトの広場](https://www.ogis-ri.co.jp/otc/hiroba/technical/detr/part1.html)

# iGPT : imageGPT (画像生成、補完)

![https://webbigdata.jp/wp-content/uploads/2020/10/1_Favorites.png](https://webbigdata.jp/wp-content/uploads/2020/10/1_Favorites.png)