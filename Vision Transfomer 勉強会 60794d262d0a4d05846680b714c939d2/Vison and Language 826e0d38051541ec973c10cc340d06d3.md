# Vison and Language

## Vison and Language

画像と言語両方のデータを同時に扱うタスク

**VQA (Visual Question Answering)：**画像に対する質問に答える

**Image Captioning：**画像の説明文を生成する

**VLN (Vision and Language navigation)：**言葉でエージェントの移動を指示する

BERTの応用研究

## VideoBERT（動画+テキスト）

BERTを拡張し、テキストに加え動画を入力とする。

動画を静止画として切り出し、テキストと静止画を連結し一つの系列データとして入力する。

![https://d3i71xaburhd42.cloudfront.net/c41a11c0e9b8b92b4faaf97749841170b760760a/4-Figure3-1.png](https://d3i71xaburhd42.cloudfront.net/c41a11c0e9b8b92b4faaf97749841170b760760a/4-Figure3-1.png)

## ViLBERT (画像+テキスト)

物体検出で得た物体(上）とテキスト（下）を入力とする。

Co-TRMで両者の情報を参照する。

![https://149695847.v2.pressablecdn.com/wp-content/uploads/2019/08/Screenshot-23.png](https://149695847.v2.pressablecdn.com/wp-content/uploads/2019/08/Screenshot-23.png)

画像に写っている物体と文中の単語の関係を扱う。

### **Bottom-UP Attention**

物体検出によって得られた各物体を一つのトークンとして扱う。

例）人と犬が写っている画像→「人」「犬」という２つのトークン

### Single-Stream, Multi-Stream

**Single-Stream:**１つのネットワークで画像とテキストをまとめて処理

**Multi-Stream:**画像とテキストを別々に処理するネットワークを用意

![IMG_0491.heic](Vison%20and%20Language%20826e0d38051541ec973c10cc340d06d3/IMG_0491.heic)