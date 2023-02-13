# MLP Head & ViT実装

# MLP Head 概要

![Untitled](MLP%20Head%20&%20ViT%E5%AE%9F%E8%A3%85%205e7f555c0b924befbe2ee79662c55dbf/Untitled.png)

MLP Headはクラス分類を行う分類器です。

分類器はシンプルで、Layer Normalizationと線形層(Linear)で構築されています。

MLPの出力次元はクラス数になります。

## 数式表現

MLP Headの入力はTransformer Encoderの$l$個目のEncoder Bolockの出力になります。

$y = LN(z^0_L)W^y$

$W^y \in \mathbb{R}^{D \times M}$：線形層の重み

$D$：Patchのベクトル長, $M$：クラス数

$y$：クラス確率ベクトル

# ViT まとめ

ViTは大きく次の3つで構成されています。各処理と数式表現は以下になります。

## Input Layer

Inpute Layerでは最初に入力画像$x$を分割しPatch $x_p$を得ます。

次にPatchを線形層$E$で埋め込みし、クラストークン$x_{class}$を結合します。

最後に位置埋め込み$E_{pos}$を加算します。

### 数式表現

$z_0 = [x_{class}; x^1_pE;x^2_pE;\cdots;x^{N_p}_pE]  + E_{pos}$ 

$x$：入力画像, $x_p$：Patch, $N_p$：Patch数

$E$：線形層, $E_{pos}$：位置埋め込み

$z_0$：Input Layer出力

## Transformer Encoder

Transformer Encoderは$L$個のEncoder Bloackが多段になって構成されています。

Encoder Blockの前半部分はMulti-Head Self-Attention ($MHSA(\cdot )$)で
後半部分はMulti Layer Perceptron ($MLP(\cdot)$)で構築されています。

### 数式表現

$z'_l = MHSA(LN(z_{l-1})) + z_{l-1}$                   $l = 1,\dotsc , L$

$z_l = MLP(LN(z')) + z'_l$                               $l = 1,\dotsc , L$

$LN(\cdot )$：Layer Normalization

$z_l$：第$l$個目のEncoder Blockの出力

## MLP Head

最後のEncoder Blockの出力 $z_L$のうちクラストークン部分の $z_L^0$が入力されます。

Layer Normalizationで正規化された後、線形層 $W^y$で埋め込み
クラス確率ベクトル $yを出力します。$

### 数式表現

$y = LN(z^0_L)W^y$

$LN(\cdot )$：Layer Normalization

# ViT 実装

```python
class Vit(nn.Module): 
    def __init__(self, in_channels:int=3, num_classes:int=10, emb_dim:int=384, num_patch_row:int=2, image_size:int=32, num_blocks:int=7, head:int=8, hidden_dim:int=384*4, dropout:float=0.):
        """ 
        引数:
            in_channels: 入力画像のチャンネル数
            num_classes: 画像分類のクラス数
            emb_dim: 埋め込み後のベクトルの長さ
            num_patch_row: 1辺のパッチの数
            image_size: 入力画像の1辺の大きさ。入力画像の高さと幅は同じであると仮定 
            num_blocks: Encoder Blockの数
            head: ヘッドの数
            hidden_dim: Encoder BlockのMLPにおける中間層のベクトルの長さ 
            dropout: ドロップアウト率
        """
        super(Vit, self).__init__()
        # Input Layer [2-3節] 
        self.input_layer = VitInputLayer(
            in_channels, 
            emb_dim, 
            num_patch_row, 
            image_size)

        # Encoder。Encoder Blockの多段。[2-5節] 
        self.encoder = nn.Sequential(*[
            VitEncoderBlock(
                emb_dim=emb_dim,
                head=head,
                hidden_dim=hidden_dim,
                dropout = dropout
            )
            for _ in range(num_blocks)])

        # MLP Head [2-6-1項] 
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        引数:
            x: ViTへの入力画像。形状は、(B, C, H, W)
                B: バッチサイズ、C:チャンネル数、H:高さ、W:幅
        返り値:
            out: ViTの出力。形状は、(B, M)。[式(10)]
                B:バッチサイズ、M:クラス数 
        """
        # Input Layer [式(14)]
        ## (B, C, H, W) -> (B, N, D)
        ## N: トークン数(=パッチの数+1), D: ベクトルの長さ 
        out = self.input_layer(x)
        
        # Encoder [式(15)、式(16)]
        ## (B, N, D) -> (B, N, D)
        out = self.encoder(out)

        # クラストークンのみ抜き出す
        ## (B, N, D) -> (B, D)
        cls_token = out[:,0]

        # MLP Head [式(17)]
        ## (B, D) -> (B, M)
        pred = self.mlp_head(cls_token)
        return pred
```

# ViTモデルバリアント

![スクリーンショット 2023-02-12 21.25.50.png](MLP%20Head%20&%20ViT%E5%AE%9F%E8%A3%85%205e7f555c0b924befbe2ee79662c55dbf/%25E3%2582%25B9%25E3%2582%25AF%25E3%2583%25AA%25E3%2583%25BC%25E3%2583%25B3%25E3%2582%25B7%25E3%2583%25A7%25E3%2583%2583%25E3%2583%2588_2023-02-12_21.25.50.png)

Vitの論文ではVitのモデルのパラメータ定義をBase, Large, Hugeに分けて行っています。

Base, LargeはBERTに基づく定義で、HugeはVitの論文で新たに定義されました。

例えばViT-L/16は、$16 \times 16$個のPatch分割を行うViT-Largeを意味します。

# 引用文献

[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)

[GitHub - ghmagazine/vit_book](https://github.com/ghmagazine/vit_book)