# Transformer Encoder

# Transformer Encoder 概要

![Untitled](Transformer%20Encoder%205f54a03260b64c6cbdb41501df01778b/Untitled.png)

Input layerの出力は次にTransformer Encoderに入力されます。

Transformer Encoder はEncoder Blockと呼ばれるブロックが多段に重なって構成されます。

入力されたクラストークンおよび各Patchに対してEncoder Blockによる計算を行い最後にクラストークンを出力します。

# Encoder Block

![スクリーンショット 2022-12-30 13.02.16.png](Transformer%20Encoder%205f54a03260b64c6cbdb41501df01778b/%25E3%2582%25B9%25E3%2582%25AF%25E3%2583%25AA%25E3%2583%25BC%25E3%2583%25B3%25E3%2582%25B7%25E3%2583%25A7%25E3%2583%2583%25E3%2583%2588_2022-12-30_13.02.16.png)

Encoder BlockはNorm, Multi-Head Self-Attention, MLPの3つのブロックで構成されています。

NormはLayer Normalizationのことを指します。

Layer Normalization層の入力を保持して出力と結合するSkip Connectionが２箇所あります。

# Layer Normalization

Layer Normalizationは正規化手法の一つです。他に有名な正規化手法としてはBatch Normalizationでバッチ内で平均・分散で正規化をするのに対し、Layer Normalizationは1つのサンプルにおける各レイヤーの隠れ層の値の平均・分散で正規化します。

Layer Normalizationの採用理由はデータごとにトークン数の異なる時系列データに有効であるためです。

[Layer Normalizationを理解する](https://data-analytics.fun/2020/07/16/understanding-layer-normalization/#toc1)

### 数式表現

$LN(\textbf{a})_i = \gamma_i \frac{a_i - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta_i$

$\mu = \frac{1}{n} \sum^K_{i = 1} a_i$：平均

$\sigma = \sqrt{\frac{1}{k}\sum^K_{i=1}(a_i - \mu)^2}$ ：標準偏差

Layer Normalizationはレイヤーへの入力$\mathbf{a}$の中で平均0, 最大値1になるように正規化します。

$\gamma, \beta$は正規化の値をシフトするためのパラメータになります。

# MHSA

MHSAの解説は長いため以下ページに記載しています。

[MHSA (Multi-Head Self-Attention)](Transformer%20Encoder%205f54a03260b64c6cbdb41501df01778b/MHSA%20(Multi-Head%20Self-Attention)%2041f6c97a219a47abb8a6899685ca1df3.md)

# MLP (Multi Layer Perceptron)

![Untitled](Transformer%20Encoder%205f54a03260b64c6cbdb41501df01778b/Untitled%201.png)

Encoder BlockのMLPは２層の線形層を用いて構成されます。

以下図のように1つ目の線形層の後のみ活性化関数GELUを用い、2つ目の層には活性化関数を用いません。線形層、活性化関数の後、２層どちらにもDropoutが行われます。

### GELU

GELUはTransformer系のモデルでReluと変わってよく用いられる活性化関数です。

Reluは入力が0以上の時に1を掛け、0未満の値には0を掛ける一方、
Geluは正規分布の分布関数を用いて確率的に入力に0または1を掛けます。

GeluはReluを滑らかな曲線にした表現になります。

![スクリーンショット 2023-02-11 18.16.55.png](Transformer%20Encoder%205f54a03260b64c6cbdb41501df01778b/%25E3%2582%25B9%25E3%2582%25AF%25E3%2583%25AA%25E3%2583%25BC%25E3%2583%25B3%25E3%2582%25B7%25E3%2583%25A7%25E3%2583%2583%25E3%2583%2588_2023-02-11_18.16.55.png)

[活性化関数GELUを理解する](https://data-analytics.fun/2020/09/04/understanding-gelu/)

### Dropout

Dropoutは学習時に一部のニューロンの出力を確率的にに0にします。ネットワークがオーバーフィッティング(過学習)するのを抑える仕組みです。

## 数式表現

先頭から$l$個目のEncoder Blockを$z_l$とすると以下の式で表します。

$z'_l = MHSA(LN(z_{l-1})) + z_{l-1}$

$z_l = MLP(LN(z'_l)) + z'_l \in\mathbb{R}^{(N_p + 1) \times D}$

$LN(\cdot)$：Layer Normalization

$N_p$：Patch数,  $D$ ：Patchのベクトル長

# 実装

```python
import torch.nn as nn

class VitEncoderBlock(nn.Module): 
    def __init__(self, emb_dim:int=384, head:int=8, hidden_dim:int=384*4, dropout: float=0.):
        """
        引数:
            emb_dim: 埋め込み後のベクトルの長さ
            head: ヘッドの数
            hidden_dim: Encoder BlockのMLPにおける中間層のベクトルの長さ 
                        原論文に従ってemb_dimの4倍をデフォルト値としている
            dropout: ドロップアウト率
        """
        super(VitEncoderBlock, self).__init__()
        # 1つ目のLayer Normalization [2-5-2項]
        self.ln1 = nn.LayerNorm(emb_dim)
        # MHSA [2-4-7項]
        self.msa = MultiHeadSelfAttention(
        emb_dim=emb_dim, head=head,
        dropout = dropout,
        )
        # 2つ目のLayer Normalization [2-5-2項] 
        self.ln2 = nn.LayerNorm(emb_dim)
        # MLP [2-5-3項]
        self.mlp = nn.Sequential( 
            nn.Linear(emb_dim, hidden_dim), 
            nn.GELU(),
            nn.Dropout(dropout), 
            nn.Linear(hidden_dim, emb_dim), 
            nn.Dropout(dropout)
        )
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """ 
        引数:
            z: Encoder Blockへの入力。形状は、(B, N, D)
                B: バッチサイズ、N:トークンの数、D:ベクトルの長さ
        返り値:
            out: Encoder Blockへの出力。形状は、(B, N, D)。[式(10)]
                B:バッチサイズ、N:トークンの数、D:埋め込みベクトルの長さ 
        """
        # Encoder Blockの前半部分 [式(12)] 
        out = self.msa(self.ln1(z)) + z
        # Encoder Blockの後半部分 [式(13)] 
        out = self.mlp(self.ln2(out)) + out 
        return
```

# 引用文献

[GitHub - ghmagazine/vit_book](https://github.com/ghmagazine/vit_book)

[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)

[Layer Normalizationを理解する](https://data-analytics.fun/2020/07/16/understanding-layer-normalization/#toc1)

[活性化関数GELUを理解する](https://data-analytics.fun/2020/09/04/understanding-gelu/)