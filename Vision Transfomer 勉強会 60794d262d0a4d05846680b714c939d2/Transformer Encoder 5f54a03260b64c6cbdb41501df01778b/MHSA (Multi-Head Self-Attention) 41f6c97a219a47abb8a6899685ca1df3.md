# MHSA (Multi-Head Self-Attention)

MHSAはSelf-Attentionをベースに構築されます。最初にSelf-Attentionについて解説し、その後MHSAについて解説していきます。

# Self-Attention

![スクリーンショット 2022-12-27 0.28.19.png](MHSA%20(Multi-Head%20Self-Attention)%2041f6c97a219a47abb8a6899685ca1df3/%25E3%2582%25B9%25E3%2582%25AF%25E3%2583%25AA%25E3%2583%25BC%25E3%2583%25B3%25E3%2582%25B7%25E3%2583%25A7%25E3%2583%2583%25E3%2583%2588_2022-12-27_0.28.19.png)

Self Attention機構はInputLayerから出力されたクラストークンと複数のPatchの結合ベクトル(Image Feature Maps)を入力とします。

入力からPatch同士の類似性(Attention Map)を算出しそれを入力元に加重和して再度埋め込むことによって特徴表現を強化します。

## 1: 入力値への埋め込み

![図1.png](MHSA%20(Multi-Head%20Self-Attention)%2041f6c97a219a47abb8a6899685ca1df3/%25E5%259B%25B31.png)

Self AttentionではまずInput Layerによって得られたPatchのベクトル(Image Featrue Maps)を３つの一層の線形層によって埋め込みを行い情報を抽出します。

抽出された各ベクトルはQ,K,Vと呼びます。ベクトルの行成分は各Patchのベクトルを指しています。それぞれのベクトルの詳細は以下になります。

### Q(Query)

Patch間の類似性(Attention Map)を計算する時、比較元のPatchを指すベクトル

### K(Key)

Patch間の類似性(Attention Map)を計算する時、比較先のPatchを指すベクトル

### V(Value)

入力を保持したPatchのベクトル。類似性(Attention Map)が後に加重和される。

## 2: Attention Mapの算出

![スクリーンショット 2022-12-03 16.21.38.png](MHSA%20(Multi-Head%20Self-Attention)%2041f6c97a219a47abb8a6899685ca1df3/%25E3%2582%25B9%25E3%2582%25AF%25E3%2583%25AA%25E3%2583%25BC%25E3%2583%25B3%25E3%2582%25B7%25E3%2583%25A7%25E3%2583%2583%25E3%2583%2588_2022-12-03_16.21.38.png)

Patch同士の類似度は各Patchのベクトルの内積によって算出されます。これはベクトルで表現された単語の類似度の求め方に基づいています。

[コサイン類似度](https://www.cse.kyoto-su.ac.jp/~g0846020/keywords/cosinSimilarity.html)

全てのPatch同士の内積を求めるにはQと転置したKの行列積によって実現できます。得られたベクトルをAttention Mapと呼びます。

![スクリーンショット 2022-12-11 2.05.40.png](MHSA%20(Multi-Head%20Self-Attention)%2041f6c97a219a47abb8a6899685ca1df3/%25E3%2582%25B9%25E3%2582%25AF%25E3%2583%25AA%25E3%2583%25BC%25E3%2583%25B3%25E3%2582%25B7%25E3%2583%25A7%25E3%2583%2583%25E3%2583%2588_2022-12-11_2.05.40.png)

Attention Mapの成分$(i,j)$はQの$i$番目のPatch、Kの$j$番目のPatchの内積になり、値はそれら2つの類似度を示します。値は大きい方が類似性が高くなります。

その後QとKの行列積によって得られたベクトルを行方向にSoftmax関数を適応し、行成分の合計値を1にします。

## 3: VへのAttention Mapの加重和

![スクリーンショット 2022-12-11 1.55.07.png](MHSA%20(Multi-Head%20Self-Attention)%2041f6c97a219a47abb8a6899685ca1df3/%25E3%2582%25B9%25E3%2582%25AF%25E3%2583%25AA%25E3%2583%25BC%25E3%2583%25B3%25E3%2582%25B7%25E3%2583%25A7%25E3%2583%2583%25E3%2583%2588_2022-12-11_1.55.07.png)

算出したAttention MapとVの行列積を算出します。

これによって1つのPatchに対して、自身のPatchを含めた全てのPatchの類似度の情報が付加されます。すなわち、Self Attentionでは1つのPatchのベクトルを計算するのに、全てのPatchのベクトルが用いられます。

これによって1つのPatchは画像全体を考慮して計算されます。そのため、Self Attentionは大域的に画像の特徴量を学習できます。

# MHSA (Multi-Head Self Attention)

Self Attentionで算出されるAttention Mapが複数あれば、各Patch間の関係をAttention Mapの数を学習することができます。複数のPatch間の関係を学習に用いることで効果的に学習ができそうなことがわかります。VitではMHSA(Multi-Head Self Attention)を導入しこの学習を実現します。

MHSAではAttention Mapの計算に用いるQとKをHeadと呼ばれる分割数分縦に分割します。分割した後それぞれの内積を計算することによってHeadの数のAttention Mapを作成することができます。

![スクリーンショット 2022-12-26 23.00.52.png](MHSA%20(Multi-Head%20Self-Attention)%2041f6c97a219a47abb8a6899685ca1df3/%25E3%2582%25B9%25E3%2582%25AF%25E3%2583%25AA%25E3%2583%25BC%25E3%2583%25B3%25E3%2582%25B7%25E3%2583%25A7%25E3%2583%2583%25E3%2583%2588_2022-12-26_23.00.52.png)

Vも同様にHeadの数分縦に分割します。分割されたVはAttention Mapより加重和されます。加重和を行なった後もPatchのベクトルはHeadの数分分割されたままなので最後結合します。

![スクリーンショット 2022-12-26 23.11.04.png](MHSA%20(Multi-Head%20Self-Attention)%2041f6c97a219a47abb8a6899685ca1df3/%25E3%2582%25B9%25E3%2582%25AF%25E3%2583%25AA%25E3%2583%25BC%25E3%2583%25B3%25E3%2582%25B7%25E3%2583%25A7%25E3%2583%2583%25E3%2583%2588_2022-12-26_23.11.04.png)

# MSHAの数式表現

## 1: 入力

Input Layerからの入力Zを引き続き以下とします。

$Z = [x_{class}; x^1_pE;x^2_pE;\cdots;x^{N_p}_pE]  + E_{pos} \in\mathbb{R}^{(N_p + 1) \times D}$

$N_p$：Patch数,  $D$ ：Patchのベクトル長

$x_{class} \in \mathbb{R}^D$ ：クラストークン

$x^i_p \in \mathbb {R} ^ {D}$       $i = (1 \sim N_p)$ ： $i$番目のパッチのベクトル

## 2: $Q, K, V$の計算

$K, Q, V$の計算の埋め込みに用いる線形層を以下とします。

$W^Q, W^K, W^V \in \mathbb{R}^{D \times D_h}$

$W$の埋め込みによって長さ$D$のベクトルが長さ$D_h$のベクトルになります。

埋め込みによって得られるは以下の式で表します。

$Q = ZW^Q, \ \ \ \ \ \ \ \ \ \ \  Q\in \mathbb{R}^{(N_p + 1) \times D_h}$

$K = ZW^K, \ \ \ \ \ \ \ \ \ \ \  K\in \mathbb{R}^{(N_p + 1) \times D_h}$

$V = ZW^V, \ \ \ \ \ \ \ \ \ \ \  V\in \mathbb{R}^{(N_p + 1) \times D_h}$

## 3: Attention Mapの計算

![スクリーンショット 2022-12-11 2.05.40.png](MHSA%20(Multi-Head%20Self-Attention)%2041f6c97a219a47abb8a6899685ca1df3/%25E3%2582%25B9%25E3%2582%25AF%25E3%2583%25AA%25E3%2583%25BC%25E3%2583%25B3%25E3%2582%25B7%25E3%2583%25A7%25E3%2583%2583%25E3%2583%2588_2022-12-11_2.05.40.png)

$A = softmax(\frac{QK^T}{\sqrt{D_h}}), \ \ A \in \mathbb{R}^{(N_p + 1)^2}$

ここで新しくQと転置Kの行列積に$\sqrt{D_h}$を除算します。

ベクトル長Dが大きくなった時に行列積の結果が大きくなりすぎないようにする意図になります。

## 4: VへのAttention Mapによる加重和

加重和はVとAttention Mapの行列積で以下の式になります。

$SA(Z) = AV$

$\ \ \ \ \ \ \ \ \ \ \ \ \ = softmax(\frac{QK^T}{\sqrt{D_h}}) V, \ \ SA(Z) \in \mathbb{R}^{(N_p + 1) \times D_h}$

## 5: **分割ベクトルの結合**

MHSAではHeadの数分のベクトルを結合していきます。

MHSAの分割ベクトルの結合は以下の式になります。

$[SA_1(z);SA_2(z);\ldots ; SA_k(z)]  \in \mathbb{R}^{(N_p + 1) \times kD_h}$

MHSAでは入力と出力の次元は等しくなります。ここで分割ベクトルの結合後のベクトルの次元$kD_h$ から入力の次元$D$に戻します。次元を戻すために１層の線形層を用いて埋め込みを行います。

![スクリーンショット 2022-12-28 1.08.35.png](MHSA%20(Multi-Head%20Self-Attention)%2041f6c97a219a47abb8a6899685ca1df3/%25E3%2582%25B9%25E3%2582%25AF%25E3%2583%25AA%25E3%2583%25BC%25E3%2583%25B3%25E3%2582%25B7%25E3%2583%25A7%25E3%2583%2583%25E3%2583%2588_2022-12-28_1.08.35.png)

埋め込みを含めたMHSAは以下の式になります。

$MHSA(z) = [SA_1(z);SA_2(z);\ldots ; SA_k(z)] W^0 \in \mathbb{R}^{(N_p + 1) \times D}$

$W^0 \in \mathbb{R}^{kD_h \times D}$：埋め込み

# 実装

```python
# ----------------------------
# 2-4 Self-Attention
# ----------------------------
print("=======2-4 Self-Attention=======")

class MultiHeadSelfAttention(nn.Module): 
    def __init__(self, emb_dim:int=384, head:int=3, dropout:float=0.):
        """ 
        引数:
            emb_dim: 埋め込み後のベクトルの長さ 
            head: ヘッドの数
            dropout: ドロップアウト率
        """
        super(MultiHeadSelfAttention, self).__init__() 
        self.head = head
        self.emb_dim = emb_dim
        self.head_dim = emb_dim // head
        self.sqrt_dh = self.head_dim**0.5 # D_hの二乗根。qk^Tを割るための係数

        # 入力をq,k,vに埋め込むための線形層。 [式(6)] 
        self.w_q = nn.Linear(emb_dim, emb_dim, bias=False) 
        self.w_k = nn.Linear(emb_dim, emb_dim, bias=False) 
        self.w_v = nn.Linear(emb_dim, emb_dim, bias=False)

        # 式(7)にはないが、実装ではドロップアウト層も用いる 
        self.attn_drop = nn.Dropout(dropout)

        # MHSAの結果を出力に埋め込むための線形層。[式(10)]
        ## 式(10)にはないが、実装ではドロップアウト層も用いる 
        self.w_o = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Dropout(dropout) 
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """ 
        引数:
            z: MHSAへの入力。形状は、(B, N, D)。
                B: バッチサイズ、N:トークンの数、D:ベクトルの長さ
        返り値:
            out: MHSAの出力。形状は、(B, N, D)。[式(10)]
                B:バッチサイズ、N:トークンの数、D:埋め込みベクトルの長さ
        """

        batch_size, num_patch, _ = z.size()

        # 埋め込み [式(6)]
        ## (B, N, D) -> (B, N, D)
        q = self.w_q(z)
        k = self.w_k(z)
        v = self.w_v(z)

        # q,k,vをヘッドに分ける [式(10)]
        ## まずベクトルをヘッドの個数(h)に分ける
        ## (B, N, D) -> (B, N, h, D//h)
        q = q.view(batch_size, num_patch, self.head, self.head_dim)
        k = k.view(batch_size, num_patch, self.head, self.head_dim)
        v = v.view(batch_size, num_patch, self.head, self.head_dim)

        ## Self-Attentionができるように、
        ## (バッチサイズ、ヘッド、トークン数、パッチのベクトル)の形に変更する 
        ## (B, N, h, D//h) -> (B, h, N, D//h)
        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)

        # 内積 [式(7)]
        ## (B, h, N, D//h) -> (B, h, D//h, N)
        k_T = k.transpose(2, 3)
        ## (B, h, N, D//h) x (B, h, D//h, N) -> (B, h, N, N) 
        dots = (q @ k_T) / self.sqrt_dh
        ## 列方向にソフトマックス関数
        attn = F.softmax(dots, dim=-1)
        ## ドロップアウト
        attn = self.attn_drop(attn)
        # 加重和 [式(8)]
        ## (B, h, N, N) x (B, h, N, D//h) -> (B, h, N, D//h) 
        out = attn @ v
        ## (B, h, N, D//h) -> (B, N, h, D//h)
        out = out.transpose(1, 2)
        ## (B, N, h, D//h) -> (B, N, D)
        out = out.reshape(batch_size, num_patch, self.emb_dim)

        # 出力層 [式(10)]
        ## (B, N, D) -> (B, N, D) 
        out = self.w_o(out) 
        return out
```

# 引用文献

[Transformers in Vision: A Survey](https://arxiv.org/abs/2101.01169)

[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)

[GitHub - ghmagazine/vit_book](https://github.com/ghmagazine/vit_book)

[コサイン類似度](https://www.cse.kyoto-su.ac.jp/~g0846020/keywords/cosinSimilarity.html)