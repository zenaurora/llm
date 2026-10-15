"""NaViT 学习笔记：从图片切块，一直走到 attention mask。

运行：.venv/bin/python src/learn/NaViT.py

阅读顺序就是 main() 中的执行顺序。这里仅演示一条打包序列，
没有训练、Q/K/V、多头注意力或完整 Transformer。
位置编码先用行、列 Embedding 查表理解，不是完整复现论文的位置编码。
"""

import torch
from torch import nn


def image_to_patches(image, patch_size):
    """[C, H, W] → [patch 数量, 每个 patch 的数值数量]。"""
    channels, height, width = image.shape
    # 这个学习版本只接受高宽能整除 patch_size 的图片，不进行裁剪或补齐。
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError("图片高宽必须能被 patch_size 整除")

    patch_rows = height // patch_size
    patch_cols = width // patch_size

    # 第一步：把高、宽分别拆成「patch 坐标 × patch 内部坐标」。
    # 原来：[通道, 高, 宽]
    # 现在：[通道, patch 行号, 内部行号, patch 列号, 内部列号]
    # 编号：   0       1          2          3          4
    split_image = image.reshape(
        channels, patch_rows, patch_size, patch_cols, patch_size
    )

    # 第二步：把两个 patch 坐标移到最前面。
    # 现在：[patch 行号, patch 列号, 通道, 内部行号, 内部列号]
    # 这样 grouped_patches[r, c] 就是一个包含全部通道的完整 patch。
    # permute 的参数是旧维度编号，不是各维度的新尺寸。
    grouped_patches = split_image.permute(1, 3, 0, 2, 4)

    # 第三步：前两维合并成 patch 编号，后三维合并成像素向量。
    # patch 按「从左到右，再从上到下」排列。
    # permute 后不保证内存连续，reshape 会在需要时复制；这里不直接用 view。
    patches = grouped_patches.reshape(
        patch_rows * patch_cols, channels * patch_size * patch_size
    )
    return patches, patch_rows, patch_cols


def main():
    # 固定随机种子：每次运行都能看到相同的随机参数和输出，方便对照。
    torch.manual_seed(42)
    torch.set_printoptions(precision=3, linewidth=120, sci_mode=False)

    # 一、准备两张尺寸不同的 RGB 图片。
    # A 保留你原来的数据，B 换成宽一些的图片。先用连续整数观察切块。
    images = [
        torch.arange(3 * 4 * 4).reshape(3, 4, 4),
        torch.arange(3 * 4 * 6).reshape(3, 4, 6),
    ]
    patch_size = 2
    d = 4  # 每个 token 用 4 个数表示，与 patch 数量无关。
    input_size = 3 * patch_size * patch_size  # 每个 RGB patch 有 12 个数。

    # 二、准备共享的参数。必须先初始化参数，再使用参数计算 token。
    # Linear 对最后一维做映射：[N, 12] → [N, 4]。
    # 所有图片、所有 patch 都使用同一个 projection，不在循环里重新创建。
    projection = nn.Linear(input_size, d)
    nn.init.normal_(projection.weight, mean=0.0, std=0.02)
    nn.init.zeros_(projection.bias)

    # Embedding 是「整数编号 → 向量」的查找表。
    # 本例最多支持 8 个 patch 行、8 个 patch 列，编号范围为 0～7。
    # 这是教学用的有限查找表，不代表已经支持任意大分辨率。
    row_embedding = nn.Embedding(8, d)
    col_embedding = nn.Embedding(8, d)
    nn.init.normal_(row_embedding.weight, mean=0.0, std=0.02)
    nn.init.normal_(col_embedding.weight, mean=0.0, std=0.02)

    token_groups = []
    image_id_groups = []

    # 三、分别把每张图片变成带位置的 token。
    for image_id, image in enumerate(images):
        patches, patch_rows, patch_cols = image_to_patches(image, patch_size)
        n = patches.shape[0]
        print(f"\n图片 {image_id}：{list(image.shape)} → patches {list(patches.shape)}")
        print("第一个通道的原始像素：\n", image[0])
        print("各 patch 的全部像素（每行一个 patch，先 R 再 G 再 B）：\n", patches)

        # 原始整数先变成浮点数，才能与 Linear 的浮点参数进行运算。
        # 除以 100 只是缩小这个假例子的数值范围，方便看内容与位置的相加。
        # 它不是 F.normalize，也不是 NaViT 必须采用的图片预处理规则。
        content_tokens = projection(patches.float() / 100.0)

        # 不再对 token 使用 F.normalize：它会把向量缩放到单位长度，
        # 当前理解切块、位置和 mask 并不需要它。

        # meshgrid 把两组坐标展开成二维网格。
        # B 的 patch 网格为 2×3，生成：
        # row_grid = [[0, 0, 0], [1, 1, 1]]
        # col_grid = [[0, 1, 2], [0, 1, 2]]
        row_grid, col_grid = torch.meshgrid(
            torch.arange(patch_rows),
            torch.arange(patch_cols),
            indexing="ij",
        )
        # reshape(-1) 展平成一维，-1 让 PyTorch 推算元素数量。
        # 它与切 patch 的顺序一致，因此每个 token 能拿到正确的坐标。
        row_ids = row_grid.reshape(-1)
        col_ids = col_grid.reshape(-1)
        print("row_ids:",row_ids)
        print("col_ids:",col_ids)
        coordinates = torch.stack([row_ids, col_ids], dim=1)
        # stack 新增一维：[N] 和 [N] → [N, 2]，每行是一个 (行号, 列号)。
        print("patch 坐标：\n", coordinates)

        # 查行表得到 [N, D]，查列表也得到 [N, D]。
        # 相加得到位置向量，再加到内容向量上，形状仍然是 [N, D]。
        position_tokens = row_embedding(row_ids) + col_embedding(col_ids)
        tokens = content_tokens + position_tokens
        print("内容向量：\n", content_tokens)
        print("位置向量：\n", position_tokens)
        print("内容 + 位置：\n", tokens)
        token_groups.append(tokens)

        # torch.full((n,), image_id) 创建长度为 n、每项都是 image_id 的向量。
        # A 有 4 个 token → [0, 0, 0, 0]
        # B 有 6 个 token → [1, 1, 1, 1, 1, 1]
        # 图片编号是生成 mask 的元数据，不会加进 token 特征里。
        # torch.full(形状, 填充值, dtype=数据类型)
        image_id_groups.append(torch.full((n,), image_id, dtype=torch.long))

    # 四、打包：沿已有的第 0 维，把 token 序列接起来。
    # cat：[4, 4] 与 [6, 4] → [10, 4]。
    # stack 是新增维度，且要求输入形状相同，所以这里用 cat。
    packed_tokens = torch.cat(token_groups, dim=0)
    image_ids = torch.cat(image_id_groups, dim=0)
    print("\n打包后的 token 形状：", list(packed_tokens.shape))
    print("打包后的 token：\n", packed_tokens)
    print("各 token 所属的图片：", image_ids)

    # 每张图片的位置坐标都从自己的 (0, 0) 开始。
    # 拼接顺序不会改变 patch 的原图坐标，也不需要把 B 的位置接着 A 编号。

    # 五、根据图片编号创建 attention mask。
    # 将来 attention scores 的形状是 [L, L]，本例 L=10：
    # 第 i 行表示「token i 正在关注谁」，第 j 列表示「被关注的 token j」。
    # 我们要求：只有属于同一张图片的 i、j 才可以相互关注。

    # unsqueeze(k) 在第 k 个位置插入一个大小为 1 的维度，不改变数值。
    query_image_ids = image_ids.unsqueeze(1)  # [10] → [10, 1]，一列编号
    key_image_ids = image_ids.unsqueeze(0)    # [10] → [1, 10]，一行编号
    print("key_image_ids:",key_image_ids)
    # 广播：一列与一行比较，得到整张 10×10 的比较表。
    # allow_mask[i, j] = (image_ids[i] == image_ids[j])。
    # 本脚本约定 True 表示允许关注，False 表示禁止关注。
    allow_mask = query_image_ids == key_image_ids # 张量的逐元素比较， == 会触发广播，广播是对应的维度大小相同/或者其中一个是 1 就可以比较
    # 现在一个是 10,1， 一个是 1,10，所以自然可以比较，然后把二者形状都变成 10，10 来进行比较
    # [  [0,0,1]  ] => [ [0,0,0],[0,0,0],[1,1,1] ] 
    print("\n允许关注的 mask（1=允许，0=禁止）：\n", allow_mask.int())

    # 应当看到左上 4×4、右下 6×6 两块为 1，其余为 0：
    #             A 的 token    B 的 token
    # A 的 token      允许          禁止
    # B 的 token      禁止          允许
    # 同一张图片中可以双向关注，也可以关注自己；这里不是语言模型的因果 mask。
    # 本例没有 padding，因此也暂时不需要处理 padding mask。

    # 六、把「是否允许」转成可加到 attention scores 上的数值。
    # 允许的位置加 0，禁止的位置加负无穷，之后才做 softmax。
    # ~allow_mask 是布尔取反：原来禁止的位置变成 True。
    # masked_fill(mask, value) 把 mask 为 True 的位置填成 value。
    additive_mask = torch.zeros_like(allow_mask, dtype=packed_tokens.dtype)
    additive_mask = additive_mask.masked_fill(~allow_mask, float("-inf"))
    print("\n数值形式的 mask（0=保留分数，-inf=屏蔽分数）：\n", additive_mask)

    # 不同 PyTorch 接口对布尔 mask 的约定可能相反：
    # F.scaled_dot_product_attention：True 表示允许。
    # nn.MultiheadAttention 的布尔 attn_mask：True 表示禁止。
    # 所以不能只凭变量叫 mask 就直接传进去，必须确认接口的定义。

    # 七、用一行随机分数观察 mask 的效果，不在这里展开 Q/K/V。
    # 真正的 attention scores 来自 Q @ K.T / sqrt(d_k)。
    # 这里只模拟 token 0 对全部 10 个 token 的分数，分数不是概率。
    scores = torch.randn(packed_tokens.shape[0])
    masked_scores = scores + additive_mask[0]

    # softmax 把分数转换成权重：exp(score) / 所有 exp(score) 之和。
    # exp(-inf)=0，因此另一张图片的 token 得到严格为 0 的权重。
    # 必须在 softmax 之前屏蔽；直接把禁止位置的分数设成 0 无效，exp(0)=1。
    # 也不要把分数乘以布尔 mask，乘 0 仍然没有排除它们。
    weights_before = torch.softmax(scores, dim=-1)
    weights_after = torch.softmax(masked_scores, dim=-1)
    print("\ntoken 0 对全部 token 的随机分数：\n", scores)
    print("加 mask 后的分数：\n", masked_scores)
    print("不加 mask 的注意力权重：\n", weights_before)
    print("加 mask 后的注意力权重：\n", weights_after)
    print("加 mask 后的权重总和：", weights_after.sum().item())

    # 一个有效 query 至少要有一个允许位置，否则全为 -inf 时 softmax 会产生 NaN。
    # 本例允许关注自己，因此不会出现全被屏蔽的行。
    # 这两个断言核实教学例子的效果：跨图片权重为 0，剩余权重总和为 1。
    assert torch.all(weights_after[~allow_mask[0]] == 0)
    assert torch.isclose(weights_after.sum(), torch.tensor(1.0))


if __name__ == "__main__":
    # 仅观察前向数值，无需记录计算图；这不会修改随机初始化的参数。
    with torch.no_grad():
        main()
