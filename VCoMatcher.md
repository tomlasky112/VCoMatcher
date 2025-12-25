VGGT 监督 CoMatcher（数据引擎与蒸馏）

version: 1.2
date: 2025.12.02->2025.12.25
author: Xinmiao Shao

**更新日志**:
- **v1.2 (2025.12.25)**: 
  - 新增 AMP + no_grad 高效在线蒸馏策略（显存 -60%，速度 +50-100%）
  - 改进 $W_{src}$ 权重计算：Quantile + Sigmoid 替代线性归一化（抗 outlier）
  - 改进几何引导图构建：半径约束 + 膨胀掩膜替代硬 k-NN（更鲁棒）
  - 改进 $\mathcal{L}_{rel}$ 关系蒸馏：教师置信度门控（过滤不可靠点对）
  - 改进课程学习：基于验证性能的动态调度替代固定 Epoch（自适应+回滚机制）
- **v1.1 (2025.12.18)**: 初始完整设计

---
**方案概述**：借鉴了 MatchGS 的思想，利用 VGGT 替代 3DGS 生成包含困难样本的密集标签，使用表征对齐策略训练 CoMatcher 在半密集匹配上具备显式的 3D 感知能力。但是本方案中的 VGGT 并没有像 MatchGS 中的 3DGS 那样直接渲染生成新的训练数据，本方案使用 VGGT 仅在现有训练数据中生成几何标签

---
# 第一阶段：基于 VGGT 的高保真半密集匹配训练数据生成
我们用 VGGT 替换 3DGS，利用其极快的推理速度和强大的几何泛化能力来生成“困难样本”的 Ground Truth ，注意我们使用的是离线预处理，将输出的伪真值存储在硬盘中（`.npz` 或 `.h5` 格式），在训练时直接读取。
### 1. CoMatcher 训练数据需求
CoMatcher 训练所需要的核心数据：
- **图像 (Images)**：统一尺寸的 RGB 图像。
- **相机参数 ($g_i$)**：内参（注意归一化和反归一化）和位姿（注意将 VGGT 输出的 $T_{c2w}$ 求逆转换为 CoMatcher 需要的 $T_{w2c}$）
- **深度图 (Depth Maps)**：每张图像对应的稠密深度图（注意 Tensor 转 Numpy）。
- **重叠矩阵 (Overlap Matrix)**：描述图像对之间共视程度的 $N \times N$ 矩阵，用于采样训练组。

### 2. VGGT 推理与基础数据获取
1. **输入**：未标注的同一场景下的多视图图像集 $\mathcal{I} = \{I_1, ..., I_N\}$（如 ScanNet, MegaDepth，它们已经按照场景进行了文件夹分类）。

2. **预处理**：遵循 VGGT 的输入要求，通常将图像 resize 至长边为 518 像素左右，并进行必要的归一化 。

3. **VGGT 推理**：将图像序列输入预训练且冻结的 VGGT 模型 $\Phi_{VGGT}$ ，VGGT 可以在单次前向传播中（<1秒）预测出所有帧的**相机参数 $g_i$** 、**深度图 $D_i$** 、**点图 $P_i \in \mathbb{R}^{3 \times H \times W}$** 、**跟踪特征** $T_i \in \mathbb{R}^{C \times H \times W}$ 和**不确定性图** $\Sigma_i^P, \Sigma_i^D$ ，将这 $N$ 组数据保存到一个 `.npz` 或 `.h5` 文件中（==注意：跟踪特征$T_i \in \mathbb{R}^{C \times H \times W}$ 不保存，因为过大会使得 I/O 时间影响训练时间==），此时保存的位姿为基于场景原点的绝对位姿。，在训练时我们需要进行位姿归一化来避免“**坐标系陷阱**”。
$$(g_i, D_i, P_i,T_i, \Sigma_i)_{i=1}^N = \Phi_{VGGT}(\mathcal{I})$$

4. **计算重叠矩阵**：计算任意两张图像 $I_i$ 和 $I_j$ 之间的重叠分数 $O_{ij}$ ，表示图像 $I_i$ 中有多少像素可以成功投影到图像 $I_j$ 中，并且未被遮挡，最终得到一个重叠矩阵，其中元素 $O_{ij}$ 表示图像 $I_i$ 到图像 $I_j$ 的重叠分数（Overlap Score），通常 $O_{ij} \in [0, 1]$ 。CoMatcher 训练非常依赖这个矩阵来采样共视图像组，利用 VGGT 输出的 $P_i \in \mathbb{R}^{3 \times H \times W}$ 可以轻松计算：
	1. **3D 坐标提取(World Coordinate Extraction)**：对于源图像 $I_i$ 中的每一个有效像素 $u = (u_x, u_y)$ ，首先需要剔除天空或无效深度的像素，在这里我们使用了三步来进行剔除：
		- 利用 VGGT 的不确定性图 $\Sigma_{i}^{P}$ 初步剔除无效像素，即只保留 $\Sigma$ 值小于阈值 $\tau$ 的像素（如天空、物体边缘的伪影）：
		  $$M_{valid}(u) = \mathbb{I}(\Sigma_{i}^{P}(u) < \tau)$$
		- 基于深度图阈值进行剔除，因为天空通常对应于深度极大值，而无效像素可能为 0、负值或 NaN。**$\tau_{min}$ (近平面阈值，设置为一个较小的正数），$\tau_{max}$ (远平面/天空阈值，设置为一个较大的数值）：
		  $$M_{depth}(u) = (D_i(u) > \tau_{min}) \land (D_i(u) < \tau_{max}) \land \neg \text{isnan}(D_i(u))$$
		- 使用一致性检查进行剔除，即检查深度图 $D$ 和点图 $P$ 两者的一致性。
			首先计算点图导出的深度 $d_{P}$（假设 $T_{w2c} = [R|t]$）：
			$$X_{cam} = R \cdot \mathcal{P}_i(u) + t$$                      $$d_{\mathcal{P}} = X_{cam}^{(z)}$$
			随后定义**一致性掩膜 $M_{consist}(u)$**：
			$$M_{consist}(u) = | D_i(u) - d_{\mathcal{P}} | < \epsilon \cdot D_i(u)$$
		最终，对于图像 $I_i$ 中的像素 $u$，我们结合上述所有条件生成最终的 **有效像素掩膜 (Final Valid Mask)** 作为`mask_loss`它在后续训练时也用于计算 Loss：
		$$\mathbb{I}_{valid}(u) = M_{valid}(u) \land M_{depth}(u) \land M_{consist}(u)$$

	2. **几何重投影 (Geometric Reprojection)**：世界点→相机坐标（外参）→齐次图像坐标（内参）→像素坐标（透视除法）
		对于 $I_i$ 中的每个有效像素 $u$，利用 VGGT 的点图 $P_i$ 获取其世界坐标：
		$$X_u = {P}_i(u)$$
		
		随后从世界坐标投影到 $I_j$ 下的相机齐次图像坐标：
		==**12.18修改**：务必确保在预处理阶段 将 VGGT 输出的 Pose（通常是 c2w）求逆变成了 w2c，并且 $P_i$​ 的坐标系与 Pose 的世界坐标系定义是完全对齐的。==
		$$\tilde{u}_{i \to j} = K_j \cdot (R_j X_u + t_j)$$
		 其中
			$T_j = [R_j | t_j]$：相机 $j$ 的**外参**（旋转 + 平移），计算 $R_j X_u + t_j$ ，就能把 $X_u$ 转换为**相机 $j$ 坐标系下的 3D 点**。
			$K_j$：相机j的**内参矩阵**，它会把 “相机坐标系下的 3D 点” 转换为**齐次形式的图像坐标 $\tilde{u}_{u \to j}$** 。
	
		最后使用透视除法得到像素坐标	：	 	
		$$u_{i\rightarrow j}=[\tilde{u}_{i\rightarrow j}^{(x)}/\tilde{u}_{i\rightarrow j}^{(z)}, \quad \tilde{u}_{i\rightarrow j}^{(y)}/\tilde{u}_{i\rightarrow j}^{(z)}]^T$$
		 其中
			$\pi$ 是**透视除法**：因为齐次图像坐标是 “带深度信息的”，需要除以第三个分量 $\tilde{u}_z$（对应相机坐标系下的深度），才能得到最终的 **2D 像素坐标**$u_{u \to j}$（即图像里的 $(x,y)$ 像素位置）。

    3. **可见性检查 (Visibility Check)**：定义指示函数 $\mathbb{I}_{vis}(u)$，判断该点在 $I_j$ 中是否可见。需同时满足以下三个条件：
		 - **视野范围内 (Field of View, FoV)**：投影点必须在图像边界内：
			$$C_{fov} = (0 \le u_{i \to j}^{(x)} < W) \land (0 \le u_{i \to j}^{(y)} < H) \land (\tilde{u}_{i \to j}^{(z)} > 0)$$
		- **遮挡检测 (Occlusion Check)**：==**12.18修改：**==
	      首先定义**有向深度差 (Signed Depth Difference)**：
	      $$\Delta{d} = d_{proj} - d_{obs}$$
	      定义非对称动态阈值 (Asymmetric Dynamic Thresholds)：我们使用观测深度 $d_{obs}$ 作为基准尺度，并引入绝对容忍项 $\delta_0$。
			- 防遮挡上限 (Occlusion Upper Bound)：
			 $$\tau_{upper}(d_{obs}) = \alpha \cdot d_{obs} + \delta_0$$
			- 防噪声下限 (Noise Lower Bound)：
             $$\tau_{lower}(d_{obs}) = \gamma \cdot d_{obs} + \delta_0$$
          最后得到
		  $$C_{occ}=-(γ\cdot d_{obs}+δ_0)≤\Delta{d}≤α\cdot d_{obs}+δ_0$$
			其中
			 - **投影深度**：$d_{proj} = \tilde{u}_{i \to j}^{(z)}$ 
			 - **观测深度**：$d_{obs} = D_j(u_{i \to j})$ （通常通过双线性插值 `grid_sample` 获取，==注意必须先将像素坐标归一化到 $[-1, 1]$ 区间，且注意 $(x, y)$ 也就是 $(u, v)$ 的顺序。==）
			 - **$\alpha$ (遮挡系数)**: 必须小，比如 `0.01` (1%)
			 - **$\gamma$ (噪声/悬空系数)**: 可以稍大，比如 `0.03` ~ `0.05` (3%-5%)
			 - **$\delta_0$ (绝对底噪)**: 根据场景尺度定
				 - 室内 (ScanNet/Indoor): $\delta_0 = 0.03 \sim 0.05$ (3cm - 5cm)
				 - 室外 (MegaDepth/Outdoor): $\delta_0 = 0.1 \sim 0.5$ (10cm - 50cm)
		- **3D 一致性检查 (3D Consistency Check, VGGT特有)**：由于 VGGT 提供了稠密 3D 点图，我们还可以进行更严格的检查：投影位置预测的 3D 点应该和原始 3D 点是同一个点：
			$$X_{obs} = \mathcal{P}_j(u_{i \to j})$$
		  ==**12.18修改：** 注意 $X_{obs}$ 也需要使用双线性插值获取==
			$$C_{consist} = \| X_u - X_{obs} \|_2 < \epsilon_{dist}$$
		  ==注意 $ε_{dist}$ 随深度做尺度归一（比如 $τ(d) = τ0 + τ1 * d_{obs}，τ0=0.05 \sim 0.1，τ1=0.02 \sim 0.05$）==
	   最后基于上述三种可见性检查得到综合可见性，通过可见性检查的点才算可见：
	   $$\mathbb{I}_{vis}(u) = C_{fov} \land C_{occ} \land C_{consist}$$
		
	4. **重叠分数公式 (Aggregation Formula)**：有了每个像素的可见性状态后，我们可以计算重叠分数。这里主要有两种计算方式，CoMatcher 训练通常推荐使用 **IoU (Intersection over Union)** 或 **覆盖率 (Coverage)**
		- **方式 A 单向覆盖率 (Coverage Score)**：衡量 $I_i$ 有多少比例的区域在 $I_j$ 中可见。这是最常用的指标。
			$$O_{ij} = \frac{\sum_{u \in \text{valid}(I_i)} \mathbb{I}_{vis}(u)}{N_{valid}(I_i)}$$
			其中
			 $N_{valid}(I_i)$：$I_i$ 中具有有效深度/3D 坐标的总像素数。
		     注意：$O_{ij}$ 不一定等于 $O_{ji}$ 。
		- **方式 B 双向 IoU (Intersection over Union)**：衡量两张图的共同视野占总视野的比例，这是一个对称指标。

			$$O_{ij}^{IoU} = \frac{|S_i \cap S_j|}{|S_i \cup S_j|}$$
			其中
			 $|S_i \cap S_j|$ 近似为 $\sum \mathbb{I}_{vis}$。在稠密计算中，通常简化为：
			$$O_{ij}^{IoU} \approx \frac{\text{Shared Pixel Count}}{\text{Valid Pixel Count}_i + \text{Valid Pixel Count}_j - \text{Shared Pixel Count}}$$
			
    5. **矩阵生成与保存**：遍历所有图像对 $(i, j)$，填充矩阵 $\mathbf{O}$。
		$$\mathbf{O} = \begin{bmatrix} 1.0 & O_{12} & \cdots & O_{1N} \\ O_{21} & 1.0 & \cdots & O_{2N} \\ \vdots & \vdots & \ddots & \vdots \\ O_{N1} & O_{N2} & \cdots & 1.0 \end{bmatrix}$$
		
	6. 同时为了构建 GNN 初始图，我们还需要计算 `geometry_mask`，这是一个较为宽松的掩码，因为我们需要在 3D 空间中确定可以用来建立连接的点， 我们保留物理上存在的区域，剔除无效深度，极近噪点：
	   $$M_{geom}(u) = (D_i(u) > \tau_{min}) \wedge (D_i(u) < \tau_{max}) \wedge \neg \text{isnan}(D_i(u))$$

### 3. 获取训练视图对`CoMatcher/src/datasets/multiview/multiview_megadepth.py`
在使用 VGGT 生成了图像的几何信息后，主要依赖基于共视性的采样策略获取训练用的源视图和目标视图。我们使用 VGGT 提供的全场景 3D 关联（重叠矩阵）使用采样算法从中挖掘出 $(I_t, \{I_{s1}, I_{s2}, I_{s3}\})$ 结构的训练对，最后使用 DataLoader 读取对应的图像文件送入网络。

##### **基础数据准备**
此时已经对一个场景（Scene）完成了 VGGT 推理和处理，拥有以下数据：
- **图像索引**：$0, 1, \dots, N-1$。
- **重叠矩阵 ($O$)**：$N \times N$ 的矩阵，其中 $O_{ij}$ 表示图像 $i$ 和 $j$ 的共视分数（值域 $[0, 1]$）。

##### **采样策略 (Sampling Strategy)**
CoMatcher 的训练通常以 **四元组 (Quadruplet)** 为单位：包含 1 张目标视图 ($I_t$) 和 3 张源视图 ($I_{s1}, I_{s2}, I_{s3}$)。
**第一步：确定候选对 (Filtering Candidate Pairs)**
- 首先，我们需要筛选出所有“合格”的图像对。
- **标准**：重叠分数必须在一定范围内。
	- **不能太小**（例如 $<0.05$）：如果重叠太少，根本无法匹配，网络学不到东西。        
    - **不能太大**（例如 $>0.7$）：如果重叠太大（几乎一样的视角），任务太简单，网络也学不到鲁棒性。
 - **操作**：遍历重叠矩阵，找到所有满足 $0.05 < O_{ij} \le 0.7$ 的索引对 $(i, j)$，标记为“好对（Good Pairs）”。
**第二步：构建共视组 (Forming Co-visible Groups)**
CoMatcher 需要构建一个**中心化**的结构：选定一个 $I_t$，然后找 3 个能同时看到 $I_t$ 内容的 $I_s$。代码中的具体逻辑如下（`sample_new_items` 函数）：
1. **寻找相互关联的源视图**：首先在“好对”中寻找三元组 $(I_1, I_2, I_3)$，使得它们之间有一定的共视关系（这保证了源视图之间也能进行几何交互，这是 CoMatcher 的核心特性）。	
2. **寻找公共目标 ($I_t$)**：对于选定的源视图组合 $(I_1, I_2, I_3)$，在所有图像中寻找一个 $I_0$（即 $I_t$），使得 $I_0$ 与这三张图的重叠分数都满足“合格”标准。
	 - 即：$Good(I_0, I_1)$ AND $Good(I_0, I_2)$ AND $Good(I_0, I_3)$。
3. **确定角色**：
	 - **目标视图 ($I_t$)**：即找到的 $I_0$。
	 - **源视图列表 ($I_s$)**：即 $[I_1, I_2, I_3]$。

##### 最终输出
###### 预处理（在线处理）：
为了解决 VGGT 的**坐标系陷阱**问题，我们还需要在 DataLoader 的 `__getitem__` 或者 `collate_fn` 中进行数据层面的几何变换，需要将所有训练样本（Target 和 Sources）的位姿和点云，从“世界坐标系”转换到“Target 相机坐标系”。
1. 计算锚点变换矩阵
   找到当前训练组中目标视图 ($I_t$) 的世界位姿 $T_{world}^{(t)}$，计算其逆矩阵：
   $$M_{anchor} = (T_{world}^{(t)})^{-1}$$
    - 这个 $M_{anchor}$ 是一个 World-to-Target 的变换矩阵。
2. 转换所有相机的位姿
   对于组内的每一张图 $I_k$（包括 Target 自己和所有 Sources），计算新位姿：
   $$T_{new}^{(k)} = M_{anchor} \cdot T_{world}^{(k)}$$
	- **检查点**：计算后，Target 的新位姿 $T_{new}^{(t)}$ **必须** 是单位矩阵（Identity Matrix）。
	- **物理含义**：现在所有相机的位姿都是相对于 Target 的了。
	- **数学本质**：它表示 **源相机 (Source) 在 目标相机 (Target) 坐标系下的位姿**
3. 转换点图 (Point Maps)
   如果使用了 VGGT 生成的点图 $P_{world}$（$3 \times H \times W$），也必须应用同样的刚体变换。
   设 $M_{anchor}$ 的旋转部分为 $R_{anchor}$，平移部分为 $t_{anchor}$：
   $$P_{new}(u,v) = R_{anchor} \cdot P_{world}(u,v) + t_{anchor}$$
	- **物理含义**：原本在世界坐标系中的点云，现在变成了在 Target 相机坐标系下的坐标。

经过上述算法筛选后，DataLoader 的 `__getitem__` 最终返回：
- **$I_t$ (Target View)**：读取索引为 $i_0$ 的图像，确保 Target View 永远是列表的第一个元素。
- **$I_s$ (Source Views)**：读取索引为 $i_1, i_2, i_3$ 的图像列表。
- **相机参数 (Camera Parameters)**：内参 $K$ 和 外参 $T_{w2c}$
- **几何真值 (Geometric Ground Truth)**：**深度图 ($D_i$)** 和 **点图 ($P_i$)**，后续在训练时进行“在线”计算匹配真值
- **不确定性/权重图 (Uncertainty/Weight Maps)**：即 $W_{src}$ 和 $\Sigma_i$
- **有效性掩膜 (Valid Masks)**：$\mathbb{I}_{valid}$ ，用于指示哪些区域是天空或无效区域，用于计算 Loss 
- `geometry_mask`：$M_{geom}$ ，用于构建 GNN 初始图
这些图像随后会被读取、Resize（例如到 1024 边长）、归一化，并送入网络进行训练。

# 注意事项
第一阶段提到要生成“基于场景原点的绝对位姿”。但在实际操作中，如果一个场景有 100 张图片（ScanNet, MegaDepth），直接把它们一次性送入 VGGT（VGGT 模型接受 [B, S, 3, H, W] 输入）可能会导致显存爆炸 (OOM)。分批（Batch）推理（为了充分利用 80g A100 的算力，我们这里使用 32 帧滑动窗口，其中使用 8 帧重叠，我们利用这 8 帧重叠窗口作为”榫卯接口“计算出如何旋转、平移、缩放 Window 2，让它和 Window 1 完美咬合）。但需要注意 VGGT 输出的位姿是相对于当前序列第一帧的，需要对齐（我们利用 8 帧重叠窗口作为”榫卯接口“计算出如何旋转、平移、缩放 Batch 2 , 让它和 Batch 1 完美咬合）。
##### 1. 窗口划分
我们借鉴了**视觉里程计 (Visual Odometry)** 的前端对齐思想，在 VRAM 受限的情况下，将碎片化的局部轨迹拼接成为一个长距离的全局轨迹。
我们先切分数据：
- **Batch A (基准):** `Idx [0, 1, ..., 31]` (共32帧)
- **Batch B (待拼接):** `Idx [24, 25, ..., 55]` (共32帧)
    - **注意：** 两个 Batch 共享了 `Idx 24~31` 这 8 帧。这就是 **锚点 (Anchor)**。

##### 2. VGGT 独立推理
将两个 Batch 分别送入 A100 进行推理。VGGT 会输出**局部坐标系**下的位姿：
- **Output A ($P_A$):** 以第 0 帧为原点（Identity）。
    - 包含 $P_{A,0}, ..., P_{A,31}$。
- **Output B ($P_B$):** 以第 24 帧为原点（Identity）。
    - 包含 $P_{B,24}, ..., P_{B,55}$。
**此时的矛盾：**
在 Batch A 中，第 24 帧的位姿可能是 $(x=5, y=2)$。
在 Batch B 中，第 24 帧的位姿是 $(x=0, y=0)$（因为它是 B 的首帧）。
它们不在一个世界里。

##### 3.刚体/相似变换 (Sim3 Alignment)
我们要找到一个变换矩阵 $T_{A \leftarrow B}$，把 B 的坐标系变成 A 的坐标系。
**输入数据（重叠的 8 帧）：**
- **目标点集 (Target / Global):** Batch A 中的第 24~31 帧的位姿中心 (Translation)。记为 $\{ t_{A, i} \}_{i=24}^{31}$。
- **源点集 (Source / Local):** Batch B 中的第 24~31 帧的位姿中心。记为 $\{ t_{B, i} \}_{i=24}^{31}$。
算法 (Umeyama):
这是一个经典的最小二乘问题。我们要解出 缩放因子 $s$，旋转矩阵 $R$，平移向量 $t$，使得下式最小：
$$\min_{s, R, t} \sum_{i=24}^{31} || t_{A, i} - (s \cdot R \cdot t_{B, i} + t) ||^2$$

##### 4. 全局拼接 (Global Stitching)
计算出 $T_{A \leftarrow B} = (s, R, t)$ 后，我们将它应用到 **Batch B 的所有帧（包括重叠帧和新帧 32-55）**：
$$P_{B, k}^{global} = T_{A \leftarrow B} \times P_{B, k}^{local}$$
现在，Batch B 的第 32~55 帧（新区域）就拥有了和 Batch A 一致的全局坐标。

##### 5. 线性平滑 (Linear Blending / Interpolation)
在重叠区域（24~31帧），我们现在有了两套位姿：
1. 来自 A 的原始位姿。
2. 来自 B 变换后的位姿。
为了防止拼接处出现微小的“跳变”（Seam），我们做加权融合：
- **Frame 24:** $100\% P_A + 0\% P_B'$ (完全信赖上一段)
- **Frame 27:** $57\% P_A + 43\% P_B'$ (混合)
- **Frame 31:** $0\% P_A + 100\% P_B'$ (完全信赖下一段，为连接再下一段做准备)

---
# 第二阶段：基于 VGGT 的几何知识蒸馏
在第一阶段，我们利用 VGGT 生成了伪真值（Pose, Depth）用于几何监督。在第二阶段，我们的目标是**特征空间的对齐（Feature Space Alignment）**。我们不再仅仅让 CoMatcher 学习“点 A 匹配点 B”，而是要让 CoMatcher 的**特征描述符（Descriptor）直接具备 VGGT 特征的几何鲁棒性**和**视点不变性**。我们将 VGGT 视为一个“冻结的 3D 教师”，强迫 CoMatcher 这个“2D 学生”学习其高维特征表达。
==注意：==邻接矩阵 $\mathcal{A}_{geom}$ 并非第一阶段预处理产物。在训练迭代中，我们读取第一阶段提供的稠密深度图 $D$ 和 位姿 $T$，在 GPU 上实时执行投影和近邻搜索来构建 $\mathcal{A}_{geom}$。这种在线计算方式避免了巨大的存储开销，并允许动态调整关键点采样策略。

---
### 1. 网络架构设计
##### 1.1 教师网络 (Teacher): Frozen VGGT
- **输入**: 图像序列 $\mathcal{I} = \{I_1, I_2, ...\}$ ，注意必须将`Target View`放在`List`的第一个位置`input_imgs = [Target, Source1, Source2，Source3]`。
- **输出**: 
	- **稠密跟踪特征图** $T_i \in \mathbb{R}^{C_{vggt} \times H \times W}$ 。根据 VGGT 论文，这些特征 $T_i$ 被用于 "Tracking-Any-Point"，具备极强的跨视角一致性
	- **点图 (Point Maps)** $P_{vggt}$：用于辅助验证几何一致性
	- **深度图 (Depth Maps)** $D_{vggt}$：辅助进行遮挡判断（Visibility Check）和几何一致性验证
	- **相机参数 (Camera Parameters) $g_i$**：用于将点图重投影到其他视图，验证匹配准确性
	- **不确定性图 (Uncertainty Maps) $\Sigma_{i}$**：用于计算源感知权重 $W_{src}$
- **状态**: 参数完全冻结，仅做推理，前向传播中提供监督信号。
##### 1.2 学生网络 (Student): CoMatcher
- **输入**: 相同的图像序列。
- **输出**:
    - 稀疏/半稠密关键点 $k_i$ 
    - 原始特征描述符 $F_{co} \in \mathbb{R}^{C_{co} \times N}$（$N$ 为关键点数量）
    - 匹配概率/相关性矩阵 $\mathcal{S}_{co}$ 
- **状态**: 参数可训练。
##### 1.3 特征适配器 (Feature Adapter)
由于 VGGT 特征维度 $C_{vggt}$ 与 CoMatcher 特征维度 $C_{co}$ 可能不同，且语义空间存在差异，我们需要一个轻量级的投影头（Projection Head $\phi(\cdot)$）将两者映射到同一空间。
- **结构**: 简单的 MLP (Linear -> ReLU -> Linear)：`Linear(C_vggt,C_mid) -> BatchNorm -> ReLU -> Linear(C_mid,C_co)`
- **作用**: 将 $F_{co}$ 映射为 $F'_{co}$，使其能与 $T_{vggt}$ 进行对比学习

---
### 2. 蒸馏策略
我们采用“跨模态特征对齐”和“关系知识蒸馏”两种策略相结合。
##### 2.1 策略一：跨模态特征对齐（Cross-Modal InfoNCE Alignment）
**目的**: 确保 CoMatcher 提取的每一个关键点特征，都与其在 VGGT 特征图上对应位置的特征高度相似。也就是让 CoMatcher 提取的特征在未进行图神经网络（GNN）交互前，就已经包含了 VGGT 理解的 3D 几何信息。
###### 步骤：
1. **采样 (Sampling)**：**采样**: 对于 CoMatcher 在图像 $I_A$ 中提取的第 $k$ 个关键点 $p_k = (u_k, v_k)$，我们在 VGGT 的特征图 $T_A$ 上进行双线性插值采样，得到对应的教师特征 $t_k^{A}$ 。 
2. **投影（Projection）**：将 CoMatcher 特征 $f_k^{A}$ 通过特征适配器 $\phi(\cdot)$ 投影：$\hat{f}_k^{A} = \phi(f_k^{A})$ 。
3. **损失函数（InfoNCE Loss）**：使用对比损失来拉进“学生特征”和“教师特征”的距离。对比损失本质是最大化正样本相似度的对数概率，同时最小化负样本的干扰。
   对于每一个匹配对 $(p_k^A, p_k^B)$（由第一阶段生成的 GT 确定）：
	- **Anchor**: VGGT 的“教师特征” $t_k^{A}$ (视为 3D Ground Truth 表示)。
    - **正样本 Positive**: CoMatcher 输出的图像 A 和 B 的特征随后进入特征匹配器进行投影得到 $\hat{f}_k^{A}$ 和 $\hat{f}_k^{B}$。
    - **负样本 Negative**: 同一 Batch 内其他不相关的关键点特征。
   **损失函数公式**：
	$$\mathcal{L}_{align} = - \frac{1}{|\mathcal{M}|} \sum_{k \in \mathcal{M}} \log \frac{\exp(\text{sim}(t_k^{A}, \hat{f}_k^{A}) / \tau)}{\sum_{j \in \mathcal{N}} \exp(\text{sim}(t_k^{A}, \hat{f}_j^{A}) / \tau)}$$
	$$\mathcal{L}_{align} = - \frac{1}{|\mathcal{M}|} \sum_{k \in \mathcal{M}} \log \frac{\exp(\text{sim}(t_k^{B}, \hat{f}_k^{B}) / \tau)}{\sum_{j \in \mathcal{N}} \exp(\text{sim}(t_k^{B}, \hat{f}_j^{B}) / \tau)}$$
	其中
	- $\text{sim}(u, v) = \frac{u \cdot v}{\|u\| \|v\|}$ 是余弦相似度，值越大表示 $u$ 和 $v$ 越相似
	- $\tau$ 是温度系数 ($0.07/0.1$)

##### 2.2 策略二：关系知识蒸馏（Relation Knowledge Distillation, RKD）
**目的**: 仅仅对齐特征是不够的，我们需要对齐**特征之间的相互关系**。即：CoMatcher 认为点 A 和点 B 的相似度，应该与 VGGT 认为的相似度一致。
###### 步骤：
1. **构建教师相关性矩阵**: 利用 VGGT 的特征计算所有采样点之间的相关性矩阵 $S_{vggt} \in \mathbb{R}^{N \times N}$：
   $$S_{vggt}(i, j) = \text{softmax}\left( \frac{t_i^A \cdot t_j^B}{\sqrt{C}} \right)$$
   (注：这里利用了 VGGT 特征的强相关性，即使在长基线下，同一物理点的 $t_i^A$ 和 $t_j^B$ 相似度也应极高，C 为 VGGT 空间的特征维度)
2. **构建学生相关性矩阵**：使用 CoMatcher 输出特征计算相关性矩阵 $S_{co}(i,j)$：
   $$S_{co}(i,j)=\text{softmax} \left( \frac{\hat{f}_i^{A} \cdot \hat{f}_j^{B}}{\sqrt{C}}\right)$$
3. **损失函数（KL Divergence + 置信度门控）**：

==**🔥 12.25 优化：教师置信度门控 (Teacher Confidence Gating)**==

**问题**：原设计无条件信任 $S_{vggt}$，但 VGGT 对某些区域（天空、弱纹理）可能不确定，强制模仿会引入噪声。

**改进**：只有当 VGGT 对点 A 和点 B **都非常有信心**时，才让学生模仿它们之间的关系。

$$\mathcal{L}_{rel} = \sum_{i,j} \underbrace{W_{src}(i) \cdot W_{src}(j)}_{\text{双向置信度门控}} \cdot KL(S_{vggt}(i,j) \| S_{co}(i,j))$$

其中 $W_{src}$ 是之前定义的源感知权重（Quantile + Sigmoid），复用同一套置信度逻辑。

**对比**：
| 场景 | 原设计 | 新设计 (门控) |
|------|--------|--------------|
| 两点都确定 | ✅ 强力监督 | ✅ 强力监督 |
| 点 A 不确定 | ❌ 错误信号 | ⚠️ 弱化监督 |
| 点 B 不确定 | ❌ 错误信号 | ⚠️ 弱化监督 |
| 两点都不确定 | ❌ 纯噪声 | ✅ 几乎忽略 |

**实现代码**：
```python
def relation_loss_with_gating(S_teacher, S_student, W_src):
    # W_src: [B, N] 每个点的置信度权重
    # 构建点对置信度: [B, N, N]
    pair_weight = W_src.unsqueeze(-1) * W_src.unsqueeze(-2)
    
    # 带权重的 KL 散度
    kl = S_teacher * (torch.log(S_teacher + 1e-8) - torch.log(S_student + 1e-8))
    loss = (kl * pair_weight).sum() / (pair_weight.sum() + 1e-6)
    return loss
```

---
### 3. 匹配损失（CoMatcher 自带损失）
上述的跨模态特征对齐与关系知识蒸馏监督只能教会 CoMatcher “如何提取好的特征”，而不能直接教会它 “如何进行最终的匹配”，以及 “如何判断自身的确定性”，因此我们还需要引入 CoMatcher 原文中的两个监督信号：对应关系监督和置信度监督。
我们在训练时“在线”计算匹配真值，使用 VGGT 计算匹配真值与第一阶段中相同，COLMAP 则直接复用 CoMatcher。
##### 3.1 改进的对应关系监督(Correspondence supervision)
**目标**：让匹配头输出的**分配矩阵** $\mathbf P$ 在真值处概率最高。用于教会网络“哪个点匹配哪个点”。
$\mathcal{L}_{corr}$ 是基于 **Negative Log-Likelihood (NLL)** 的损失函数（该损失函数可以平衡正例和负例）直接作用于最终的分配矩阵（Assignment Matrix）。对每个“源–目标”对，基于来自 **VGGT 生成的伪真值**（针对 Hard 样本）和 **COLMAP**（针对 Easy 样本）的真值的总和 $\mathcal{M}_{gt}$ ，把**分配矩阵的负对数似然**最小化：
$$\begin{aligned} \mathcal{L}_{corr}^{improved} = & -\frac{1}{|\mathcal{C}_{match}|} \sum_{(u,v) \in \mathcal{C}_{match}} \textcolor{red}{W_{src}(u)} \cdot \log P(u,v) \\ & -\frac{1}{2|\mathcal{C}_{unmatch}^{src}|} \sum_{u \in \mathcal{C}_{unmatch}^{src}} \textcolor{red}{W_{src}(u)} \cdot \log(1 - \sigma_u^{src}) \\ & -\frac{1}{2|\mathcal{C}_{unmatch}^{tgt}|} \sum_{v \in \mathcal{C}_{unmatch}^{tgt}} \textcolor{red}{W_{src}(v)} \cdot \log(1 - \sigma_v^{tgt}) \end{aligned}$$
- $\sigma_u \approx 1$：网络认为点 $u$ 在对面一定有朋友。
- $\sigma_u \approx 0$：网络认为点 $u$ 在对面是孤立的（被遮挡或移出视野）。
- VGGT 生成数据时，不仅能算出哪些点匹配（$\mathbb{I}_{vis}=1$），还能利用深度图精准算出哪些点**被遮挡了**（$\mathbb{I}_{vis}=0$, 2）。
- 这里的核心改进项是 **$W_{src}(u)$**（源感知权重）。

==***3.1.1 权重项 $W_{src}(u)$ 的详细定义==***
  该权重根据当前样本的来源动态调整，通过 $W_{src}$ 抑制噪声：
$$W_{src}(u) = \begin{cases} \mathbf{1.0} & \text{if Source is COLMAP} \\[8pt] \mathbb{I}_{valid}(u) \cdot \sigma\left(\frac{\text{clamp}(\Sigma(u), \tau_{min}, \Sigma_{95\%}) - \mu}{\sigma_{robust}}\right) & \text{if Source is VGGT} \end{cases}$$

- **对于 COLMAP 样本**：
	- 视为绝对真值（Gold Standard）。权重设为 **1.0**，进行强监督
	- 这保证了模型在简单场景下的高精度
- **对于 VGGT 样本**：
	- **鲁棒统计量**：使用 **95% 分位数** 作为上界，避免 outlier 干扰
	  $$\Sigma_{95\%} = \text{quantile}(\Sigma_{valid}, 0.95)$$
	  $$\Sigma_{clamped} = \text{clamp}(\Sigma, \tau_{min}, \Sigma_{95\%})$$
	- **Sigmoid 映射**：相比线性映射，Sigmoid 对异常值更鲁棒，且能保留中间区域的梯度
	  $$W_{src}(u) = \sigma\left(\frac{\Sigma_{clamped}(u) - \mu}{\sigma_{robust}}\right)$$
	  其中 $\mu, \sigma_{robust}$ 是有效区域 ($\Sigma > \tau_{min}$) 的均值和标准差
	- **Sigmoid 特性**：
		- 中间区域 ($\mu \pm \sigma$)：梯度最大，区分度高
		- 两端区域：平滑饱和，不受极端值影响
		- 输出范围自然落在 $[0, 1]$

==**🔥 12.25 优化：为什么用 Quantile + Sigmoid 替代线性归一化？**==

| 方案 | 公式 | 问题 |
|------|------|------|
| **原始 (线性)** | $\frac{\Sigma - \tau_{min}}{\Sigma_{max} - \tau_{min}}$ | ❌ 若 $\Sigma_{max}$ 是 outlier，大部分权重被压缩到 0 附近 |
| **改进 (Quantile+Sigmoid)** | $\sigma\left(\frac{\text{clamp}(\Sigma) - \mu}{\sigma}\right)$ | ✅ 抗 outlier，S 曲线平滑 |

**实现代码**：
```python
def compute_w_src(uncertainty, tau_min=0.5):
    valid_mask = uncertainty > tau_min
    valid_sigma = uncertainty[valid_mask]
    
    # 鲁棒统计量
    sigma_95 = torch.quantile(valid_sigma, 0.95)
    sigma_clamped = torch.clamp(uncertainty, tau_min, sigma_95)
    
    # Sigmoid 映射
    mu = valid_sigma.mean()
    std = valid_sigma.std() + 1e-6
    W_src = torch.sigmoid((sigma_clamped - mu) / std)
    
    return W_src * valid_mask.float()
```

##### 3.2 改进的置信度监督(Confidence supervision)
**目标**：训练每一层的**点置信头**，让它能在中期就识别“易错/模糊”的点，为后续的**多视图特征相关校正**服务。用于训练网络判断“自己算的匹配准不准”。
把“当前层基于 dual-softmax 的临时匹配”与**最终层的匹配**做一致性判断，一致为 1 ，否则为 0 。同时我们还加入了外部不稳定信号 ($W_{src}(u)$) ，它来自 VGGT ，若 VGGT 认为该区域是不可信的（如天空，高光），则强制置信度归零。
$$\mathcal{L}_{conf} = \frac{1}{L-1} \sum_{l=1}^{L-1} \frac{1}{|\mathcal{P}|} \sum_{u \in \mathcal{P}} \left[ \underbrace{W_{src}(u) \cdot \text{BCE}(c_u^{(l)}, y_u^{(l)})}_{\text{Part A: 稳定性学习}} + \underbrace{(1 - W_{src}(u)) \cdot \text{BCE}(c_u^{(l)}, 0)}_{\text{Part B: 不确定性注入}} \right]$$
- $L$：GNN 的总层数（例如 9 层）。
- $\mathcal{P}$：所有采样点集合。
- $\text{BCE}(p, t) = - (t \cdot \log p + (1-t) \cdot \log(1-p))$：二元交叉熵损失。
- 核心改进项 **$W_{src}(u)$** 与改进的对应关系损失中的权重项保持一致。

***3.2.1 稳定性学习 (Stability Learning)***
$$W_{src}(u) \cdot \text{BCE}(c_u^{(l)}, y_u^{(l)})$$
- **触发条件**：仅在数据**可靠**区域生效（$W_{src}$ 高）。
- **标签 $y_u^{(l)}$ 的定义**（继承自 CoMatcher ）：
  $$y_u^{(l)} = \mathbb{I}\left( \| \mathcal{M}^{(l)}(u) - \mathcal{M}^{(final)}(u) \| < \epsilon \right)$$
  即：如果在第 $l$ 层预测的匹配位置 $\mathcal{M}^{(l)}(u)$ 已经非常接近最终输出位置 $\mathcal{M}^{(final)}(u)$，则标签为 1（这一层“猜对了”）。
- **目的**：教会网络在纹理清晰区域**快速收敛**。如果网络能做到在早期层就锁定正确匹配，它应该输出高置信度。

***3.2.2 不确定性注入 (Uncertainty Injection)***
$$(1 - W_{src}(u)) \cdot \text{BCE}(c_u^{(l)}, 0)$$
- **触发条件**：仅在数据**不可靠**区域生效（$W_{src}$ 低，例如天空、无纹理白墙、动态物体）。 
- **标签**：**强制为 0** 。
- **物理含义**：
	- 即使 CoMatcher 在这些区域“自以为”达到了一致（比如每一层都错误地匹配到了同一个噪点），我们也要通过这个 Loss 告诉它：**“错！这里连老师（VGGT）都看不清，你不应该自信！”**
	- 这实际上是对 CoMatcher **过度自信（Over-confidence）** 的一种惩罚。
- **效果**：训练出的模型在遇到模糊区域时，会倾向于输出极低的 $c_u$ 。在 CoMatcher 的推理机制中，低 $c_u$ 会导致 Attention Map 变得平滑且广泛（Global Context Seeking），这正是处理弱纹理区域的最优策略。

---
### 4. 总损失
总损失函数融合了 **CoMatcher 的基础架构**（对应关系监督、置信度监督）与 **VGGT 的增强策略**（跨模态特征对齐、关系知识蒸馏）
##### 1.VCoMatcher 总损失函数 ($\mathcal{L}_{total}$)
为了适配 CoMatcher 的 `1-to-M` 架构（1 张 Target 图 $I_t$ + M 张 Source 图 $I_s$），总损失是在**每一对图像 $(I_t, I_s)$ 上独立计算后求平均**得到的。
$$\mathcal{L}_{total} = \frac{1}{|\mathcal{S}|} \sum_{I_s \in \mathcal{S}} \left( \underbrace{\mathcal{L}_{corr}+ \lambda_{conf} \mathcal L{conf}}_{\text{任务损失 (Task)}} + \underbrace{\lambda_{align} \mathcal{L}_{align} + \lambda_{rel} \mathcal L_{rel}}_{\text{蒸馏损失 (Distill)}} \right)$$
- **$\mathcal{S}$**：源图像集合（例如 $\{I_{s1}, I_{s2}, I_{s3}\}$）
- 任务损失为了让 CoMatcher 学会**匹配**和**自省**
- 蒸馏损失为了让 CoMatcher 继承 VGGT 的几何感知能力
- 快速设置 $\lambda_{conf}=1.0,\lambda_{align}=1.0,\lambda_{rel}=0.5$

==tricks:== 
观察 Loss 曲线：在第一个 Epoch 结束时，观察各项 Loss 的数值大小：
- 如果 $\mathcal{L}_{align}$ 是 $\mathcal{L}_{corr}$ 的 **10 倍以上** $\rightarrow$ 将 $\lambda_{align}$ 降为 **0.1**。
- 如果 $\mathcal{L}_{rel}$ 只有 $\mathcal{L}_{corr}$ 的 **1/100** $\rightarrow$ 将 $\lambda_{rel}$ 升为 **10.0**。
理想的平衡状态（训练中期）：
$$\lambda_{align} \cdot \mathcal{L}_{align} \approx \lambda_{rel} \cdot \mathcal{L}_{rel} \approx 0.5 \times \mathcal{L}_{corr}$$
即：蒸馏带来的梯度贡献应该是主任务的一半左右，起到正则化和辅助引导的作用，而不是喧宾夺主。

---
# 完整训练推理全流程
### 1. 基于 COLMAP+VGGT 的混合寻略策略
我们使用 COLMAP+VGGT 混合训练策略，核心在于"用 COLMAP 稳住下限（高精度），用 VGGT 突破上限（鲁棒性与稠密性）"。
我们将训练样本根据**匹配难度**（由重叠率、视角变化、纹理丰富度决定）分为三类，并分配不同的数据源：

| 样本类型    | 特征描述                         | 数据源分配                         | 目的                                              |
| ------- | ---------------------------- | ----------------------------- | ----------------------------------------------- |
| Easy    | 重叠率高（0.4~0.7），纹理丰富，视角变化小     | COLMAP（SfM：MegaDepth/ScanNet） | **保证精度**：利用 SfM 的亚像素精度，确保模型在常规场景下的高准确性。         |
| Hard    | 重叠率低（0.1~0.4），弱纹理，大视角变化      | VGGT（Pseudo-GT）               | **提升鲁棒性**：填补 COLMAP 在弱纹理/大视角下的空白，防止模型在困难区域“瞎猜”。 |
| Extreme | 极低重叠（0.05~0.1），几乎无视觉重叠，靠几何强连 | VGGT                          | **探索极限**：作为可选的课程学习后期加入，需配合极高的不确定性过滤             |
##### 1.1 数据加载器设计（Mixed DataLoader）
一个能够动态混合两个数据源的 DataLoader 。
###### **采样策略：**

*1.1.1 宏观策略*
为了保证训练的稳定性，我们不在一个组（Quadruplet）内部混合数据源，而是在 Batch 层面混合。
- **操作**：在生成每一个训练样本时，抛硬币决定：
    - 如果 `random() < P_vggt` $\rightarrow$ 进入 VGGT 池采样一个四元组。
    - 否则 $\rightarrow$ 进入 COLMAP 池采样一个四元组。
- **结果**：一个 Batch（例如 Size=8）可能包含 4 个 COLMAP 组和 4 个 VGGT 组。

*1.1.2 微观策略*
无论来自哪个池，CoMatcher 都需要满足 **1 Target + 3 Source** 的输入结构。VCoMatcher 对此进行了适配，特别是利用 VGGT 改进了重叠率的计算方式。
我们根据重叠分数 $O_{ij}$ 将数据分为不同难度的桶（Bins）：
- standard: $0.4 < O_{ij} \le 0.7$ (COLMAP 监督)
- Hard: $0.1 < O_{ij} \le 0.4$ (VGGT)
- Extreme: $0.05 < O_{ij} \le 0.1$ (VGGT)
**采样逻辑**：
1. 首先随机选择一张 **Target Image ($I_t$)**。
2. 在重叠矩阵中，寻找与 $I_t$ 重叠率和相对旋转满足特定范围（如 Hard）的候选图像。
3. 从中筛选出 3 张 **Source Images ($I_{s1}, I_{s2}, I_{s3}$)**，并且这 3 张源图像之间也要有一定的共视关系（保证 GNN 消息传递有效）。

### 2. 完整训练流程

#### **Phase 1：数据工厂构建与预处理（离线）**
在训练开始前，我们需要构建两个性质截然不同的数据池。
1. **精密池 (Precision Pool - COLMAP)**
    - **数据来源**：MegaDepth / ScanNet 等标准数据集。
    - **真值生成**：使用 COLMAP (SfM) 重建的稀疏 3D 点云和相机位姿。
    - **样本特征**：重叠率较高 ($>0.4$)，纹理丰富，视角变化适中。
    - **作用**：**“压舱石”**。保证模型在常规场景下能达到亚像素级的匹配精度，防止模型学偏。
2. **鲁棒池 (Robustness Pool - VGGT)**
    - **数据来源**：利用预训练 VGGT 对困难场景进行推理生成的稠密伪真值（Pseudo-GT）。
    - **真值生成**：VGGT 预测的稠密深度图 $D_{i}$ 、点图 $P_{i}$ 和相机位姿 $g_{i}$（伪真值）。
    - **样本特征**：重叠率低 ($0.05 \sim 0.4$)，包含弱纹理、大视角、动态物体等 COLMAP 无法重建的区域。
    - **附加信息**：必须存储 VGGT 输出的 **不确定性图 $\Sigma_{i}$** 和 用于计算 Loss 的**几何有效性掩膜 $\mathbb{I}_{valid}$** 和用于构建 GNN 初始图的 $M_{geom}(u)$ 。
    - **作用**：**“磨刀石”**。强迫模型在极端条件下学习几何结构而非简单的纹理匹配。
    - **注意**：不保存稠密特征图 $T_{vggt}$（数据量太大，留到在线训练时计算）。

#### Phase 2：在线训练循环(在线)
训练过程采用 **课程学习 (Curriculum Learning)** 策略，随着 Epoch 的增加动态调整两个池子的采样比例。
**参数定义**：
- $P_{vggt}$：当前 Batch 从 VGGT 池采样的概率。
- $\lambda$：损失函数权重 ($\lambda_{conf}=1.0, \lambda_{align}=1.0, \lambda_{rel}=0.5$)。

###### **Step 1：混合采样与调度 (Curriculum Sampling)**

==**🔥 12.25 优化：基于验证性能的动态调度 (Performance-Based Scheduling)**==

**原设计问题**：固定 Epoch 调度无法适应不同模型的收敛速度，可能过早/过晚引入困难样本。

**改进方案**：基于 **Easy 样本验证性能** 动态决定何时增加难度。

**核心参数**：
- $P_{vggt}$：当前从 VGGT 池采样的概率（初始 = 0）
- $\tau_{cap} = 0.80$：能力阈值（Capability Threshold）
- $\tau_{safe} = 0.90$：相对下降回滚阈值（Safety Net）
- $N_{patience} = 3$：连续达标次数要求

**调度逻辑**：
```
每 validate_interval 个 Epoch:
    1. 在 MegaDepth 验证集 (Easy 部分) 上评估 Matching Precision
    2. 更新指标 EMA: metric_ema = 0.9 * metric_ema + 0.1 * current
    3. 检查回滚条件: if current < best * τ_safe → 回滚
    4. 检查达标条件: if metric_ema > τ_cap (连续 N 次) → 增加 P_vggt
```

**状态转移**：
| 条件 | 动作 |
|------|------|
| metric_ema > 0.80 连续 3 次 | $P_{vggt} += 0.1$ (最大 0.7) |
| current < best × 0.9 | $P_{vggt} -= 0.1$，学习率 ×0.5 |
| 其他 | 维持当前 $P_{vggt}$ |

**为什么比固定 Epoch 更好**：
1. **自适应**：不同模型收敛速度不同，动态调度自动适应
2. **EMA 平滑**：减少验证指标的随机波动干扰
3. **连续达标**：确保真正收敛而非运气
4. **相对回滚**：比绝对阈值更鲁棒，适应不同基线性能

**实现代码**：
```python
class PerformanceScheduler:
    def __init__(self, cap_threshold=0.80, patience=3, ema_decay=0.9):
        self.P_vggt = 0.0
        self.best = 0.0
        self.ema = 0.0
        self.count = 0
        self.cap = cap_threshold
        self.patience = patience
        self.decay = ema_decay
    
    def step(self, metric):
        self.ema = self.decay * self.ema + (1 - self.decay) * metric
        self.best = max(self.best, metric)
        
        # 回滚检查 (相对下降 > 10%)
        if metric < self.best * 0.9:
            self.P_vggt = max(0, self.P_vggt - 0.1)
            self.count = 0
            return 'rollback'
        
        # 达标检查
        if self.ema > self.cap:
            self.count += 1
            if self.count >= self.patience:
                self.P_vggt = min(0.7, self.P_vggt + 0.1)
                self.count = 0
                return 'increase'
        else:
            self.count = 0
        
        return 'maintain'
```

###### Step 2：坐标归一化 (Target-Centric Canonicalization) 🔥关键步骤
为了解决 VGGT 的“相对坐标系陷阱”，必须在送入网络前对数据进行**强行对齐**。
1. **数据层变换**：
    - 计算 Target View 的世界位姿逆矩阵 $T_{target}^{-1}$。
    - 将所有视图（Target + Sources）的 Pose 和 Point Map 全部转换到 **Target 相机坐标系**下，隐式计算并生成 $T_{s \to t}$ (源到目标的变换) $\equiv$  **$T_{new}^{(s)}$** (源图像的新位姿)
    - **验证**：变换后，Target 的 Pose 必须为单位矩阵（Identity）。 
2. **模型层排序**：
    - 构建输入列表时，**强制将 Target View 放在 Index 0**。
    - 输入序列结构：`[Target, Source1, Source2, Source3]`。

**Batch 构成**：如果 Batch Size=8，可能 4 个来自 COLMAP，4 个来自 VGGT。
每个样本包含：
- 归一化后的图像张量 $I_{t},I_{s}$
- 几何真值：
	- **相对位姿** $T_{rel}$ (即 $T_{new}^{(s)}$)：用于几何投影
	- **稠密深度图** $D_{s}, D_{t}$：用于几何投影和一致性检查
	- **相机内参** $K_{s}, K_{t}$
- 不确定性图 $U_{map}$ (COLMAP 样本全为 0，VGGT 样本为 $\Sigma_{i}$)，用于计算 $W_{src}$
- 双重掩膜 (Dual Masks)：
	- $\mathbb{I}_{valid}$ ：**严格**掩膜。剔除天空、遮挡、不确定区域，仅用于计算 Loss
	- $M_{geom}$ ：**宽松**掩膜。仅剔除无效深度，用于 Step 4 构建 GNN 初始图
- 元数据 (Metadata)：
	- `sample_type`: 标记样本类型（`'standard'`, `'hard'`, `'extreme'`），用于在 Step 4 触发“强制几何引导”逻辑 (`'hard'`,`'extreme'`)

###### Step 3：教师网络推理（Teacher Forward - Frozen）
- **模型**：冻结的 VGGT
- **输入**：归一化且排序后的图像序列
- **操作**：
	- VGGT 会自动以序列第一张图（即 Target）为原点在线生成稀疏特征，这与 Step 2 处理后的几何标签**完美对齐**。
- **输出**：**稠密跟踪特征图** $T_{vggt}$：用于特征蒸馏。

==**🔥 12.25 优化：AMP + no_grad 高效在线蒸馏策略**==

**问题背景**：在线运行冻结的 VGGT 会占用大量显存（~18GB 激活值）并拖慢训练速度。

**优化方案**：结合 **混合精度训练 (AMP)** 和 **无梯度推理 (no_grad)** 实现高效蒸馏。

**核心原理**：
1. **`torch.no_grad()`**：Teacher 不需要反向传播，禁用梯度计算可节省 ~40% 显存
2. **`torch.cuda.amp.autocast(fp16/bf16)`**：将 Teacher 前向的浮点精度从 FP32 降为 FP16，显存减半、速度翻倍
3. **组合效果**：显存节省 ~60%，速度提升 ~50-100%

**实现代码**：
```python
from torch.cuda.amp import autocast, GradScaler

class VCoMatcherTrainer:
    def __init__(self):
        self.teacher = load_vggt().eval()
        for p in self.teacher.parameters():
            p.requires_grad = False  # 冻结参数
        
        self.student = VCoMatcher()
        self.scaler = GradScaler()  # 混合精度缩放器
    
    def train_step(self, batch):
        images = batch['images'].cuda()
        
        # ======== Teacher 前向 (无梯度 + FP16) ========
        with torch.no_grad():
            with autocast(dtype=torch.float16):
                teacher_out = self.teacher(images)
                T_vggt = teacher_out['track_features'].detach()
        
        # ======== Student 前向 + 反向 (混合精度) ========
        self.optimizer.zero_grad()
        with autocast(dtype=torch.float16):
            student_out = self.student(batch)
            
            # 在 Student 关键点位置采样 Teacher 特征
            teacher_kp_feat = grid_sample(T_vggt, student_out['keypoints'])
            
            # 蒸馏损失 (在 FP32 下计算 softmax 避免溢出)
            with autocast(enabled=False):
                L_align = info_nce_loss(student_out['features'].float(), 
                                        teacher_kp_feat.float())
                L_rel = kl_div_loss(student_out['features'].float(),
                                    teacher_kp_feat.float())
            
            # 任务损失
            L_corr = self.correspondence_loss(student_out, batch)
            L_conf = self.confidence_loss(student_out, batch)
            
            loss = L_corr + L_conf + λ_align * L_align + λ_rel * L_rel
        
        # 混合精度反向传播
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
```

**显存对比 (A100 80GB, Batch=4×4)**：
| 组件 | FP32 (原始) | AMP + no_grad |
|------|------------|---------------|
| VGGT 激活 | 18 GB | **4.5 GB** |
| Student 激活+梯度 | 9 GB | 5.5 GB |
| **总计** | ~35 GB | **~15 GB** ✅ |

**注意事项**：
1. **数值稳定性**：softmax/log 等操作需在 FP32 下执行（使用 `autocast(enabled=False)` 包裹）
2. **BF16 更安全**：A100 支持 BF16，范围更大不易溢出，推荐使用 `dtype=torch.bfloat16`
3. **梯度截断**：确保 `T_vggt.detach()` 阻断梯度流向 Teacher

###### Step 4：学生网络推理（Student Forward - Trainable）
- **模型**：待训练的 CoMatcher
- **输入**：同一组图像序列
- **操作**：
	- **特征提取 (Backbone)**：提取多尺度特征图 $F$ 和关键点坐标 $U$
	- **动态图构建 (Dynamic Graph Construction)**：
		- 这是 GNN 的前置步骤，用于生成邻接矩阵 $\mathcal A$
		- 如果是COLMAP 样本使用视觉特征 (LightGlue/MNN) 构建 $\mathcal A_{vis}$
		- 如果是VGGT 样本启用几何引导连接 (Geometry-Guided Connecting)：
			1. **输入:** 读取 Step 2 归一化后的**位姿** $T_{s \to t} = [\mathbf{R} | \mathbf{t}] \in \mathbb{R}^{4 \times 4}$（源坐标系到目标坐标系的相对位姿变换矩阵）、源视图的稠密深度图**深度图** $D_s$ 、源视图和目标视图的**相机内参矩阵** $K_s, K_t \in \mathbb{R}^{3 \times 3}$、源视图的`geometry_mask` $M_{geom}^s$ 以及 Backbone 提取的关键点 $U_S$ 和 $U_t$
			2. **采样:** 使用双线性插值（Bilinear Interpolation）获取源图像每个关键点位置的深度值
			   $$d_i = \text{Interp}(D_s, \mathbf{U}_i^s), \quad \forall i \in \{1, \dots, N\}$$
			3. **检查:** 检查这些点是否在几何掩膜内有效：
			   $$v_i = \text{Interp}(M_{geom}^s, \mathbf{U}_i^s) > 0.5$$
				 其中 $v_i \in \{0, 1\}$ 是第 $i$ 个点的有效性标志。
			4. **投影:** 将源关键点反投影到 3D 并重投影到目标视图
				1. 将源图像的 2D 像素坐标提升为源相机坐标系下的 3D 点 $\mathbf{P}_i^s$。设 $\mathbf{u}_i^s = [u, v]^T$，其齐次坐标为 $\tilde{\mathbf{u}}_i^s = [u, v, 1]^T$。
				   $$\mathbf{P}_i^s = d_i \cdot K_s^{-1} \cdot \tilde{\mathbf{u}}_i^s$$
				2. 将 3D 点从源坐标系变换到目标坐标系，并投影回目标图像平面，得到**预测投影点 (Projected Keypoints)** $\hat{\mathbf{u}}_i^{s \to t}$。
				   首先进行刚体变换：
				   $$\mathbf{P}_i^t = \mathbf{R} \cdot \mathbf{P}_i^s + \mathbf{t}$$
						其中 $\mathbf{P}_i^t = [X_t, Y_t, Z_t]^T$。
				   随后进行透视投影：
				   $$\tilde{\mathbf{u}}_{proj, i} = K_t \cdot \mathbf{P}_i^t = [x', y', z']^T$$
                   $$\hat{\mathbf{u}}_i^{s \to t} = \pi(\tilde{\mathbf{u}}_{proj, i}) = \left[ \frac{x'}{z'}, \frac{y'}{z'} \right]^T$$
			5. **建边:** ==**🔥 12.25 优化：半径约束 + 膨胀掩膜**==
				计算源投影点与目标实际关键点的欧氏距离，使用**改进的鲁棒建边策略**：
				1. **距离矩阵 (Distance Matrix) $\mathcal{D}$:** 计算每一个投影点 $\hat{\mathbf{u}}_i^{s \to t}$ 与目标视图中所有关键点 $\mathbf{u}_j^t$ 的欧氏距离：
				   $$\mathcal{D}_{ij} = \| \hat{\mathbf{u}}_i^{s \to t} - \mathbf{u}_j^t \|_2, \quad \mathcal{D} \in \mathbb{R}^{N \times M}$$
				2. **半径约束 (Radius Constraint):** 只有距离小于阈值 $R$ 的点对才能建边（避免远距离错误连接）：
				   $$R = 15 \text{ px}  \quad \text{(可调参数)}$$
				3. **膨胀掩膜 (Dilated Guidance) - 软权重分配:**
				   不再使用硬 k-NN，而是连接半径 $R$ 内的**所有点**，权重由高斯函数决定：
				   $$\mathcal{A}_{geom}(i,j) = \begin{cases} \exp\left(-\frac{\mathcal{D}_{ij}^2}{2\sigma^2}\right) & \text{if } \mathcal{D}_{ij} < R \\ 0 & \text{otherwise} \end{cases}$$
				   其中 $\sigma = R/3$（3σ 原则）
				4. **归一化:** 每行归一化使权重和为 1：
				   $$\mathcal{A}_{geom}(i,:) = \frac{\mathcal{A}_{geom}(i,:)}{\sum_j \mathcal{A}_{geom}(i,j) + \epsilon}$$
				5. **有效性过滤:** 
				   $$\mathcal{A}_{geom}(i,j) = \mathcal{A}_{geom}(i,j) \cdot v_i \cdot \mathbb{I}(Z_{t,i} > 0)$$
					- **$v_i=1$：** 源点在 `geometry_mask` 内有效
					- **$Z_{t,i} > 0$：** 深度检查 (Cheirality Check)
				
				**为什么膨胀掩膜更鲁棒？**
				- **原始 k-NN (硬连接):** 只连接 1 个最近邻，如果投影有误差则 GNN 收到错误信息
				- **膨胀掩膜 (软连接):** 连接半径内所有点，GNN 通过 Cross-Attention 自动选择正确的
				- **$\mathcal{L}_{rel}$ 协同作用:** 关系蒸馏损失进一步软化硬连接，弥补几何先验误差
				
				**实现代码:**
				```python
				def build_geometry_graph(proj_kp, tgt_kp, radius=15.0):
				    dist = torch.cdist(proj_kp, tgt_kp)  # [N, M]
				    sigma = radius / 3.0
				    A = torch.exp(-dist**2 / (2 * sigma**2))
				    A = A * (dist < radius).float()  # 半径约束
				    A = A / (A.sum(dim=1, keepdim=True) + 1e-6)  # 归一化
				    return A
				```
	- **动态上下文聚合 (GNN)**：以 $\mathcal{A}$ (即 $\mathcal{A}_{vis}$ 或 $\mathcal{A}_{geom}$) 作为消息传递的路径 (Edges)计算相对位置 $\Delta p$ ，在源图像和目标图像特征之间进行 Cross-Attention 交互。
	- **Head**：输出关键点坐标 $K$、最终描述符 $F_{co}$、匹配矩阵 $P$、置信度 $C$
- **输出**：关键点坐标 $K$、最终描述符 $F_{co}$、匹配矩阵 $P$（用于 $\mathcal{L}_{corr}$）、置信度 $C$（用于 $\mathcal{L}_{conf}$）

###### Step 5：蒸馏对齐准备（Alignment Prep）
1. **特征投影 (Adapter)**：将学生特征 $F_{co}$ 输入特征适配器（MLP），得到投影特征 $\hat{F}_{co}$，使其维度与 VGGT 一致 。
2. **特征采样 (Sampling)**：根据学生预测的关键点 $K$，在老师的特征图 $T_{vggt}$ 上进行双线性插值，得到目标特征 $t_k$ 。

#### Phase 3：损失计算与优化
总损失函数由**任务损失**和**蒸馏损失**组成，并由**源感知权重** $W_{src}$ 进行动态调节。
###### Step 1：计算源感知权重 $W_{src}$ (Frozen)
（详见 Section 3.1.1 的改进版本：Quantile + Sigmoid）
$$W_{src}(u) = \begin{cases} \mathbf{1.0} & \text{if Source is COLMAP} \\[8pt] \mathbb{I}_{valid}(u) \cdot \sigma\left(\frac{\text{clamp}(\Sigma(u), \tau_{min}, \Sigma_{95\%}) - \mu}{\sigma_{robust}}\right) & \text{if Source is VGGT} \end{cases}$$

###### Step 2：计算基础任务损失 (Task Loss)
- **对应关系损失 ($\mathcal{L}_{corr}$)**：
    - 使用归一化后的几何真值（Pose/Point）计算真值匹配标签。
    - 应用 $W_{src}$ 加权，过滤掉 VGGT 不确定的伪标签（包括匹配项和无匹配项）。
- **置信度损失 ($\mathcal{L}_{conf}$)**：
    - **可靠区域 ($W_{src} \approx 1$)**：监督网络预测“内部一致性”。
    - **不可靠区域 ($W_{src} \approx 0$)**：强制网络输出 **0 置信度**（学会“谦虚”）。

###### Step 3：计算蒸馏损失 (Distillation Loss)
- **特征对齐损失 ($\mathcal{L}_{align}$)**：
    - 使用 **InfoNCE** 损失，拉近投影后的学生特征 $\hat{f}_k$ 与老师特征 $t_k$ 的距离。
- **关系蒸馏损失 ($\mathcal{L}_{rel}$)**：
    - 构建学生和老师的特征**相关性矩阵**。
    - 使用 L2 损失强迫两者的矩阵结构一致（学习几何拓扑）。

###### Step 4：反向传播 (Optimization)
$$\mathcal{L}_{total} = \mathcal{L}_{corr} + \lambda_{conf}\mathcal{L}_{conf} + \lambda_{align}\mathcal{L}_{align} + \lambda_{rel}\mathcal{L}_{rel}$$
- 执行 `Backprop`(`loss_total.backward()`)，更新 CoMatcher 主干及适配器参数。

### 3. 完整推理流程 (复用 CoMatcher)
训练完成后，进入推理阶段：推理阶段 **完全移除** 了教师网络（VGGT）、特征适配器和所有损失计算模块。推理仅依赖训练好的 **CoMatcher（学生网络）**，其参数已内化了 VGGT 的几何感知能力。

#### Phase 0：全局检索与分组 (Grouping) —— 🏛️ 系统前置
这是在送入神经网络之前的物理步骤。
- **输入**：一个无序的图像集合（Image Collection）。
- **操作**：
    1. **检索 (Retrieval)**：使用 NetVLAD 等方法计算全局描述符，检索出共视图像对。
    2. **重叠图构建**：构建场景的共视图（View Graph）。
    3. **分组算法**：将大图切分为一个个小的 **“共视组（Co-visible Groups）”**。每个组包含 $\{I_t, I_{s1}, I_{s2}, I_{s3}\}$。
- **目的**：为 VCoMatcher 准备符合格式的输入数据。

#### Phase 1：组内连接 (Connecting) —— 🔗 几何引导
注意：CoMatcher 中有一个显式的 Connecting 步骤，用于生成初始的粗糙轨迹 $\mathcal{M}(\mathcal{G})$ 作为网络输入。得益于第二阶段的特征蒸馏，CoMatcher 的 Backbone 提取的特征在 Hard 样本上已具备较强的几何一致性。因此，即使是基于互近邻 (MNN) 或 LightGlue 的简单连接，其召回率也远高于未蒸馏版本，足以支撑 GNN 的推理。
- **操作**：在组内使用轻量级匹配器（如 LightGlue 或甚至 SIFT）快速建立初步连接。
- **VCoMatcher 的考量**：
    - 虽然 VCoMatcher 通过 VGGT 蒸馏获得了更强的几何感知力，但为了保持架构兼容性，**这个步骤通常保留**。
    - 这些粗糙的连接提供了初始的几何先验（Geometric Priors），帮助 VCoMatcher 的 GNN 更快聚焦。

#### Phase 2: VCoMatcher 推理 (Matching) —— 🧠 核心计算
**输入**：Step 0 分好的组 + Step 1 生成的粗糙连接。

###### ***step 1：图像输入与特征提取 (Input & Feature Extraction)****
**特征提取**：
- 图像经过共享的 CNN/ViT Backbone（如 ResNet+FPN 或类似结构）提取多尺度特征。
- 提取出初始特征图 $F_A, F_B$ 以及对应的关键点位置 $P_A, P_B$（如果是稀疏匹配）或密集网格（如果是半稠密匹配）。
- 公式：
    $$f_u^{(0)} = \text{Backbone}(I_A)_u$$
    其中 $f_u^{(0)}$ 是位置 $u$ 处的初始特征向量。

###### ***Step 2：动态上下文聚合 (Attentional GNN Aggregation)***
这是 VCoMatcher 的核心处理单元，通过 $L$ 层图神经网络（GNN）进行特征更新。每一层包含 **Self-Attention（自注意力）** 和 **Cross-Attention（交叉注意力）**。
- **位置编码**：将关键点坐标 $p_u$ 编码为高维向量 $\text{PE}(p_u)$ 并叠加到特征上。
- **消息传递（以 Cross-Attention 为例）**：
    - 计算 Query ($q$), Key ($k$), Value ($v$) 向量。
    - 计算注意力分数 $a_{uv}$：
        $$a_{uv} = \text{Softmax}\left(\frac{q_u \cdot k_v}{\sqrt{d}}\right)$$
    - **VCoMatcher 的增强点（隐式）**：由于经过了第二阶段的 **特征对齐 ($\mathcal{L}_{align}$)** 训练，这里的 $q$ 和 $k$ 向量在高维空间中已经具备了 **视点不变性**。即使 $I_A$ 和 $I_B$ 视角差异巨大，对应点的特征内积 $q_u \cdot k_v$ 依然会很高，从而准确捕获长程依赖。、

###### ***Step 3：置信度预测与注意力修正 (Confidence-Aware Rectification)***
这是 CoMatcher 架构的独特机制，也是 VCoMatcher 利用 **不确定性监督 ($\mathcal{L}_{conf}$)** 强化后的关键步骤。
在 GNN 的每一层（或特定层），网络会预测每个点的置信度，并据此调整注意力行为。
- 置信度预测公式：
    $$c_u = \text{Sigmoid}(\text{MLP}(f_u^{(l)})) \in [0, 1]$$
    - **VCoMatcher 的增强**：在训练时，如果 VGGT 认为某区域（如天空、高光）不可靠，我们强制 $c_u \to 0$。因此，在推理时，遇到类似的模糊区域，模型会输出极低的置信度。
- 注意力修正逻辑（基于 CoMatcher 原理）：
    如果置信度 $c_u$ 低于阈值，网络会调整其注意力分布 $\alpha_u$，不再局限于单一的匹配假设，而是可能融合更多邻域信息或抑制该点的响应：
    $$\alpha_u^{new} = \text{Update}(\alpha_u^{orig}, c_u)$$
    (注：具体实现表现为 GNN 权重的动态调整，使得低置信度点不会主导后续计算
	
###### ***Step 4：匹配预测 (Matching Prediction)***
经过 $L$ 层 GNN 更新后，得到最终特征 $f_u^{(L)}$ 和 $f_v^{(L)}$。
- A. 相似度矩阵计算：
    计算源图像点 $u$ 和目标图像点 $v$ 之间的得分矩阵 $S$：
    $$S(u, v) = \langle f_u^{(L)}, f_v^{(L)} \rangle$$
    - **增强点**：得益于 **关系蒸馏 ($\mathcal{L}_{rel}$)**，此矩阵的稀疏性和结构性更接近 VGGT 的几何真值矩阵。
- B. 概率分布 (Dual-Softmax)：
    对 $S$ 的行和列分别进行 Softmax，得到双向匹配概率：
    $$P_{match}(u, v) = \text{Softmax}(S(u, \cdot))_v \cdot \text{Softmax}(S(\cdot, v))_u$$
- C. 可匹配性预测 (Matchability Prediction)：
    网络还独立预测每个点是否“可见/可匹配”（即是否在共视区域内且未被遮挡）：
    $$\sigma_u = \text{Sigmoid}(\text{Linear}(f_u^{(L)}))$$
    $$\sigma_v = \text{Sigmoid}(\text{Linear}(f_v^{(L)}))$$
    
    - **增强点**：训练时使用了 VGGT 生成的精确遮挡掩膜作为负样本。推理时，对于遮挡点，$\sigma$ 会趋近于 0。
- D. 最终分配矩阵 (Final Assignment Matrix)：
    结合匹配概率和可匹配性分数：
    $$P(u, v) = P_{match}(u, v) \cdot \sigma_u \cdot \sigma_v$$
###### ***Step 5：过滤与输出 (Filtering & Output)***
- **互近邻 (MNN)**：选取 $P(u, v) > \text{threshold}$ 且互为最大值的点对。
- **输出**：匹配点对集合 $\{(u_i, v_i)\}$。

#### Phase 3 几何验证与合并 (Verification & Merging) —— 🛠️ 后处理
- **操作**：将所有组的匹配结果汇总，通过 RANSAC 进行几何验证，合并成全局的 Tracks。
- **输出**：用于 COLMAP 重建的稀疏点云和对应关系。