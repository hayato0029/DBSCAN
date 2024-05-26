from collections import deque

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_array


def dbscan(
    X: ArrayLike, eps: float = 0.5, min_samples: int = 5, metric: str = "euclidean"
) -> NDArray[np.int_]:
    """
    Density-Based Spatial Clustering of Applications with Noise (DBSCAN)を実行する。

    Args:
        X (ArrayLike): 入力データ。
        eps (float, optional): 同じ近傍と見なされる2つのサンプル間の最大距離。デフォルトは0.5。
        min_samples (int, optional): コアポイントと見なされる近傍内の最小サンプル数。デフォルトは5。
        metric (str, optional): ポイント間のペアワイズ距離の計算に使用される距離メトリック。デフォルトは"euclidean"。

    Returns:
        NDArray[np.int_]: 各サンプルに割り当てられたクラスタラベルの配列。
    """
    X: NDArray[np.float_] = check_array(X)
    # NearestNeighborsでeps内の隣接ポイントを取得
    nn = NearestNeighbors(radius=eps, metric=metric)
    nn.fit(X)
    neighbors = nn.radius_neighbors(X, radius=eps, return_distance=False)

    # 近傍数がmin_samples以上のコアポイントとする
    n_neighbors = np.array([len(nei) for nei in neighbors])
    core_points = np.where(n_neighbors >= min_samples)[0]
    core_point_set = set(core_points)

    # ラベルの初期化
    labels = np.full(X.shape[0], -1, dtype=int)  # 全ての点をノイズとして初期化
    cluster_id = 0

    # コアポイントからクラスタを作成
    for core_point in core_points:
        if labels[core_point] != -1:
            # 既にクラスタに所属している場合はスキップ
            continue

        # 新しいクラスタを作成
        labels[core_point] = cluster_id

        # BFSでクラスタを拡張
        queue = deque([core_point])
        while queue:
            current_point = queue.popleft()
            # 隣接点を探索
            for neighbor in neighbors[current_point]:
                if labels[neighbor] == -1:  # 未訪問の隣接点を現在のクラスタに追加
                    labels[neighbor] = cluster_id
                    if neighbor in core_point_set:  # 隣接点がコアポイントなら探索リストに追加
                        queue.append(neighbor)

        # クラスタIDを更新
        cluster_id += 1

    return labels
