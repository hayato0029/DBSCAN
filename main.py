import click
import matplotlib.pyplot as plt
from loguru import logger
from sklearn.cluster import DBSCAN as SkDBSCAN
from sklearn.datasets import make_moons
from ttimer import get_timer

from src.dbscan import dbscan


@click.command()
@click.option("--eps", type=float, default=0.1)
@click.option("--min_samples", type=int, default=5)
@click.option("--metric", type=str, default="euclidean")
@click.option("--n_samples", type=int, default=500)
@click.option("--noise", type=float, default=0.1)
@click.option("--random_state", type=int, default=42)
def main(
    eps: float, min_samples: int, metric: str, n_samples: int, noise: float, random_state: int
) -> None:
    timer = get_timer("dbscan")
    # データ生成
    X, _ = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)

    with timer("my dbscan"):
        # 自作のDBSCAN
        labels1 = dbscan(X, eps=eps, min_samples=min_samples, metric=metric)
    with timer("sklearn dbscan"):
        # scikit-learnのDBSCAN
        sk_dbscan = SkDBSCAN(eps=eps, min_samples=min_samples, metric=metric)
        labels2 = sk_dbscan.fit_predict(X)

    logger.info(f"\n{timer.render()}")

    # plot
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.title(f"My DBSCAN (exec time: {timer['my dbscan'].time:.3f} sec)")
    plt.scatter(X[:, 0], X[:, 1], c=labels1)
    plt.subplot(122)
    plt.title(f"sklearn DBSCAN (exec time: {timer['sklearn dbscan'].time:.3f} sec)")
    plt.scatter(X[:, 0], X[:, 1], c=labels2)

    # save
    plt.savefig("resources/dbscan.png")


if __name__ == "__main__":
    main()
