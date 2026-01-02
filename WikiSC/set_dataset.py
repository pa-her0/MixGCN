import os
from torch_geometric.datasets import Planetoid, Coauthor
from ogb.nodeproppred import PygNodePropPredDataset


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def download_planetoid(root):
    for name in ["Cora", "CiteSeer", "PubMed"]:
        print(f"Downloading Planetoid dataset: {name}")
        Planetoid(root=root, name=name)


def download_coauthor(root):
    print("Downloading Coauthor CS")
    Coauthor(root=root, name="CS")

    print("Downloading Coauthor Physics")
    Coauthor(root=root, name="Physics")


def download_ogbn_arxiv(root):
    print("Downloading ogbn-arxiv")
    PygNodePropPredDataset(name="ogbn-arxiv", root=root)


def main():
    base_dir = "./runs/data"
    ensure_dir(base_dir)

    # 各数据集子目录
    planetoid_dir = os.path.join(base_dir, "Planetoid")
    coauthor_dir = os.path.join(base_dir, "Coauthor")
    ogb_dir = os.path.join(base_dir, "ogbn_arxiv")

    ensure_dir(planetoid_dir)
    ensure_dir(coauthor_dir)
    ensure_dir(ogb_dir)

    download_planetoid(planetoid_dir)
    download_coauthor(coauthor_dir)
    download_ogbn_arxiv(ogb_dir)

    print("\n✅ All datasets downloaded successfully.")
    print("Root directory:", os.path.abspath(base_dir))


if __name__ == "__main__":
    main()
