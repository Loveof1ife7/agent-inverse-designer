from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
GRAPHMETAMAT_ROOT = ROOT / "third-party" / "GraphMetaMat"


def _prepare_imports(project_dir: Path) -> None:
    sys.path.insert(0, str(project_dir))


def _relabel_graph(graph):
    import networkx as nx

    mapping = {node: index for index, node in enumerate(sorted(graph.nodes()))}
    if all(node == index for node, index in mapping.items()):
        return graph.copy()
    return nx.relabel_nodes(graph, mapping, copy=True)


def _graph_with_features(graph_path: Path, output_path: Path) -> None:
    _prepare_imports(GRAPHMETAMAT_ROOT)
    from src.dataset_feats_edge import get_edge_feats, get_edge_index, get_edge_li
    from src.dataset_feats_node import get_node_feats

    with graph_path.open("rb") as handle:
        graph = pickle.load(handle)
    graph = _relabel_graph(graph)
    graph.graph["gid"] = 0
    graph.graph.setdefault("rho", 0.1)
    edge_li = get_edge_li(graph)
    edge_index = get_edge_index(edge_li)
    graph.graph["edge_index"] = edge_index
    graph.graph["node_feats"] = get_node_feats(graph, edge_index)
    graph.graph["edge_feats"] = get_edge_feats(graph, edge_li)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(graph, handle)


def _write_dummy_curve(path: Path, resolution: int = 256) -> None:
    strain = np.linspace(0.0, 0.3, resolution)
    stress = np.linspace(1e-6, 1e-3, resolution)
    payload = {
        "curve": np.stack([strain, stress], axis=-1),
        "cid": 0,
        "is_monotonic": True,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def _build_temp_dataset(graph_path: Path, output_dir: Path) -> Path:
    dataset_root = output_dir / "_forward_dataset"
    if dataset_root.exists():
        shutil.rmtree(dataset_root)
    graph_out = dataset_root / "test" / "graphs" / "0.gpkl"
    curve_out = dataset_root / "test" / "curves" / "0.pkl"
    mapping_out = dataset_root / "test" / "mapping.tsv"
    _graph_with_features(graph_path, graph_out)
    _write_dummy_curve(curve_out)
    mapping_out.parent.mkdir(parents=True, exist_ok=True)
    mapping_out.write_text("0\t0\n", encoding="utf-8")
    return dataset_root


def _curve_payload(curve_obj: Any) -> dict[str, Any]:
    c = np.asarray(curve_obj.c, dtype=float).reshape(-1)
    shape = np.asarray(curve_obj.c_shape, dtype=float).reshape(-1)
    shape_std = (
        np.asarray(curve_obj.c_shape_std, dtype=float).reshape(-1)
        if getattr(curve_obj, "c_shape_std", None) is not None
        else None
    )
    magnitude_std = (
        np.asarray(curve_obj.c_magnitude_std, dtype=float).reshape(-1)
        if getattr(curve_obj, "c_magnitude_std", None) is not None
        else None
    )
    return {
        "strain_grid": np.linspace(0.0, 0.3, len(c)).tolist(),
        "stress": c.tolist(),
        "c_shape": shape.tolist(),
        "c_magnitude": float(curve_obj.c_magnitude),
        "c_shape_std": shape_std.tolist() if shape_std is not None else None,
        "c_magnitude_std": magnitude_std.tolist() if magnitude_std is not None else None,
    }


def predict_forward(graph_path: str | Path, output_dir: str | Path, device: str = "cuda") -> dict[str, Any]:
    project_dir = GRAPHMETAMAT_ROOT
    _prepare_imports(project_dir)

    import torch
    from torch.utils.data import DataLoader

    from src.config import args
    from src.dataset import DataLoaderFactory, GraphCurveDataset
    from src.generative_curve.model import Model, ModelEnsemble
    from src.generative_curve.test import test

    graph_path = Path(graph_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = _build_temp_dataset(graph_path, output_dir)

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")

    dataset_args = dict(args["dataset"])
    dlf = DataLoaderFactory(**dataset_args)
    dataset_train = dlf.get_train_dataset()
    dataset = GraphCurveDataset(
        graph_pn=str(dataset_root / "test" / "graphs"),
        curve_pn=str(dataset_root / "test" / "curves"),
        mapping_pn=str(dataset_root / "test"),
        split="test",
        curve_norm_cfg=dataset_args.get("curve_norm_cfg") or {},
        node_feat_cfg=dataset_args["node_feat_cfg"],
        edge_feat_cfg=dataset_args["edge_feat_cfg"],
        digitize_cfg=dataset_args.get("digitize_cfg"),
        is_zscore_graph=dataset_args["is_zscore_graph"],
        is_zscore_curve_magni=dataset_args["is_zscore_curve_magni"],
        is_zscore_curve_shape=dataset_args["is_zscore_curve_shape"],
        use_cosine_node=dataset_args.get("use_cosine_node", 0),
        use_cosine_edge=dataset_args.get("use_cosine_edge", 0),
        g_stats=dataset_train.dataset.g_stats,
        c_stats=dataset_train.dataset.c_stats,
        augment_graph=False,
        augment_curve=False,
    )
    loader = DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=dataset.collate_fn, shuffle=False)

    model = Model.init_from_cfg(dataset=dataset_train, **args["forward_model"])
    if args["forward"]["train_config"]["use_snapshot"] is not None:
        model = ModelEnsemble.from_path(str(Path(args["load_model"]).parent), model)
    else:
        model.load_state_dict(torch.load(args["load_model"], map_location=device))
    model = model.to(torch.device(device))
    model.eval()

    _metrics_collated, _metrics_raw, graph_li, curve_pred_li, _curve_true_li = test(
        loader,
        model,
        device=device,
        plot_num_samples_max=1,
    )
    if not curve_pred_li:
        raise RuntimeError("GraphMetaMat forward model returned no prediction.")
    graph = graph_li[0]
    prediction = _curve_payload(curve_pred_li[0])
    payload = {
        "status": "success",
        "structure_family": "truss",
        "representation": "graph_truss",
        "input_graph_path": str(graph_path),
        "predicted_property": {
            "task": "compression_stress_strain",
            **prediction,
            "rho": float(graph.graph.get("rho", 0.0)),
            "num_nodes": int(graph.number_of_nodes()),
            "num_edges": int(graph.number_of_edges()),
        },
        "artifacts": {
            "output_dir": str(output_dir),
        },
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict GraphMetaMat forward response for one truss gpkl.")
    parser.add_argument("--graph", required=True, help="Input truss graph .gpkl.")
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    response = predict_forward(args.graph, args.out_dir, device=args.device)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(response, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
