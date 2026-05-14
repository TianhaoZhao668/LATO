import argparse
import os
import random
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import torch
import trimesh
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lato.datasets.vertex_head import VoxelVertexDataset_edge, collate_fn_pointnet
from lato.models.lato_vae.lato_vae import VoxelVAE
from lato.modules.sparse.basic import SparseTensor
from utils import load_pretrained_woself
from vertex_encoder import ConnectionHead, VoxelFeatureEncoder_active_pointnet


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(checkpoint_path: str, config_path: str | None = None) -> dict:
    if config_path is not None:
        if not os.path.exists(config_path):
            fallback = PROJECT_ROOT / "configs" / "infer_vae_512.yaml"
            print(f"Config not found: {config_path}")
            print(f"Falling back to: {fallback}")
            config_path = str(fallback)
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    checkpoint_dir = os.path.dirname(checkpoint_path)
    candidates = [
        os.path.join(checkpoint_dir, "config.yaml"),
        os.path.join(os.path.dirname(checkpoint_dir), "config.yaml"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            with open(candidate, "r") as f:
                print(f"Loaded config from: {candidate}")
                return yaml.safe_load(f)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "config" in checkpoint:
        print("Loaded config from checkpoint")
        return checkpoint["config"]

    raise FileNotFoundError(
        "Could not find a config file. Pass --config or save config.yaml next to the checkpoint."
    )


def make_model(config: dict, checkpoint_path: str, device: torch.device):
    voxel_encoder = VoxelFeatureEncoder_active_pointnet(
        in_channels=15,
        hidden_dim=256,
        out_channels=1024,
        scatter_type="mean",
        n_blocks=5,
        resolution=128,
    ).to(device)

    connection_head = ConnectionHead(
        channels=512 * 2,
        out_channels=1,
        mlp_ratio=0.75,
    ).to(device)

    vae = VoxelVAE(
        in_channels=config["model"]["in_channels"],
        latent_dim=config["model"]["latent_dim"],
        encoder_blocks=config["model"]["encoder_blocks"],
        decoder_blocks_vtx=config["model"]["decoder_blocks_vtx"],
        num_heads=8,
        num_head_channels=64,
        mlp_ratio=4.0,
        attn_mode="swin",
        window_size=8,
        pe_mode="ape",
        use_fp16=False,
        use_checkpoint=False,
        qk_rms_norm=False,
        using_subdivide=True,
        using_attn=config["model"].get("using_attn", False),
        attn_first=config["model"].get("attn_first", True),
        pred_direction=config["model"].get("pred_direction", False),
    ).to(device)

    load_pretrained_woself(
        checkpoint_path=checkpoint_path,
        voxel_encoder=voxel_encoder,
        connection_head=connection_head,
        vae=vae,
    )

    voxel_encoder.eval()
    connection_head.eval()
    vae.eval()
    return vae, voxel_encoder, connection_head


def build_single_mesh_batch(mesh_path: str, config: dict, pc_sample_number: int):
    mesh_path = os.path.abspath(mesh_path)
    mesh_dir = os.path.dirname(mesh_path)
    mesh_name = os.path.basename(mesh_path)

    dataset_cfg = config.get("dataset", {})
    dataset = VoxelVertexDataset_edge(
        root_dir=mesh_dir,
        base_resolution=dataset_cfg.get("base_resolution", 512),
        min_resolution=dataset_cfg.get("min_resolution", 128),
        renders_dir=dataset_cfg.get("renders_dir", ""),
        filter_active_voxels=False,
        cache_filter_path=dataset_cfg.get("cache_filter_path", ""),
        sample_type=dataset_cfg.get("sample_type", "uniform"),
        active_voxel_res=128,
        pc_sample_number=pc_sample_number,
    )

    try:
        sample_idx = dataset.obj_files.index(mesh_name)
    except ValueError as exc:
        raise FileNotFoundError(f"Mesh {mesh_name!r} was not found in dataset directory {mesh_dir!r}.") from exc

    sample = dataset[sample_idx]
    if sample is None:
        raise RuntimeError(f"Failed to preprocess mesh: {mesh_path}")
    return collate_fn_pointnet([sample])


def predict_edges(
    connection_head: torch.nn.Module,
    vertex_feats: torch.Tensor,
    vertex_coords: torch.Tensor,
    threshold: float,
    batch_size: int,
    k_neighbors: int | None,
    device: torch.device,
) -> np.ndarray:
    num_vertices = vertex_feats.shape[0]
    if num_vertices < 3:
        return np.empty((0, 2), dtype=np.int64)

    if k_neighbors is not None and 0 < k_neighbors < num_vertices:
        dist_mat = torch.cdist(vertex_coords.float(), vertex_coords.float())
        _, indices = torch.topk(dist_mat, k=k_neighbors + 1, dim=1, largest=False)
        neighbor_indices = indices[:, 1:]
        src = torch.arange(num_vertices, device=device).unsqueeze(1).repeat(1, k_neighbors).flatten()
        dst = neighbor_indices.flatten()
        mask = src < dst
        u_indices = src[mask]
        v_indices = dst[mask]
    else:
        u_indices, v_indices = torch.triu_indices(num_vertices, num_vertices, offset=1, device=device)

    probs = []
    with torch.no_grad():
        for start in range(0, u_indices.shape[0], batch_size):
            end = min(start + batch_size, u_indices.shape[0])
            batch_u = u_indices[start:end]
            batch_v = v_indices[start:end]
            feat_u = vertex_feats[batch_u]
            feat_v = vertex_feats[batch_v]
            logits_uv = connection_head(torch.cat([feat_u, feat_v], dim=-1))
            logits_vu = connection_head(torch.cat([feat_v, feat_u], dim=-1))
            probs.append(torch.sigmoid(logits_uv + logits_vu).squeeze(-1))

    if not probs:
        return np.empty((0, 2), dtype=np.int64)

    edge_mask = torch.cat(probs) > threshold
    edges = torch.stack([u_indices[edge_mask], v_indices[edge_mask]], dim=1)
    return edges.cpu().numpy()


def build_triangles_from_edges(edges: np.ndarray, num_vertices: int) -> np.ndarray:
    if len(edges) == 0:
        return np.empty((0, 3), dtype=np.int64)

    graph = nx.Graph()
    graph.add_nodes_from(range(num_vertices))
    graph.add_edges_from(edges)

    triangles = []
    adjacency = [set(graph.neighbors(node)) for node in range(num_vertices)]
    for u, v in edges:
        if u > v:
            u, v = v, u
        for w in adjacency[u].intersection(adjacency[v]):
            if w > v:
                triangles.append([u, v, w])
    return np.asarray(triangles, dtype=np.int64)


def reconstruct_mesh(
    mesh_path: str,
    checkpoint_path: str,
    output_path: str,
    config_path: str | None = None,
    pc_sample_number: int = 819200,
    vertex_threshold: float = 0.5,
    edge_threshold: float = 0.45,
    edge_batch_size: int = 4096,
    k_neighbors: int | None = None,
    seed: int = 42,
) -> str:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(checkpoint_path, config_path)
    vae, voxel_encoder, connection_head = make_model(config, checkpoint_path, device)
    batch = build_single_mesh_batch(mesh_path, config, pc_sample_number)

    active_coords = batch["active_voxels_128"].to(device)
    point_cloud = batch["point_cloud_128"].to(device)
    use_amp = device.type == "cuda"

    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
        active_voxel_feats = voxel_encoder(
            p=point_cloud,
            sparse_coords=active_coords,
            res=128,
            bbox_size=(-0.5, 0.5),
        )
        sparse_input = SparseTensor(feats=active_voxel_feats, coords=active_coords.int())
        latent, _ = vae.encode(sparse_input)
        decoded = vae.decode(
            latent,
            gt_vertex_voxels_list=[],
            training=False,
            inference_threshold=vertex_threshold,
            vis_last_layer=False,
        )

        vertex_result = decoded[-1]["vertex"]
        vertex_coords = vertex_result["coords"].float()
        vertex_feats = vertex_result["feats"]
        edges = predict_edges(
            connection_head=connection_head,
            vertex_feats=vertex_feats,
            vertex_coords=vertex_coords,
            threshold=edge_threshold,
            batch_size=edge_batch_size,
            k_neighbors=k_neighbors,
            device=device,
        )

    faces = build_triangles_from_edges(edges, vertex_coords.shape[0])
    vertices = vertex_coords.cpu().numpy() / 512.0 - 0.5
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if len(mesh.faces) > 0:
        trimesh.repair.fix_normals(mesh)

    output_path = os.path.abspath(output_path)
    if os.path.isdir(output_path) or not os.path.splitext(output_path)[1]:
        mesh_name = os.path.splitext(os.path.basename(mesh_path))[0]
        output_path = os.path.join(output_path, f"{mesh_name}_recon.obj")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mesh.export(output_path)
    print(f"Saved reconstructed mesh to: {output_path}")
    print(f"Vertices: {len(vertices)}, edges: {len(edges)}, faces: {len(faces)}")
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Reconstruct a mesh with the LATO VAE.")
    parser.add_argument("--mesh", required=True, help="Input mesh path (.obj, .ply, .glb).")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path.")
    parser.add_argument("--output", required=True, help="Output reconstructed mesh path, e.g. out/recon.obj.")
    parser.add_argument("--config", default=None, help="Optional config YAML path.")
    parser.add_argument("--pc-sample-number", type=int, default=819200, help="Point samples for active voxel features.")
    parser.add_argument("--vertex-threshold", type=float, default=0.5, help="VAE vertex occupancy threshold.")
    parser.add_argument("--edge-threshold", type=float, default=0.45, help="Connection head threshold.")
    parser.add_argument("--edge-batch-size", type=int, default=4096, help="Candidate edge inference batch size.")
    parser.add_argument("--k-neighbors", type=int, default=0, help="Use KNN candidate edges; 0 means all pairs.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    reconstruct_mesh(
        mesh_path=args.mesh,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        config_path=args.config,
        pc_sample_number=args.pc_sample_number,
        vertex_threshold=args.vertex_threshold,
        edge_threshold=args.edge_threshold,
        edge_batch_size=args.edge_batch_size,
        k_neighbors=args.k_neighbors if args.k_neighbors > 0 else None,
        seed=args.seed,
    )
