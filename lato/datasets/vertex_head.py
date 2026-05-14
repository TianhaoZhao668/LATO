import os
from typing import Dict, List, Tuple

import numpy as np
import open3d as o3d
import torch
import trimesh
from torch.utils.data import Dataset
from trimesh import grouping


MESH_EXTENSIONS = (".obj", ".ply", ".glb")
CUBE_DILATE_OFFSETS = np.array(
    [
        [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 0, -1], [0, -1, 0], [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1],
        [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 0, -1], [1, -1, 0], [1, 1, 1], [1, -1, 1], [1, 1, -1], [1, -1, -1],
        [-1, 0, 0], [-1, 0, 1], [-1, 1, 0], [-1, 0, -1], [-1, -1, 0], [-1, 1, 1], [-1, -1, 1], [-1, 1, -1], [-1, -1, -1],
    ],
    dtype=np.float32,
)


def normalize_mesh(mesh_path: str) -> trimesh.Trimesh:
    loaded = trimesh.load(mesh_path, process=False, force="scene")
    meshes = []
    for node_name in loaded.graph.nodes_geometry:
        geom_name = loaded.graph[node_name][1]
        geometry = loaded.geometry[geom_name]
        transform = loaded.graph[node_name][0]
        if isinstance(geometry, trimesh.Trimesh):
            mesh = geometry.copy()
            mesh.apply_transform(transform)
            meshes.append(mesh)

    if not meshes and isinstance(loaded, trimesh.Trimesh):
        meshes = [loaded]
    if not meshes:
        raise ValueError(f"No mesh geometry found in {mesh_path}")

    mesh = trimesh.util.concatenate(meshes)
    center = mesh.bounding_box.centroid
    mesh.apply_translation(-center)
    scale = max(mesh.bounding_box.extents)
    if scale <= 0:
        raise ValueError(f"Invalid mesh scale for {mesh_path}")
    mesh.apply_scale(1.0 / scale)
    return mesh


def _make_o3d_mesh(vertices: np.ndarray, faces: np.ndarray) -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    return mesh


def _voxelize_mesh_and_points(
    mesh: o3d.geometry.TriangleMesh,
    points: np.ndarray,
    volume_resolution: int,
) -> torch.Tensor:
    offsets = CUBE_DILATE_OFFSETS / (volume_resolution * 4 - 1)
    voxelization_mesh = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        mesh,
        voxel_size=1.0 / volume_resolution,
        min_bound=[-0.5, -0.5, -0.5],
        max_bound=[0.5, 0.5, 0.5],
    )
    voxel_mesh = np.asarray([voxel.grid_index for voxel in voxelization_mesh.get_voxels()])

    dilated_points = np.clip(
        (points[np.newaxis] + offsets[..., np.newaxis, :]).reshape(-1, 3),
        -0.5 + 1e-6,
        0.5 - 1e-6,
    )
    point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(dilated_points))
    voxelization_points = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
        point_cloud,
        voxel_size=1.0 / volume_resolution,
        min_bound=[-0.5, -0.5, -0.5],
        max_bound=[0.5, 0.5, 0.5],
    )
    voxel_points = np.asarray([voxel.grid_index for voxel in voxelization_points.get_voxels()])

    if voxel_mesh.size == 0:
        voxels = voxel_points
    elif voxel_points.size == 0:
        voxels = voxel_mesh
    else:
        voxels = np.unique(np.concatenate([voxel_mesh, voxel_points], axis=0), axis=0)
    return torch.from_numpy(voxels.astype(np.float32))


def _sorted_vertex_directions(points: np.ndarray, face_vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    view_dtype = np.dtype((np.void, face_vertices.dtype.itemsize * face_vertices.shape[-1]))
    vertex_view = face_vertices.view(view_dtype).squeeze(-1)
    sort_idx = np.argsort(vertex_view, axis=1)
    batch_indices = np.arange(face_vertices.shape[0])[:, None]
    sorted_vertices = face_vertices[batch_indices, sort_idx]
    return (
        sorted_vertices[:, 0, :] - points,
        sorted_vertices[:, 1, :] - points,
        sorted_vertices[:, 2, :] - points,
    )


def sample_edges_dora(mesh: trimesh.Trimesh, n_samples: int):
    adjacent_faces = mesh.face_adjacency
    adjacent_edges = mesh.face_adjacency_edges
    internal_data = None

    if len(adjacent_faces) > 0:
        normals = mesh.face_normals[adjacent_faces[:, 0]] + mesh.face_normals[adjacent_faces[:, 1]]
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms < 1e-6] = 1.0
        normals = normals / norms

        starts = mesh.vertices[adjacent_edges[:, 0]]
        ends = mesh.vertices[adjacent_edges[:, 1]]
        face_pairs = mesh.faces[adjacent_faces]
        unique_0 = np.sum(face_pairs[:, 0], axis=1) - np.sum(adjacent_edges, axis=1)
        unique_1 = np.sum(face_pairs[:, 1], axis=1) - np.sum(adjacent_edges, axis=1)
        virtual = (mesh.vertices[unique_0] + mesh.vertices[unique_1]) * 0.5
        internal_data = (starts, ends, normals, virtual)

    boundary_data = None
    edge_rows = mesh.edges_sorted
    if len(edge_rows) > 0:
        boundary_group = grouping.group_rows(edge_rows, require_count=1)
        if len(boundary_group) > 0:
            boundary_indices = np.concatenate([np.atleast_1d(g) for g in boundary_group])
            face_indices = boundary_indices // 3
            starts = mesh.vertices[edge_rows[boundary_indices, 0]]
            ends = mesh.vertices[edge_rows[boundary_indices, 1]]
            normals = mesh.face_normals[face_indices]
            face_vertices = mesh.faces[face_indices]
            unique = np.sum(face_vertices, axis=1) - np.sum(edge_rows[boundary_indices], axis=1)
            virtual = mesh.vertices[unique]
            boundary_data = (starts, ends, normals, virtual)

    parts = [part for part in (internal_data, boundary_data) if part is not None]
    if not parts:
        return None, None, None

    starts = np.concatenate([part[0] for part in parts], axis=0)
    ends = np.concatenate([part[1] for part in parts], axis=0)
    normals = np.concatenate([part[2] for part in parts], axis=0)
    virtual = np.concatenate([part[3] for part in parts], axis=0)

    lengths = np.linalg.norm(ends - starts, axis=1)
    if lengths.sum() < 1e-9:
        probabilities = np.ones(len(lengths), dtype=np.float64) / len(lengths)
    else:
        probabilities = lengths / lengths.sum()

    chosen = np.random.choice(len(lengths), size=n_samples, p=probabilities)
    t = np.random.rand(n_samples, 1)
    points = starts[chosen] + (ends[chosen] - starts[chosen]) * t
    triplets = np.stack([starts[chosen], ends[chosen], virtual[chosen]], axis=1).astype(np.float32)
    return points.astype(np.float32), normals[chosen].astype(np.float32), triplets


def load_quantized_mesh_original(
    mesh_path: str,
    mesh_load: trimesh.Trimesh | None = None,
    volume_resolution: int = 256,
    use_normals: bool = True,
    pc_sample_number: int = 4096000,
):
    if mesh_load is None:
        mesh_o3d = o3d.io.read_triangle_mesh(mesh_path)
        vertices = np.clip(np.asarray(mesh_o3d.vertices), -0.5 + 1e-6, 0.5 - 1e-6)
        faces = np.asarray(mesh_o3d.triangles)
        mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
    else:
        vertices = np.clip(np.asarray(mesh_load.vertices), -0.5 + 1e-6, 0.5 - 1e-6)
        faces = np.asarray(mesh_load.faces)
        mesh_o3d = _make_o3d_mesh(vertices, faces)

    trimesh_obj = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    points, face_indices = trimesh_obj.sample(count=pc_sample_number, return_index=True)
    points = points.astype(np.float32)
    voxels = _voxelize_mesh_and_points(mesh_o3d, points, volume_resolution)

    feature_parts = [torch.from_numpy(points)]
    if use_normals:
        mesh_o3d.compute_triangle_normals()
        normals = np.asarray(mesh_o3d.triangle_normals)[face_indices].astype(np.float32)
        feature_parts.append(torch.from_numpy(normals))

    face_vertex_indices = faces[face_indices]
    face_vertices = np.stack(
        [vertices[face_vertex_indices[:, 0]], vertices[face_vertex_indices[:, 1]], vertices[face_vertex_indices[:, 2]]],
        axis=1,
    )
    for direction in _sorted_vertex_directions(points, face_vertices):
        feature_parts.append(torch.from_numpy(direction.astype(np.float32)))

    return voxels, torch.cat(feature_parts, dim=-1)


def load_quantized_mesh_dora(
    mesh_path: str,
    mesh_load: trimesh.Trimesh | None = None,
    volume_resolution: int = 256,
    use_normals: bool = True,
    pc_sample_number: int = 4096000,
    edge_sample_ratio: float = 0.2,
):
    if mesh_load is None:
        mesh_o3d = o3d.io.read_triangle_mesh(mesh_path)
        vertices = np.clip(np.asarray(mesh_o3d.vertices), -0.5 + 1e-6, 0.5 - 1e-6)
        faces = np.asarray(mesh_o3d.triangles)
        mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
    else:
        vertices = np.clip(np.asarray(mesh_load.vertices), -0.5 + 1e-6, 0.5 - 1e-6)
        faces = np.asarray(mesh_load.faces)
        mesh_o3d = _make_o3d_mesh(vertices, faces)

    trimesh_obj = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    n_edge = int(pc_sample_number * edge_sample_ratio)
    edge_points, edge_normals, edge_triplets = sample_edges_dora(trimesh_obj, n_edge)
    n_surface = pc_sample_number if edge_points is None else pc_sample_number - n_edge

    surface_points, surface_face_indices = trimesh_obj.sample(n_surface, return_index=True)
    surface_points = surface_points.astype(np.float32)
    surface_normals = trimesh_obj.face_normals[surface_face_indices].astype(np.float32)
    surface_triplets = vertices[faces[surface_face_indices]].astype(np.float32)

    if edge_points is None:
        points = surface_points
        normals = surface_normals
        triplets = surface_triplets
    else:
        points = np.concatenate([surface_points, edge_points], axis=0).astype(np.float32)
        normals = np.concatenate([surface_normals, edge_normals], axis=0).astype(np.float32)
        triplets = np.concatenate([surface_triplets, edge_triplets], axis=0).astype(np.float32)

    voxels = _voxelize_mesh_and_points(mesh_o3d, points, volume_resolution)
    feature_parts = [torch.from_numpy(points)]
    if use_normals:
        feature_parts.append(torch.from_numpy(normals))
    for direction in _sorted_vertex_directions(points, triplets):
        feature_parts.append(torch.from_numpy(direction.astype(np.float32)))

    return voxels, torch.cat(feature_parts, dim=-1)


class VoxelVertexDataset_edge(Dataset):
    def __init__(
        self,
        root_dir: str,
        base_resolution: int = 256,
        min_resolution: int = 128,
        renders_dir: str = "",
        active_voxel_res: int = 64,
        pc_sample_number: int = 409600,
        filter_active_voxels: bool = False,
        min_active_voxels: int = 2000,
        max_active_voxels: int = 40000,
        cache_filter_path: str = "",
        sample_type: str = "uniform",
        augment_data: bool = False,
        aug_rotate: bool = True,
        aug_scale_range: Tuple[float, float] = (0.75, 0.95),
        aug_translate: float = 0.05,
        **_: object,
    ):
        del renders_dir, min_active_voxels, max_active_voxels
        self.root_dir = root_dir
        self.active_voxel_res = active_voxel_res
        self.pc_sample_number = pc_sample_number
        self.sample_type = sample_type
        self.augment_data = augment_data
        self.aug_rotate = aug_rotate
        self.aug_scale_range = aug_scale_range
        self.aug_translate = aug_translate

        if base_resolution & (base_resolution - 1):
            raise ValueError("base_resolution must be a power of two")
        if min_resolution & (min_resolution - 1):
            raise ValueError("min_resolution must be a power of two")

        self.res_levels = [active_voxel_res if active_voxel_res is not None else base_resolution]

        obj_files = sorted(f for f in os.listdir(root_dir) if f.lower().endswith(MESH_EXTENSIONS))
        if filter_active_voxels and cache_filter_path and os.path.exists(cache_filter_path):
            with open(cache_filter_path, "r", encoding="utf-8") as f:
                allowed = {line.strip() for line in f if line.strip()}
            obj_files = [f for f in obj_files if os.path.splitext(f)[0] in allowed]

        if not obj_files:
            raise ValueError(f"No mesh files found in {root_dir}")
        self.obj_files = obj_files

    def __len__(self) -> int:
        return len(self.obj_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        name = os.path.splitext(self.obj_files[idx])[0]
        mesh_path = os.path.join(self.root_dir, self.obj_files[idx])

        try:
            mesh = normalize_mesh(mesh_path)
            if mesh.is_empty or len(mesh.vertices) < 3 or len(mesh.faces) < 1:
                raise ValueError("invalid or empty mesh")
            if self.augment_data:
                self._augment_mesh(mesh)
        except Exception as exc:
            print(f"[ERROR] Failed to load mesh {self.obj_files[idx]}: {exc}")
            return self.__getitem__((idx + 1) % len(self))

        vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
        faces = torch.tensor(mesh.faces, dtype=torch.long)
        data: Dict[str, torch.Tensor] = {
            "original_faces": faces.clone(),
            "original_vertices": vertices.clone(),
            "image_path": "",
            "name": name,
        }

        for resolution in self.res_levels:
            data.update(self._build_resolution_data(mesh, mesh_path, vertices, faces, resolution))
        return data

    def _augment_mesh(self, mesh: trimesh.Trimesh) -> None:
        mesh.apply_scale(np.random.uniform(*self.aug_scale_range))
        if self.aug_rotate:
            mesh.apply_transform(trimesh.transformations.random_rotation_matrix())
        if self.aug_translate > 1e-6:
            mesh.apply_translation(np.random.uniform(-self.aug_translate, self.aug_translate, size=3))

    def _build_resolution_data(
        self,
        mesh: trimesh.Trimesh,
        mesh_path: str,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        resolution: int,
    ) -> Dict[str, torch.Tensor]:
        del vertices, faces
        loader = load_quantized_mesh_original if self.sample_type == "uniform" else load_quantized_mesh_dora
        active_voxels, point_cloud = loader(
            mesh_path=mesh_path,
            mesh_load=mesh,
            volume_resolution=resolution,
            use_normals=True,
            pc_sample_number=self.pc_sample_number,
        )
        return {
            f"active_voxels_{resolution}": active_voxels.int(),
            f"point_cloud_{resolution}": point_cloud.float(),
        }


def collate_fn_pointnet(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    batch = [sample for sample in batch if sample is not None]
    if not batch:
        return {}

    collated: Dict[str, object] = {
        "original_faces": [sample["original_faces"] for sample in batch],
        "original_vertices": [sample["original_vertices"] for sample in batch],
        "image_path": [sample.get("image_path", "") for sample in batch],
        "name": [sample.get("name", "unknown") for sample in batch],
    }

    res_levels = sorted(
        int(key.split("_")[-1])
        for key in batch[0]
        if key.startswith("active_voxels_")
    )

    for resolution in res_levels:
        collated.update(_collate_resolution(batch, resolution))
    return collated  # type: ignore[return-value]


def _cat_or_empty(items: List[torch.Tensor], shape: Tuple[int, ...], dtype: torch.dtype, device: torch.device):
    return torch.cat(items, dim=0) if items else torch.empty(shape, dtype=dtype, device=device)


def _collate_resolution(batch: List[Dict[str, torch.Tensor]], resolution: int) -> Dict[str, torch.Tensor]:
    device = torch.device("cpu")
    for value in batch[0].values():
        if isinstance(value, torch.Tensor):
            device = value.device
            break

    active_voxels_list = []
    point_clouds_list = []

    for batch_idx, sample in enumerate(batch):
        active_voxels = sample.get(f"active_voxels_{resolution}", torch.empty(0, 3, dtype=torch.int32)).to(device)
        if active_voxels.numel() > 0:
            batch_col = torch.full((active_voxels.shape[0], 1), batch_idx, dtype=torch.int32, device=device)
            active_voxels_list.append(torch.cat([batch_col, active_voxels], dim=1))

        point_cloud = sample.get(f"point_cloud_{resolution}", torch.empty(0, 15, dtype=torch.float32)).to(device)
        if point_cloud.numel() > 0:
            point_clouds_list.append(point_cloud)

    result = {
        f"active_voxels_{resolution}": _cat_or_empty(active_voxels_list, (0, 4), torch.int32, device),
    }
    if point_clouds_list:
        result[f"point_cloud_{resolution}"] = torch.stack(point_clouds_list, dim=0)
    else:
        result[f"point_cloud_{resolution}"] = torch.empty((0, 15), dtype=torch.float32, device=device)
    return result
