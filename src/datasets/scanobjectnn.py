import os
import random
import shutil
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
import glob
import h5py
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import matplotlib.pyplot as plt
from src.datasets import ModelNet40

from PIL import Image
import open3d as o3d
# Data structures and functions for rendering
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    HardPhongShader,
    Textures,
    FoVOrthographicCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PulsarPointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor
)


# this code is borrowed from https://github.com/ajhamdi/mvtorch/blob/main/mvtorch/data.py
class ScanObjectNNPointCloud(Dataset):
    """
    This class loads ScanObjectNN from a given directory into a Dataset object.
    ScanObjjectNN is a point cloud dataset of realistic shapes of from the ScanNet dataset and can be downloaded from
    https://github.com/hkust-vgd/scanobjectnn .
    """

    def __init__(
            self,
            data_dir,
            split,
            nb_points=100000,
            normals: bool = False,
            suncg: bool = False,
            variant: str = "obj_only",
            dset_norm: str = "inf",

    ):
        """
        Store each object's synset id and models id from data_dir.
        Args:
            data_dir: Path to ShapeNetCore data.
            synsets: List of synset categories to load from ShapeNetCore in the form of
                synset offsets or labels. A combination of both is also accepted.
                When no category is specified, all categories in data_dir are loaded.
            version: (int) version of ShapeNetCore data in data_dir, 1 or 2.
                Default is set to be 1. Version 1 has 57 categories and verions 2 has 55
                categories.
                Note: version 1 has two categories 02858304(boat) and 02992529(cellphone)
                that are hyponyms of categories 04530566(watercraft) and 04401088(telephone)
                respectively. You can combine the categories manually if needed.
                Version 2 doesn't have 02858304(boat) or 02834778(bicycle) compared to
                version 1.
            load_textures: Boolean indicating whether textures should loaded for the model.
                Textures will be of type TexturesAtlas i.e. a texture map per face.
            texture_resolution: Int specifying the resolution of the texture map per face
                created using the textures in the obj file. A
                (texture_resolution, texture_resolution, 3) map is created per face.
        """
        super().__init__()
        self.data_dir = data_dir
        self.nb_points = nb_points
        self.normals = normals
        self.suncg = suncg
        self.variant = variant
        self.dset_norm = dset_norm
        self.split = split
        self.classes = {0: 'bag', 10: 'bed', 1: 'bin', 2: 'box', 3: 'cabinet', 4: 'chair', 5: 'desk', 6: 'display',
                        7: 'door', 11: 'pillow', 8: 'shelf', 12: 'sink', 13: 'sofa', 9: 'table', 14: 'toilet'}

        self.labels_dict = {"train": {}, "test": {}}
        self.objects_paths = {"train": [], "test": []}

        if self.variant != "hardest":
            pcdataset = pd.read_csv(os.path.join(
                data_dir, "split_new.txt"), sep="\t", names=['obj_id', 'label', "split"])
            for ii in range(len(pcdataset)):
                if pcdataset["split"][ii] != "t":
                    self.labels_dict["train"][pcdataset["obj_id"]
                    [ii]] = pcdataset["label"][ii]
                else:
                    self.labels_dict["test"][pcdataset["obj_id"]
                    [ii]] = pcdataset["label"][ii]

            all_obj_ids = glob.glob(os.path.join(self.data_dir, "*/*.bin"))
            filtered_ids = list(filter(lambda x: "part" not in os.path.split(
                x)[-1] and "indices" not in os.path.split(x)[-1], all_obj_ids))

            self.objects_paths["train"] = sorted(
                [x for x in filtered_ids if os.path.split(x)[-1] in self.labels_dict["train"].keys()])
            self.objects_paths["test"] = sorted(
                [x for x in filtered_ids if os.path.split(x)[-1] in self.labels_dict["test"].keys()])
        else:
            filename = os.path.join(data_dir, "{}_objectdataset_augmentedrot_scale75.h5".format(self.split))
            with h5py.File(filename, "r") as f:
                self.labels_dict[self.split] = np.array(f["label"])
                self.objects_paths[self.split] = np.array(f["data"])
            #     print("1############", len(self.labels_dict[self.split]))
            # print("2############", len(self.labels_dict[self.split]))

    def __getitem__(self, idx: int) -> Dict:
        """
        Read a model by the given index. no mesh is availble in this dataset so retrun None and correction factor of 1.0

        """
        if self.variant != "hardest":
            obj_path = self.objects_paths[self.split][idx]
            # obj_path,label
            points = self.load_pc_file(obj_path)
            # sample the required number of points randomly
            if len(points) > self.nb_points:
                # points = points[np.random.randint(points.shape[0], size=self.nb_points), :]
                points = points[np.linspace(0, points.shape[0] - 1, self.nb_points).astype(int), :]
            # print(pc.min(),classes[label],obj_path)
            label = self.labels_dict[self.split][os.path.split(obj_path)[-1]]
        else:

            points = self.objects_paths[self.split][idx]
            label = self.labels_dict[self.split][idx]

        points = points
        points[:, :3] = np_center_and_normalize(points[:, :3], p=self.dset_norm)
        return label, None, points

    def __len__(self):
        return len(self.objects_paths[self.split])

    def load_pc_file(self, filename):
        # load bin file
        # pc=np.fromfile(filename, dtype=np.float32)
        pc = np.fromfile(filename, dtype=np.float32)

        # first entry is the number of points
        # then x, y, z, nx, ny, nz, r, g, b, label, nyu_label
        if (self.suncg):
            pc = pc[1:].reshape((-1, 3))
        else:
            pc = pc[1:].reshape((-1, 11))

        # return pc

        # only use x, y, z for now
        if self.variant == "with_bg":
            # pc = np.array(pc[:, 0:3])
            return pc

        else:
            ##To remove backgorund points
            ##filter unwanted class
            filtered_idx = np.intersect1d(np.intersect1d(np.where(
                pc[:, -1] != 0)[0], np.where(pc[:, -1] != 1)[0]), np.where(pc[:, -1] != 2)[0])
            (values, counts) = np.unique(pc[filtered_idx, -1], return_counts=True)
            max_ind = np.argmax(counts)
            idx = np.where(pc[:, -1] == values[max_ind])[0]
            # pc = np.array(pc[idx, 0:3])
            pc = np.array(pc[idx])
            return pc


def np_center_and_normalize(points, p="inf"):
    """
    a helper pytorch function that normalize and center 3D points clouds
    """
    N = points.shape[0]
    center = points.mean(0)
    if p != "fro" and p != "no":
        scale = np.max(np.linalg.norm(points - center, ord=float(p), axis=1))
    elif p == "fro":
        scale = np.linalg.norm(points - center, ord=p)
    elif p == "no":
        scale = 1.0
    points = points - center[None]
    points = points * (1.0 / float(scale))
    return points


def render_pointcloud_mv_img(points, num_cam, renderer, cameras, lights, device='cuda'):
    #  x, y, z, nx, ny, nz, r, g, b, label, nyu_label
    point_cloud = Pointclouds(points=[torch.tensor(points[:, :3], dtype=torch.float32, device=device)],
                              normals=[torch.tensor(points[:, 3:6], dtype=torch.float32, device=device)],
                              features=[torch.tensor(points[:, 6:9], dtype=torch.float32, device=device) / 256]).extend(num_cam)
    images = renderer(point_cloud)
    return images


def render_mesh_mv_img(points, num_cam, renderer, cameras, lights, device='cuda'):
    #  x, y, z, nx, ny, nz, r, g, b, label, nyu_label
    # Create an open3d.geometry.PointCloud object from the NumPy array
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points[:, :3])
    point_cloud.normals = o3d.utility.Vector3dVector(points[:, 3:6])
    point_cloud.colors = o3d.utility.Vector3dVector(points[:, 6:9])

    # Perform poisson surface reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=8)
    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(point_cloud,
    #                                                                        o3d.utility.DoubleVector([0.01, 0.1]))
    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(point_cloud, alpha=0.2)

    # Smooth the mesh
    # mesh = mesh.filter_smooth_simple()

    # Create a batch of meshes by repeating the cow mesh and associated textures.
    # Meshes has a useful `extend` method which allows us do this very easily.
    # This also extends the textures.
    meshes = Meshes(verts=[torch.tensor(mesh.vertices, dtype=torch.float32, device=device)],
                    faces=[torch.tensor(mesh.triangles, dtype=torch.int64, device=device)],
                    textures=Textures(verts_rgb=torch.tensor(mesh.vertex_colors, dtype=torch.float32, device=device)[
                                                    None] / 255)).extend(num_cam)

    # We can pass arbitrary keyword arguments to the rasterizer/shader via the renderer
    # so the renderer does not need to be reinitialized if any of the settings change.
    images = renderer(meshes, cameras=cameras, lights=lights)
    return images


def save_mv_img_dataset(base, root, num_cam, split, visualize=False):
    point_cnt_avg = 0
    device = 'cuda'

    # # Define the settings for rasterization and shading. Here we set the output image to be of size
    # # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
    # # and blur_radius=0.0. We also set bin_size and max_faces_per_bin to None which ensure that
    # # the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for
    # # explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of
    # # the difference between naive and coarse-to-fine rasterization.
    # raster_settings = RasterizationSettings(image_size=512, blur_radius=0.0, faces_per_pixel=1, )

    # # Get a batch of viewing angles.
    # elev = 30
    # azim = torch.linspace(-180, 180, num_cam + 1)[:num_cam]

    # # All the cameras helper methods support mixed type inputs and broadcasting. So we can
    # # view the camera from the same distance and specify dist=2.7 as a float,
    # # and then specify elevation and azimuth angles for each viewpoint as tensors.
    # R, T = look_at_view_transform(dist=2, elev=elev, azim=azim)
    # cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    # lights = PointLights(device=device, location=[[0.0, 3.0, 0.0]])

    # # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will
    # # interpolate the texture uv coordinates for each vertex, sample from a texture image and
    # # apply the Phong lighting model
    # renderer = MeshRenderer(rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
    #                         shader=HardPhongShader(device=device, cameras=cameras, lights=lights))

    # Initialize a camera.

    # Get a batch of viewing angles.
    elev = 30
    azim = torch.linspace(-180, 180, num_cam + 1)[:num_cam]

    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
    # and blur_radius=0.0. Refer to raster_points.py for explanations of these parameters.
    raster_settings = PointsRasterizationSettings(
        image_size=512,
        radius=0.003,
        points_per_pixel=10
    )

    # All the cameras helper methods support mixed type inputs and broadcasting. So we can
    # view the camera from the same distance and specify dist=2.7 as a float,
    # and then specify elevation and azimuth angles for each viewpoint as tensors.
    R, T = look_at_view_transform(dist=2, elev=elev, azim=azim)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    lights = PointLights(device=device, location=[[0.0, 3.0, 0.0]])

    # Create a points renderer by compositing points using an alpha compositor (nearer points
    # are weighted more heavily). See [1] for an explanation.
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    renderer = PointsRenderer(
        rasterizer=rasterizer,
        compositor=AlphaCompositor()
    )
    for class_name in base.classes.values():
        if os.path.exists(f'{root}/{class_name}/{split}/'):
            shutil.rmtree(f'{root}/{class_name}/{split}/')

    for idx in tqdm(range(len(base))):
        target, mesh, points = base.__getitem__(idx)
        imgs = render_pointcloud_mv_img(points, num_cam, renderer, cameras, lights)
        # imgs = self.mvrenderer(mesh, points[None].cuda(),
        #                        azim=torch.linspace(0, 360, self.num_cam + 1)[None, :self.num_cam].cuda(),
        #                        elev=30 * torch.ones([1, self.num_cam]).cuda(),
        #                        dist=2 * torch.ones([1, self.num_cam]).cuda())[0][0].permute(0, 2, 3, 1)
        # imgs = render_mesh_mv_img(points, num_cam, renderer, cameras, lights)
        for cam, img in enumerate(imgs):
            if visualize:
                plt.imshow(img.cpu().numpy())
                plt.show()
            os.makedirs(f'{root}/{base.classes[target]}/{split}/', exist_ok=True)
            img = Image.fromarray((img.cpu().numpy() * 255).astype('uint8'))
            img.save(f'{root}/{base.classes[target]}/{split}/{idx:04d}_{cam + 1:02d}.png')
        print(f'{root}/{base.classes[target]}/{split}/{idx:04d}')
        point_cnt_avg += len(points)
        pass
    print(point_cnt_avg / len(base))


class ScanObjectNN(ModelNet40):
    classnames = ['bag', 'bin', 'box', 'cabinet', 'chair', 'desk', 'display',
                  'door', 'shelf', 'table', 'bed', 'pillow', 'sink', 'sofa', 'toilet']

    def __init__(self, root, split='train', per_cls_instances=0, dropout=0.0):
        super().__init__(root, 12, split, per_cls_instances, dropout)


if __name__ == '__main__':
    split = 'train'
    dataset = ScanObjectNNPointCloud('/home/houyz/Data/ScanObjectNN', split)
    save_mv_img_dataset(dataset, '/home/houyz/Data/ScanObjectNN_pc', 12, split)
    split = 'test'
    dataset = ScanObjectNNPointCloud('/home/houyz/Data/ScanObjectNN', split)
    save_mv_img_dataset(dataset, '/home/houyz/Data/ScanObjectNN_pc', 12, split)
    # dataset = ScanObjectNN('/home/houyz/Data/ScanObjectNN')
    # dataset.__getitem__(0)
    # dataset.__getitem__(len(dataset) - 1, visualize=True)
    pass
