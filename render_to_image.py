# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
# import os
# os.environ['EGL_PLATFORM'] = 'surfaceless'   # Ubuntu 20.04+
# os.environ['OPEN3D_CPU_RENDERING'] = 'true'  # Ubuntu 18.04

from typing import Any
import open3d as o3d
import open3d.visualization.rendering as rendering
import cv2 as cv
import numpy as np
import os
import csv
from itertools import chain
from tqdm import tqdm

path_blade = "/home/jiri/blade_generator/rv12/rv12.obj"
path_bkgnd_dir = "/home/jiri/winpart/Edwards/Edwards/Datasety/detector-blade/zaznam_edwards"
path_albedo = "/home/jiri/blade_generator/rv12/albedo3.png"

# load all backgroud files recursivelly in all subdirectories

def init_background():
    path_bkgnd = []
    for root, dirs, file in os.walk(path_bkgnd_dir):
        for f in file:
            if ".jpg" in f:
                path_bkgnd.append(os.path.join(root, f))

    return path_bkgnd[:1000]


def project(K, cMw, u):
    N = u.shape[0]
    newcol = np.ones((N, 1))
    hu = np.hstack([u, newcol])
    hv = K @ cMw[0:3, :] @ np.transpose(hu)
    hv /= hv[-1, ...]
    return hv[:-1]


def random_pos():
    # return 0
    return 2.0 * np.random.rand(1)[0] - 1.0


class BladeGenerator:
    def __init__(self):
        self.path_bkgnd = init_background()
        self.ibkgnd = 0
        self.max_bkgnd = len(self.path_bkgnd)
        self.index = 0  # number of calls to 'cos' function

        self.render = rendering.OffscreenRenderer(1280, 720)

        metal_texture_data = o3d.data.MetalTexture()

        grey = rendering.MaterialRecord()
        # grey.base_color = [0.25, 0.25, 0.25, 1.0]
        grey.shader = "defaultLit"
        # grey.metallic_img  = o3d.t.io.read_image(
        #     metal_texture_data.metallic_texture_path)
        grey.albedo_img = o3d.io.read_image(path_albedo)
        
        self.blade = o3d.io.read_triangle_mesh(path_blade)
        self.blade.translate(-self.blade.get_center())
        self.blade.scale(0.001, center=self.blade.get_center())

        # Create a Material object
        material = o3d.visualization.Material()

        # self.blade.material = rendering.MaterialRecord('defaultLit')
        # self.blade.material.texture_maps['albedo'] = o3d.t.io.read_image(
        #     metal_texture_data.albedo_texture_path)
        # self.blade.material.texture_maps['normal'] = o3d.t.io.read_image(
        #     metal_texture_data.normal_texture_path)
        # self.blade.material.texture_maps['roughness'] = o3d.t.io.read_image(
        #     metal_texture_data.roughness_texture_path)
        # self.blade.material.texture_maps['metallic'] = o3d.t.io.read_image(
        #     metal_texture_data.metallic_texture_path)


        self.render.scene.add_geometry("blade", self.blade, grey)

        # plane
        self.image = rendering.MaterialRecord()
        self.image.base_color = [1.0, 1.0, 1.0, 1.0]
        self.image.shader = "defaultLit"
        self.image.albedo_img = o3d.io.read_image(
            os.path.join(path_bkgnd_dir, self.path_bkgnd[self.ibkgnd])
        )

        plane = o3d.geometry.TriangleMesh()
        verts = np.array(
            [[-1, -1, 0], [1, -1, 0], [-1, 1, 0], [1, 1, 0]],
            dtype=np.float64,
        )
        triangles = np.array([[0, 1, 2], [2, 1, 3]])
        plane.vertices = o3d.utility.Vector3dVector(0.1 * verts)
        plane.triangles = o3d.utility.Vector3iVector(triangles)
        plane.compute_vertex_normals()

        v_uv = np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
        )

        plane.triangle_uvs = o3d.utility.Vector2dVector(v_uv)

        T = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.3], [0, 0, 0, 1]])

        s = 5.0

        S = np.array([[s, 0, 0, 0], [0, s, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        plane.transform(T @ S)

        self.render.scene.add_geometry("image", plane, self.image)

        # camera setup
        self.K = np.array(
            [
                [9.1190216064453125e02, 0.0, 6.3450402832031250e02],
                [0.0, 9.1069586181640625e02, 3.5893045043945312e02],
                [0.0, 0.0, 1.0],
            ]
        )

        cRw = np.eye(4)
        cTw = np.eye(4)
        cTw[2, -1] = 1.25  # + 0.25 * np.cos(index / 10)  # meters
        self.cMw = cTw @ cRw
        self.render.setup_camera(self.K, self.cMw, 1280, 720)

        # constant illumination
        self.render.scene.scene.set_sun_light(cTw[:3, 3], [1.0, 1.0, 1.0], 75000)
        self.render.scene.scene.enable_sun_light(True)

        self.padding = 40

        d = 4.0
        step = (np.pi / d) / 3.0

        rng1 = np.arange(1.0 * np.pi / d, (d - 1) * np.pi / d, step)
        rng2 = np.arange((d + 1) * np.pi / d, (2 * d - 1) * np.pi / d, step)

        self.y_range = list(np.arange(-160 + self.padding, 160, self.padding))
        self.x_range = list(np.arange(-160 + self.padding, 160, self.padding))
        self.az_range = list(np.arange(-np.pi / d, np.pi / d, step))
        self.ay_range = list(chain(rng1, rng2))
        self.ax_range = list(np.arange(0, 2.0 * np.pi, step))

        self.total = (
            len(self.y_range)
            * len(self.x_range)
            * len(self.ax_range)
            * len(self.ay_range)
            * len(self.az_range)
        )

        self.generator = self.gen_params()

    # Deleting (Calling destructor)
    def __del__(self):
        print("Destructor called, Employee deleted.")

    def gen_params(self):
        for y in self.y_range:
            for x in self.x_range:
                for az in self.az_range:
                    for ay in self.ay_range:
                        for ax in self.ax_range:
                            yield y, x, ax, ay, az  

    def __call__(self):
        y, x, ax, ay, az = next(self.generator)

        if y is None:
            raise StopIteration()

        rx = ax + random_pos() * 5.0 / 180.0 * np.pi
        ry = ay + random_pos() * 5.0 / 180.0 * np.pi
        rz = az + random_pos() * 5.0 / 180.0 * np.pi
        wRo = np.eye(4)
        wRo[:3, :3] = o3d.geometry.get_rotation_matrix_from_yzx((ry, rz, rx))

        # object translation wrt world
        wTo = np.eye(4)
        wTo[2, -1] = 0.25 * np.cos(self.index / 10)  # meters

        # apply model matrix to object
        wMo = wTo @ wRo
        self.render.scene.set_geometry_transform("blade", wMo)

        # bbox
        cMo = self.cMw @ wMo
        image_of_tet = project(self.K, cMo, np.array(self.blade.vertices))
        image_of_tet_reshaped = np.reshape(
            np.transpose(image_of_tet), (-1, 1, 2)
        ).astype(np.float32)
        bbox = cv.boundingRect(image_of_tet_reshaped)

        img = self.render.render_to_image()

        frame = cv.cvtColor(np.asarray(img), cv.COLOR_RGB2BGR)
        x1 = int((1280 - 320) / 2 + x + random_pos() * self.padding / 10.0)
        x2 = x1 + 320
        y1 = int((720 - 320) / 2 + y + random_pos() * self.padding / 10.0)
        y2 = y1 + 320
        thumb = frame[y1:y2, x1:x2, :]

        off_box = tuple(np.subtract(bbox, (x1, y1, 0, 0)))

        self.ibkgnd += 1
        if self.ibkgnd == self.max_bkgnd:
            self.ibkgnd = 0

        self.image.albedo_img = o3d.io.read_image(
            os.path.join(path_bkgnd_dir, self.path_bkgnd[self.ibkgnd])
        )

        self.render.scene.modify_geometry_material("image", self.image)

        yield [thumb, off_box, cMo]


if __name__ == "__main__":
    gen = BladeGenerator()

    out_path = "images"
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    cv.namedWindow("img", cv.WINDOW_NORMAL)
    with open(os.path.join(out_path, "meta.csv"), "w") as f:
        writer = csv.writer(f)
        with tqdm(total=gen.total) as pbar:
            for index in range(0, gen.total):
                thumb, off_box, cMo = gen()
                image_path = os.path.join(out_path, "{:07d}.png".format(index))
                written = cv.imwrite(image_path, thumb)
                # written = False

                if False and index % 100 == 0 or not written:
                    cv.rectangle(thumb, off_box, (0, 255, 0), 1)
                    cv.imshow("img", thumb)
                    key = cv.waitKey(1 if written else 0)
                    if key == 27:
                        f.close()

                if written:
                    row = [
                        "TRAIN",
                        image_path,
                        "rv12",
                        "{:0.3f}".format(off_box[0] / 320.0),
                        "{:0.3f}".format(off_box[1] / 320.0),
                        None,
                        None,
                        "{:0.3f}".format((off_box[0] + off_box[2]) / 320),
                        "{:0.3f}".format((off_box[1] + off_box[3]) / 320),
                        "{:0.3f}".format(cMo[0, 0]),
                        "{:0.3f}".format(cMo[1, 0]),
                        "{:0.3f}".format(cMo[2, 0]),
                        "{:0.3f}".format(cMo[0, 1]),
                        "{:0.3f}".format(cMo[1, 1]),
                        "{:0.3f}".format(cMo[2, 1]),
                    ]
                    writer.writerow(row)
                    index += 1
                    pbar.update()
