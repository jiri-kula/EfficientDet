import numpy as np
import matplotlib.pyplot as plt

def show_gizmo(angle, ax, x_min, y_min, w, h):
        r1 = angle[:3]
        r2 = angle[3:]
        r3 = np.cross(r1, r2)

        cRo = np.stack([r1, r2, r3], axis=1)

        s = 15.0

        xo = np.array([1, 0, 0])
        yo = np.array([0, 1, 0])
        zo = np.array([0, 0, 1])

        xc = np.dot(cRo, xo)
        yc = np.dot(cRo, yo)
        zc = np.dot(cRo, zo)

        cx = x_min + w / 2.0
        cy = y_min + h / 2.0

        a = np.array([cx, cy, 0])
        b = a + s * xc
        c = a + s * yc
        d = a + s * zc

        ax.add_line(plt.Line2D([a[0], c[0]], [a[1], c[1]], color="green"))
        ax.add_line(plt.Line2D([a[0], d[0]], [a[1], d[1]], color="blue"))
        ax.add_line(
            plt.Line2D([a[0], b[0]], [a[1], b[1]], color="red")
        )  # last to become visible

