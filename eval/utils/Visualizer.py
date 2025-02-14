import numpy as np
import scipy as sp
import pandas as pd

import seaborn as sns
import matplotlib as mpl

import matplotlib.pyplot as plt


class Visualizer():
    def __init__(self, *args, **kwargs):
        self.options = {
            'backend': mpl.get_backend(),
            'style': 'default', # plt.rcParams['style'],
        }

    def change_options(self, backend=None, style=None):
        if backend is not None:
            self.options['backend'] = backend
            self.change_backend(backend)
        if style is not None:
            self.options['style'] = style
            self.change_style(style)

    def change_backend(self, backend, force=False):
        mpl.use(backend, force=force)

    def change_style(self, style):
        if style == 'white': style = 'seaborn-v0_8-whitegrid'
        elif style == 'dark': style = 'seaborn-v0_8-darkgrid'
        plt.style.use(style)

    def new_figure(self, title=None, size=None):
        fig = plt.figure(figsize=size)

        if title:
            fig.suptitle(title)

        return fig

    def current_figure(self):
        return plt.gcf()

    def new_plot(self, fig, title=None, axes=None, grid=None, align=None, legend=None):
        if isinstance(axes, list):
            if len(axes) == 2:
                axs = fig.add_subplot()
            elif len(axes) == 3:
                axs = fig.add_subplot(projection='3d')
        else:
            if axes in [2, '2d']:
                axs = fig.add_subplot()
            elif axes in [3, '3d']:
                axs = fig.add_subplot(projection='3d')

        if grid:
            shape = np.sqrt(len(fig.axes))

            major = int(np.round(shape))
            minor = int(np.ceil(shape))

            if align in ['r', 'row', 'h', 'horizontal']:
                rows = major
                cols = minor
            elif align in ['c', 'col', 'v', 'vertical']:
                rows = minor
                cols = major
        else:
            if align in ['r', 'row', 'h', 'horizontal']:
                rows = 1
                cols = len(fig.axes)
            elif align in ['c', 'col', 'v', 'vertical']:
                rows = len(fig.axes)
                cols = 1

        gs = fig.add_gridspec(rows, cols)

        for i, axs in enumerate(fig.axes):
            axs.set_subplotspec(gs[i])

        if title:
            axs.set_title(title)

        if isinstance(axes, list):
            if len(axes) > 0: axs.set_xlabel(axes[0])
            if len(axes) > 1: axs.set_ylabel(axes[1])
            if len(axes) > 2: axs.set_zlabel(axes[2])

        if legend:
            axs.legend()

        return axs

    def current_plot(self):
        return plt.gca()

    def generate(self, fig, path=None):
        plt.legend()
        plt.tight_layout()

        if path is None: plt.show()
        else: plt.savefig(path)

        plt.close(fig)

    def imshow(self, axs, image, **kwargs):
        axs.imshow(image, **kwargs)

    def matshow(self, axs, array, **kwargs):
        axs.matshow(array, **kwargs)

    def pie_1D(self, axs, name, data, orientation=None, **kwargs):
        if orientation is None:
            axs.pie(data, labels=name, **kwargs)
        elif orientation in ['v', 'vertical']:
            axs.pie(data, labels=name, **kwargs)
        elif orientation in ['h', 'horizontal']:
            axs.pie(data, labels=name, **kwargs)

    def hist_1D(self, axs, name, data, orientation=None, **kwargs):
        if orientation is None:
            axs.hist(data, label=name, **kwargs)
        elif orientation in ['v', 'vertical']:
            axs.hist(data, label=name, orientation='vertical', **kwargs)
        elif orientation in ['h', 'horizontal']:
            axs.hist(data, label=name, orientation='horizontal', **kwargs)

    def box_1D(self, axs, name, data, orientation=None, **kwargs):
        if orientation is None:
            axs.boxplot(data, labels=name, notch=True, meanline=True, **kwargs)
        elif orientation in ['v', 'vertical']:
            axs.boxplot(data, labels=name, notch=True, meanline=True, vert=True, **kwargs)
        elif orientation in ['h', 'horizontal']:
            axs.boxplot(data, labels=name, notch=True, meanline=True, vert=False, **kwargs)

    def violin_1D(self, axs, name, data, orientation=None, **kwargs):
        if orientation is None:
            axs.violinplot(data, showextrema=True, showmeans=True, showmedians=True, **kwargs) # labels=name
        elif orientation in ['v', 'vertical']:
            axs.violinplot(data, showextrema=True, showmeans=True, showmedians=True, vert=True, **kwargs) # labels=name
        elif orientation in ['h', 'horizontal']:
            axs.violinplot(data, showextrema=True, showmeans=True, showmedians=True, vert=False, **kwargs) # labels=name

    def event_1D(self, axs, name, data, orientation=None, **kwargs):
        if orientation is None:
            axs.eventplot(data, label=name, **kwargs)
        elif orientation in ['v', 'vertical']:
            axs.eventplot(data, label=name, orientation='vertical', **kwargs)
        elif orientation in ['h', 'horizontal']:
            axs.eventplot(data, label=name, orientation='horizontal', **kwargs)

    def stairs_1D(self, axs, name, data, orientation=None, **kwargs):
        if orientation is None:
            axs.stairs(data, label=name, **kwargs)
        elif orientation in ['v', 'vertical']:
            axs.stairs(data, label=name, orientation='vertical', **kwargs)
        elif orientation in ['h', 'horizontal']:
            axs.stairs(data, label=name, orientation='horizontal', **kwargs)

    def curve_2D(self, axs, name, x_data, y_data, **kwargs):
        if x_data is None: x_data = np.arange(len(y_data))
        name = name * y_data.shape[-1]
        axs.plot(x_data, y_data, label=name, **kwargs)

    def fill_2D(self, axs, name, x_data, y_data, **kwargs):
        if x_data is None: x_data = np.arange(len(y_data))
        axs.fill(x_data, y_data, label=name, **kwargs)

    def bar_2D(self, axs, name, x_data, y_data, orientation=None, **kwargs):
        if x_data is None: x_data = np.arange(len(y_data))
        if orientation is None:
            axs.bar(x_data, y_data, label=name, **kwargs)
        elif orientation in ['v', 'vertical']:
            axs.bar(x_data, y_data, label=name, **kwargs)
        elif orientation in ['h', 'horizontal']:
            axs.barh(x_data, y_data, label=name, **kwargs)

    def scatter_2D(self, axs, name, x_data, y_data, **kwargs):
        if x_data is None: x_data = np.tile(np.arange(len(y_data)), y_data.shape[1:]).reshape(y_data.shape)
        name = name * y_data.shape[-1]
        axs.scatter(x_data, y_data, label=name, **kwargs)

    def stem_2D(self, axs, name, x_data, y_data, **kwargs):
        if x_data is None: x_data = np.arange(len(y_data))
        axs.stem(x_data, y_data, label=name, **kwargs)

    def step_2D(self, axs, name, x_data, y_data, **kwargs):
        if x_data is None: x_data = np.arange(len(y_data))
        axs.step(x_data, y_data, label=name, **kwargs)

    def stack_2D(self, axs, name, x_data, y_data, **kwargs):
        if x_data is None: x_data = np.arange(len(y_data))
        axs.stackplot(x_data, y_data, label=name, **kwargs)

    def hist_2D(self, axs, name, x_data, y_data, **kwargs):
        axs.hist2d(x_data, y_data, label=name, **kwargs)

    def hexa_2D(self, axs, name, x_data, y_data, **kwargs):
        axs.hexbin(x_data, y_data, label=name, **kwargs)

    def triplot_2D(self, axs, name, x_data, y_data, **kwargs):
        axs.triplot(x_data, y_data, label=name, **kwargs)

    def line_2D(self, axs, name, data, minimum, maximum, orientation=None, **kwargs):
        if orientation is None:
            axs.vlines(data, minimum, maximum, label=name, **kwargs)
        elif orientation in ['v', 'vertical']:
            axs.vlines(data, minimum, maximum, label=name, **kwargs)
        elif orientation in ['h', 'horizontal']:
            axs.hlines(data, minimum, maximum, label=name, **kwargs)

    def error_2D(self, axs, name, x_data, y_data, x_err, y_err, style=None, **kwargs):
        if style is None:
            axs.errorbar(x_data, y_data, xerr=x_err, yerr=y_err, label=name, **kwargs)
        elif style in ['fill', 'area']:
            axs.plot(x_data, y_data, label=name, **kwargs)
            if y_err is not None:
                axs.fill_between(x_data, y_data - y_err, y_data + y_err, alpha=0.25)
            if x_err is not None:
                axs.fill_betweenx(y_data, x_data - x_err, x_data + x_err, alpha=0.25)

    def quiver_2D(self, axs, name, x_data, y_data, x_grad, y_grad, **kwargs):
        axs.quiver(x_data, y_data, x_grad, y_grad, label=name, **kwargs)

    def streamplot_2D(self, axs, name, x_data, y_data, x_grad, y_grad, **kwargs):
        axs.streamplot(x_data, y_data, x_grad, y_grad, label=name, **kwargs)

    def heatmap_v1_2D(self, axs, name, x_data, y_data, z_data, **kwargs):
        axs.pcolor(x_data, y_data, z_data, label=name, **kwargs)

    def heatmap_v2_2D(self, axs, name, x_data, y_data, z_data, **kwargs):
        axs.pcolormesh(x_data, y_data, z_data, label=name, **kwargs)

    def contour_2D(self, axs, name, x_data, y_data, z_data, style=None, **kwargs):
        if style is None:
            axs.contour(x_data, y_data, z_data, label=name, **kwargs)
        elif style in ['l', 'line']:
            axs.contour(x_data, y_data, z_data, label=name, **kwargs)
        elif style in ['a', 'area']:
            axs.contourf(x_data, y_data, z_data, label=name, **kwargs)

    def tricontour_2D(self, axs, name, x_data, y_data, z_data, style=None, **kwargs):
        if style is None:
            axs.tricontour(x_data, y_data, z_data, label=name, **kwargs)
        elif style in ['l', 'line']:
            axs.tricontour(x_data, y_data, z_data, label=name, **kwargs)
        elif style in ['a', 'area']:
            axs.tricontourf(x_data, y_data, z_data, label=name, **kwargs)

    def surface_3D(self, axs, name, x_data, y_data, z_data, **kwargs):
        axs.plot_surface(x_data, y_data, z_data, label=name, **kwargs)

    def trisurf_3D(self, axs, name, x_data, y_data, z_data, **kwargs):
        axs.plot_trisurf(x_data, y_data, z_data, label=name, **kwargs)

    def wireframe_3D(self, axs, name, x_data, y_data, z_data, **kwargs):
        axs.plot_wireframe(x_data, y_data, z_data, label=name, **kwargs)

    def curve_3D(self, axs, name, x_data, y_data, z_data, **kwargs):
        axs.plot(x_data, y_data, z_data, label=name, **kwargs)

    def bar_3D(self, axs, name, x_data, y_data, z_data, x_val=None, y_val=None, z_val=None, **kwargs):
        if x_val is None or y_val is None or z_val is None:
            axs.bar(x_data, y_data, z_data, label=name, **kwargs)
        else:
            axs.bar3d(x_data, y_data, z_data, x_val, y_val, z_val, label=name, **kwargs)

    def scatter_3D(self, axs, name, x_data, y_data, z_data, **kwargs):
        axs.scatter(x_data, y_data, z_data, label=name, **kwargs)

    def stem_3D(self, axs, name, x_data, y_data, z_data, **kwargs):
        axs.stem(x_data, y_data, z_data, label=name, **kwargs)

    def tripcolor_3D(self, axs, name, x_data, y_data, z_data, **kwargs):
        axs.tripcolor(x_data, y_data, z_data, label=name)

    def voxels_3D(self, axs, name, x_data, y_data, z_data, **kwargs):
        axs.voxels(x_data, y_data, z_data, label=name)

    def error_3D(self, axs, name, x_data, y_data, z_data, x_err, y_err, z_err, **kwargs):
        axs.errorbar(x_data, y_data, z_data, x_err=x_err, y_err=y_err, z_err=z_err, label=name, **kwargs)

    def quiver_3D(self, axs, name, x_data, y_data, z_data, x_grad, y_grad, z_grad, **kwargs):
        axs.quiver(x_data, y_data, z_data, x_grad, y_grad, z_grad, label=name, **kwargs)

    def contour_3D(self, axs, name, x_data, y_data, z_data, style=None, **kwargs):
        if style is None:
            axs.contour(x_data, y_data, z_data, label=name, **kwargs)
        elif style in ['l_2d', 'line_2d']:
            axs.contour(x_data, y_data, z_data, extend3d=False, label=name, **kwargs)
        elif style in ['a_2d', 'area_2d']:
            axs.contourf(x_data, y_data, z_data, extend3d=False, label=name, **kwargs)
        elif style in ['l_3d', 'line_3d']:
            axs.contour(x_data, y_data, z_data, extend3d=True, label=name, **kwargs)
        elif style in ['a_3d', 'area_3d']:
            axs.contourf(x_data, y_data, z_data, extend3d=True, label=name, **kwargs)

    def tricontour_3D(self, axs, name, x_data, y_data, z_data, style=None, **kwargs):
        if style is None:
            axs.tricontour(x_data, y_data, z_data, label=name, **kwargs)
        elif style in ['l_2d', 'line_2d']:
            axs.tricontour(x_data, y_data, z_data, extend3d=False, label=name, **kwargs)
        elif style in ['a_2d', 'area_2d']:
            axs.tricontourf(x_data, y_data, z_data, extend3d=False, label=name, **kwargs)
        elif style in ['l_3d', 'line_3d']:
            axs.tricontour(x_data, y_data, z_data, extend3d=True, label=name, **kwargs)
        elif style in ['a_3d', 'area_3d']:
            axs.tricontourf(x_data, y_data, z_data, extend3d=True, label=name, **kwargs)
