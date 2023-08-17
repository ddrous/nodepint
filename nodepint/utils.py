## Just a bunch of utility functions


## Wrapper function for matplotlib and seaborn
def sbplot(*args, ax=None, figsize=(6,3.5), x_label=None, y_label=None, title=None, x_scale='linear', y_scale='linear', xlim=None, ylim=None, **kwargs):

    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(context='notebook', style='ticks',
            font='sans-serif', font_scale=1, color_codes=True, rc={"lines.linewidth": 2})
    plt.style.use("dark_background")

    if ax==None:
        _, ax = plt.subplots(1, 1, figsize=figsize)
    # sns.despine(ax=ax)
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    ax.plot(*args, **kwargs)
    ax.set_xscale(x_scale)
    ax.set_yscale(y_scale)
    if "label" in kwargs.keys():
        ax.legend()
    if ylim:
        ax.set_ylim(ylim)
    if xlim:
        ax.set_xlim(xlim)
    plt.tight_layout()
    return ax



def pvplot(x, y, show=True, xlabel=None, ylabel=None, title=None, ax=None, **kwargs):
    import pyvista as pv
    # pv.start_xvfb()           ## TODO Only do this on LINUX

    if ax is None:
        ax = pv.Chart2D()

    _ = ax.line(x, y, **kwargs)

    if xlabel is not None:
        ax.x_label = xlabel

    if ylabel is not None:
        ax.y_label = ylabel

    if title is not None:
        ax.title = title

    if show == True:
        ax.show()

    return ax
