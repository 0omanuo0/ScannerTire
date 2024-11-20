import matplotlib.pyplot as plt
import numpy as np


def interactive_plot(y, x=None, xlabel='x', ylabel='y', title='Interactive Plot'):
    """
    Creates an interactive plot with a hover feature to display the value of each point.

    Parameters:
        y (array-like): Data for the y-axis.
        x (array-like, optional): Data for the x-axis. If None, indices of y will be used.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        title (str): Title of the plot.

    Returns:
        None
    """
    if x is None:
        x = range(len(y))  # Generate x as indices if not provided

    fig, ax = plt.subplots()
    line, = ax.plot(x, y, picker=5)  # 'picker=5' activates hover sensitivity

    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.grid()

    # Create annotation text
    annot = ax.annotate("", xy=(0, 0), xytext=(-40, 20),
                        textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    # Update the annotation
    def update_annot(ind):
        index = ind["ind"][0]
        annot.xy = (x[index], y[index])
        text = f"{xlabel}={x[index]:.2f}\n{ylabel}={y[index]:.2f}"
        annot.set_text(text)
        annot.get_bbox_patch().set_alpha(0.4)

    # Event handler for mouse hover
    def on_hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = line.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

    # Connect the hover event to the plot
    fig.canvas.mpl_connect("motion_notify_event", on_hover)

    # Show the plot
    plt.show()