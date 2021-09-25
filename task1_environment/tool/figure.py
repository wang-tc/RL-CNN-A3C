import matplotlib.pyplot as plt


# https://stackoverflow.com/questions/31729948/matplotlib-how-to-show-a-figure-that-has-been-closed
def show_figure(fig):
    # create a dummy figure and use its
    # manager to display "fig"

    dummy = plt.figure()
    new_manager = dummy.canvas.manager
    new_manager.canvas.figure = fig
    fig.set_canvas(new_manager.canvas)

# show_figure(game.graphic.graph)