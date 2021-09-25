import matplotlib.pyplot as plt


def create_figure(array):
    # this function is static, because it has noting to do with 'self',
    # so it is now outside the class,
    # please see below
    graph, ax = plt.subplots(figsize=(30, 10))
    ax.axis('off')
    # draw maze
    frame = ax.imshow(array, interpolation=None, cmap='magma')
    # disable automatically showing graph in notebook
    plt.close()
    return graph, ax, frame


class Graphic:
    def __init__(self, array):
        # background
        self.bg_figure, self.bg_ax, self.bg_frame = create_figure(array)

        # dynamic frame
        self.frame_figure, self.frame_ax, self.frame = create_figure(array)

    # 'Self' parameter is never used in the function definition,
    # so it is outside the class definition for now.

    # def create_figure(self, array):
    #     graph, ax = plt.subplots(figsize=(30, 10))
    #     ax.axis('off')
    #     # draw maze
    #     frame = ax.imshow(array, interpolation=None, cmap='magma')
    #     # disable automatically showing graph in notebook
    #     plt.close()
    #     return graph, ax, frame
