import matplotlib.pyplot as plt


def plot_box(np_img, np_box):

    # Create a figure and axis
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(np_img)

    for obj in np_box:
        name = 'bird'
        rect = plt.Rectangle((obj[0], obj[1]), obj[2] - obj[0], obj[3] - obj[1], fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)

    plt.show()