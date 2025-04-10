import matplotlib.pyplot as plt

if __name__ == '__main__':
    models = ["CascadePSP", "GLNet", "FCtL", "STDC", "ISDNet", "WSDNet", "MYNet"]
    iou = [69.4, 71.2, 73.7, 72.44, 74.23, 75.2, 76.5]  # Mean IoU (%)
    fps = [0.03, 0.05, 0.04, 4.97, 6.9, 7.8, 9.49]  # Inference Speed (FPS)

    # Highlight specific model
    highlight_models = ["MYNet"]
    highlight_indices = [models.index(m) for m in highlight_models]

    # Create plot
    plt.figure(figsize=(9, 5))
    plt.scatter(fps, iou, color='blue', label="Other Models")  # Blue points for all models
    plt.scatter([fps[i] for i in highlight_indices], [iou[i] for i in highlight_indices], color='red',
                label="Ours")  # Red highlight for MYNet
    plt.grid(True, linestyle='-', color='gray', alpha=0.1)

    # Annotate models with text to the right of the points
    for i, model in enumerate(models):
        if model == "MYNet":
            # Place MYNet label below the point
            plt.annotate(model, (fps[i], iou[i]), textcoords="offset points", xytext=(-27, -20), ha='left', fontsize=16)
        else:
            # Place other model labels to the right of the points
            plt.annotate(model, (fps[i], iou[i]), textcoords="offset points", xytext=(8, 0), ha='left', fontsize=16)

    # Axis labels
    plt.xlabel("Inference Speed (FPS)",fontsize=18)
    plt.ylabel("Mean IoU (%)",fontsize=18)
    # plt.title("Model Performance Comparison")

    # Legend and grid
    plt.legend()
    plt.grid(True)

    # Display plot
    plt.show()
