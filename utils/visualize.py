import matplotlib.pyplot as plt

def plot_best_model_results(
    metrics_df,
    metric_col="f1_test",
    vectorizer_name="CountVectorizer (BOW)",
    total_configs=None,
    ylim=(0.8, 1.0),
    color="#693FF3"
):
    """
    Generates two plots:
    1. Best F1 per model
    2. Best F1 per model + hyperparameters
    """

    # Get best config per model
    best_per_model = (
        metrics_df
        .sort_values(metric_col, ascending=False)
        .groupby("model")
        .first()
        .reset_index()
    )

    models = best_per_model["model"]
    scores = best_per_model[metric_col]

    config_text = (
        f"{total_configs} Configs tested per Model"
        if total_configs else ""
    )

    # --------------------------
    # Plot 1: Only Best F1
    # --------------------------
    plt.figure(figsize=(8, 5))

    bars = plt.bar(models, scores, color=color, width=0.6)

    plt.title(
        f"Best {metric_col} per Model | {config_text} | {vectorizer_name}",
        fontsize=10
    )
    plt.ylabel(metric_col)
    plt.ylim(ylim)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            height + 0.005,
            f"{height*100:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.tight_layout()
    plt.show()

    # --------------------------
    # Plot 2: Best F1 + Params
    # --------------------------
    plt.figure(figsize=(8, 5))

    bars = plt.bar(models, scores, color=color, width=0.6)

    plt.title(f"Best {metric_col} Configured Hyperparameters | {vectorizer_name}",fontsize=10)
    plt.ylabel(metric_col)
    plt.ylim(ylim)

    for i, bar in enumerate(bars):
        height = bar.get_height()

        label_text = (
            f"F1={height*100:.2f}%\n"
            f"ngram={best_per_model['ngram_range'].iloc[i]}\n"
            f"min_df={best_per_model['min_df'].iloc[i]}\n"
            f"max_df={best_per_model['max_df'].iloc[i]}"
        )

        plt.text(
            bar.get_x() + bar.get_width()/2,
            height + 0.005,
            label_text,
            ha="center",
            va="bottom",
            fontsize=8
        )

    plt.tight_layout()
    plt.show()