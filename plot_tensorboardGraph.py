import random

import numpy as np
import tensorboard as tb
import pandas as pd
from matplotlib import pyplot as plt

colors = [
    '#1f77b4',  # Blu
    '#ff7f0e',  # Arancione
    '#2ca02c',  # Verde
    '#d62728',  # Rosso
    '#9467bd',  # Viola
    '#e377c2',  # Rosa
    '#8c564b',  # Marrone
    '#7f7f7f',  # Grigio
    '#bcbd22',  # Giallo
    '#17becf'   # Ciano
]


def get_csv():
    experiment_id = "lV7YojHGTvOneW4ZDRaLdA"
    experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
    df = experiment.get_scalars()
    df.to_csv("csv_tensorboard/experiments.csv")


def main():
    df = pd.read_csv('csv_tensorboard/experiments.csv')

    df = df[df['run'].str.contains("COMA_FULL_FRAME")]

    names_graph = sorted(np.unique(df[['tag']]))
    graph = {}
    for t in names_graph:
        tmp_vec = {}
        tmp = df[df['tag'].str.contains(t)]
        runs = np.unique(tmp[['run']])
        for r in runs:
            current_df = df[(df['tag'].str.contains(t)) & (df['run'].str.contains(r))]
            tmp_step = []
            tmp_value = []
            for index in current_df.index:
                tmp_value.append(current_df['value'][index])
                tmp_step.append(current_df['step'][index])
            tmp_vec[r] = [tmp_step, tmp_value]
        graph[t] = tmp_vec

    for k, v in graph.items():
        save_path = "csv_tensorboard/graph/" + k.replace("/", "-") + ".pdf"
        j = 0
        for k_run, v_run in v.items():
            if "Classification" in k:
                if "COMA_Florence" in k_run:
                    leg = k_run[15:-31]
                else:
                    if "COMA_FULL_FRAME" in k_run:
                        leg = k_run[15:-33]
                    else:
                        leg = k_run[15:-22]
            else:
                if "COMA_Florence" in k_run:
                    leg = k_run[:-31]
                else:
                    leg = k_run[:-22]

            if "train" in k:
                plt.xlabel('Time-Step')
                plt.ylabel('Loss')
            else:
                if ("Validation_Accuracy") in k:
                    plt.xlabel('Time-Step')
                    plt.ylabel('Accuracy')
                if "validation" in k:
                    plt.xlabel('Time-Step')
                    plt.ylabel('Loss')

            alpha = 0.2  # Fattore di smoothing
            ema = [v_run[1][0]]  # Il primo valore è uguale al dato iniziale

            for i in range(1, len(v_run[1])):
                ema.append(alpha * v_run[1][i] + (1 - alpha) * ema[i - 1])

            red = random.randint(0, 255)
            green = random.randint(0, 255)
            blue = random.randint(0, 255)

            # Converti i valori RGB in esadecimale
            hex_color = "#{:02x}{:02x}{:02x}".format(red, green, blue)
            # Crea il grafico
            plt.plot(v_run[0], ema, label=leg, c=colors[j])
            plt.plot(v_run[0], v_run[1], c=colors[j], alpha=0.2)
            j = j + 1

        plt.grid(True)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.savefig(save_path, bbox_inches='tight', dpi=3000, format="pdf")
        plt.close()


if __name__ == "__main__":
    main()
