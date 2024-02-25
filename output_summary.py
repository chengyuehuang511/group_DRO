import pandas as pd
import matplotlib.pyplot as plt

# Function to plot a specific category (either grad_norm or loss)
def plot_category(data, category, filename):
    plt.figure(figsize=(10, 6))  # Set the figure size
    for key, values in data.items():
        if category in key:  # Filter keys by category
            plt.plot(values, label=key)  # Plot each series
    plt.title(f'{category} over time')  # Set the title
    plt.xlabel('Step')  # Set the x-axis label
    plt.ylabel(category)  # Set the y-axis label
    plt.legend()  # Show legend
    
    # Save the figure
    plt.savefig(filename, format='png', dpi=300)  # Save as PNG with high DPI
    plt.close()  # Close the figure to free memory


def compare(df, c1, c2):
    return sum(df[c1] == df[c2]) / len(df)


def summary(output):
    output = pd.read_csv(output, index_col=0)
    g1 = output[(output['labels'] == 0) & (output['aux_labels'] == 0)]
    g2 = output[(output['labels'] == 0) & (output['aux_labels'] == 1)]  #
    g3 = output[(output['labels'] == 1) & (output['aux_labels'] == 0)]  #
    g4 = output[(output['labels'] == 1) & (output['aux_labels'] == 1)]
    
    for i, g in enumerate([output, g1, g2, g3, g4]):
        print(f"=====Group {i} {(len(g))}=====")
        print("Prediction accuracy:", compare(g, 'labels', 'predictions'))
        for simfunc in ["dotprod", "cossim", "l2sim"]:
            for a in ['labels', 'predictions', 'aux_labels']:
                for b in ['train_rank_top_label_' + simfunc, 'train_rank_top_aux_label_' + simfunc]:
                    print(f"Compare {a} and {b}", compare(g, a, b))


def summary_epoch_group(output, column_list, name_list, criterion='loss'):
    marker_styles = ['o', 's', '^', 'D', '*', 'p', 'h', '+', 'x', 'd']
    output = pd.read_csv(output, index_col=0).head(60)

    plt.figure(figsize=(10, 6))
    for i, (column, name) in enumerate(zip(column_list, name_list)):
        plt.plot(output[column], label=name, marker=marker_styles[i], markersize=3)

    # Adding title and labels
    plt.title(f'Average {criterion} per group')
    plt.xlabel('Epoch')
    plt.ylabel(f'Average {criterion}')

    # Show legend
    plt.legend()

    # Save the figure
    plt.savefig(f'avg_{criterion}_groups.png', format='png', dpi=300)
    plt.close()


if __name__ == "__main__":
    # # summary("output_new.csv")
    # # summary("output_grad_new.csv")
    # output = pd.read_csv("output_grad_new.csv", index_col=0)
    # # predict_wrong_0 = output[(output['labels'] == 1) & (output['train_rank_top_label_dotprod'] == 0)]
    # # spu_prop = sum(predict_wrong_0['aux_labels'] == 0)/len(predict_wrong_0)
    # # print(spu_prop)
    
    # # predict_wrong_0_ = output[(output['labels'] == 1) & (output['predictions'] == 0)]
    # # spu_prop_ = sum(predict_wrong_0_['aux_labels'] == 0)/len(predict_wrong_0_)
    # # print(spu_prop_)

    # # tmp = output[(output['labels'] == 0) & (output['predictions'] == 0) & (output['aux_labels'] == 1)]
    # # print(sum(tmp['train_rank_top_label_l2sim'] == 1) / len(tmp))

    # label_pred_dif = output[(output['labels'] != output['predictions'])]
    # label_pred_dif_minority = label_pred_dif[(label_pred_dif['labels'] != label_pred_dif['aux_labels'])]
    # print(len(label_pred_dif_minority) / len(label_pred_dif))

    # label_pred_dif = output[(output['labels'] != output['train_rank_top_label_dotprod'])]
    # label_pred_dif_minority = label_pred_dif[(label_pred_dif['labels'] != label_pred_dif['aux_labels'])]
    # print(len(label_pred_dif_minority) / len(label_pred_dif))

    # minority = output[(output['labels'] != output['aux_labels'])]
    # predict_wrong = minority[(minority['labels'] != minority['predictions'])]
    # rank_wrong = minority[(minority['labels'] != minority['train_rank_top_label_dotprod'])]
    # print(len(predict_wrong) / len(minority))
    # print(len(rank_wrong) / len(minority))
    
    # output = pd.read_csv("output_grad_loss_eval_try_2.csv", index_col=0)
    # g1 = output[(output['labels'] == 0) & (output['aux_labels'] == 0)]
    # g2 = output[(output['labels'] == 0) & (output['aux_labels'] == 1)]  #
    # g3 = output[(output['labels'] == 1) & (output['aux_labels'] == 0)]  #
    # g4 = output[(output['labels'] == 1) & (output['aux_labels'] == 1)]

    # for i, g in enumerate([output, g1, g2, g3, g4]):
    #     print(f"=====Group {i} {(len(g))}=====")
    #     print(g['loss'].mean())

    """test output"""
    # output = pd.read_csv("output_train_grad_loss_tmp.csv", index_col=0)
    # g1 = output[(output['aux_labels'] == 0)]
    # g2 = output[(output['aux_labels'] == 1)]  #
    # g3 = output[(output['aux_labels'] == 2)]  #
    # g4 = output[(output['aux_labels'] == 3)]

    # for col in ["grad_norm", "loss"]:
    #     print(f"========== {col} ==========")
    #     for i, g in enumerate([output, g1, g2, g3, g4]):
    #         print(f"=====Group {i} {(len(g))}=====")
    #         print(g[col].mean())
    
    # print("========== ACC ==========")
    # for i, g in enumerate([output, g1, g2, g3, g4]):
    #     print(f"=====Group {i} {(len(g))}=====")
    #     print((g["labels"] == g["predictions"]).mean())


    output = pd.read_csv("logs_metrics_fix_epoch1/val_metrics.csv", index_col=0)
    g1 = output[(output['aux_labels'] == 0)]
    g2 = output[(output['aux_labels'] == 1)]  #
    g3 = output[(output['aux_labels'] == 2)]  #
    g4 = output[(output['aux_labels'] == 3)]

    d = {}
    for col in ["grad_norm", "loss", "predictions"]:
        # print(f"========== {col} ==========")
        for i, g in enumerate([g1, g2, g3, g4]):
            d[col + f'_g{i}'] = []
            for epoch in range(1):
                # print(f"=====Group {i} {(len(g))}=====")
                # print(g[col].mean())
                if col == "predictions":
                    d[col + f'_g{i}'].append((g["labels"] == g[col + f'_{epoch}']).mean())
                else:
                    d[col + f'_g{i}'].append(g[col + f'_{epoch}'].mean())
        
        plot_category(d, col, f'{col}_plot.png')
    print(d)

    """summary wrt epoch per group"""
    # name_list = ['Landbird on Land (3498)', 'Landbird on Water (184)', 'Waterbird on Land (56)', 'Waterbid on Water (1057)']
    # loss_column_list = ['avg_loss_group:0', 'avg_loss_group:1', 'avg_loss_group:2', 'avg_loss_group:3']
    # acc_column_list = ['avg_acc_group:0', 'avg_acc_group:1', 'avg_acc_group:2', 'avg_acc_group:3']

    # summary_epoch_group("/nethome/chuang475/flash/projects/group_DRO/logs_a40/train.csv", loss_column_list, name_list, criterion='loss')
    # summary_epoch_group("/nethome/chuang475/flash/projects/group_DRO/logs_a40/train.csv", acc_column_list, name_list, criterion='accuracy')