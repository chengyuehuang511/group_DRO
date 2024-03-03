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


def summary_epoch_group(output, column_list, name_list, color, criterion='loss', split="train", axs=None):
    marker_styles = ['o', 's', '^', 'D', '*', 'p', 'h', '+', 'x', 'd']

    # plt.figure(figsize=(9, 6))
    for i, (column, name) in enumerate(zip(column_list, name_list)):
        axs.plot(output[column], label=name, marker=marker_styles[i], markersize=2, color=color[i])

    # Adding title and labels
    if criterion == "loss_each_uniform":
        criterion = "loss_uniform"

    axs.set_title(f'{criterion}', fontsize=14)
    # axs.xlabel('Epoch')
    # axs.ylabel(f'Average {criterion}')

    # Show legend
    # plt.legend()

    # # Save the figure
    # plt.savefig(f'plot/{split}/avg_{criterion}_groups.png', format='png', dpi=300)
    # plt.close()


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


    # output = pd.read_csv("logs_metrics_fix_epoch300/train_metrics.csv", index_col=0)
    # output_order = pd.read_csv("logs_order/train_metrics.csv", index_col=0)

    # d = {}
    # for col, order_col in zip(["grad_norm", "loss", "predictions", "feat_norm"], 
    #                           ["grad", "grad", "", "feat"]):
    #     # print(f"========== {col} ==========")
    #     for i in range(4):
    #         d[col + f'_g{i}'] = []
        
    #     for epoch in range(50):
    #         if order_col == "":
    #             name = f'aux_labels_{epoch}'
    #             label_name = f"labels_{epoch}"
    #         else:
    #             name = f'aux_labels_{order_col}_{epoch}'
    #             label_name = f"labels_{order_col}_{epoch}"
            
    #         g1 = output[(output_order[name] == 0)]
    #         g2 = output[(output_order[name] == 1)]  #
    #         g3 = output[(output_order[name] == 2)]  #
    #         g4 = output[(output_order[name] == 3)]

    #         label1 = output_order[(output_order[name] == 0)][label_name]
    #         label2 = output_order[(output_order[name] == 1)][label_name]
    #         label3 = output_order[(output_order[name] == 2)][label_name]
    #         label4 = output_order[(output_order[name] == 3)][label_name]

    #         for i, (g, label) in enumerate(zip([g1, g2, g3, g4], [label1, label2, label3, label4])):
    #             # print(f"=====Group {i} {(len(g))}=====")
    #             # print(g[col].mean())
    #             if col == "predictions":
    #                 d[col + f'_g{i}'].append((label == g[col + f'_{epoch}']).mean())
    #             else:
    #                 d[col + f'_g{i}'].append(g[col + f'_{epoch}'].mean())
        
    #     plot_category(d, col, f'{col}_plot.png')
    # print(d)

    def summary_wrt_epoch_per_group(dataset="waterbird", log_root="logs_fix", epoch_max=100):
        loss_column_list = ['avg_loss_group:0', 'avg_loss_group:1', 'avg_loss_group:2', 'avg_loss_group:3']
        acc_column_list = ['avg_acc_group:0', 'avg_acc_group:1', 'avg_acc_group:2', 'avg_acc_group:3']
        avg_group_grad_norm = ['avg_group_grad_norm:0', 'avg_group_grad_norm:1', 'avg_group_grad_norm:2', 'avg_group_grad_norm:3']
        avg_group_grad_norm_uniform = ['avg_group_grad_norm_uniform:0', 'avg_group_grad_norm_uniform:1', 'avg_group_grad_norm_uniform:2', 'avg_group_grad_norm_uniform:3']
        avg_group_loss_each_uniform = ['avg_group_loss_each_uniform:0', 'avg_group_loss_each_uniform:1', 'avg_group_loss_each_uniform:2', 'avg_group_loss_each_uniform:3']
        avg_group_feat_norm = ['avg_group_feat_norm:0', 'avg_group_feat_norm:1', 'avg_group_feat_norm:2', 'avg_group_feat_norm:3']

        for split in ["train", "val", "test"]:
            output = pd.read_csv(f"{log_root}/{split}.csv")
            output = output[output['epoch'] < epoch_max]
            update_data_count_group = [f"update_data_count_group:{i}" for i in range(4)]
            count = output[output['epoch'] == 0][update_data_count_group].sum(axis=0).tolist()
            count = [int(c) for c in count]
            # for the output with same epoch, only keep the row where batch is the biggest
            output = output.sort_values(by=['epoch', 'batch'], ascending=False).drop_duplicates(subset=['epoch'])
            # output ascending order
            output = output.sort_values(by=['epoch'])
            # output reset index
            output = output.reset_index(drop=True)
            print(output)

            if dataset == "waterbird":
                name_list = [f'Landbird on Land ({count[0]})', f'Landbird on Water ({count[1]})', f'Waterbird on Land ({count[2]})', f'Waterbid on Water ({count[3]})']
                color_list = ['darkred', 'lightcoral', 'lightblue', 'darkblue']
            elif dataset == "celebA":
                name_list = [f'Non-blond Hair, Female ({count[0]})', f'Non-blond Hair, Male ({count[1]})', f'Blond Hair, Female ({count[2]})', f'Blond Hair, Male ({count[3]})']
                color_list = ['lightcoral', 'darkred', 'darkblue', 'lightblue']
            
            fig, axs = plt.subplots(3, 2, figsize=(10, 10))

            summary_epoch_group(output, loss_column_list, name_list, criterion='loss', split=split, axs=axs[0, 0], color=color_list)
            summary_epoch_group(output, avg_group_grad_norm, name_list, criterion='grad_norm', split=split, axs=axs[0, 1], color=color_list)
            summary_epoch_group(output, acc_column_list, name_list, criterion='accuracy', split=split, axs=axs[1, 0], color=color_list)
            summary_epoch_group(output, avg_group_feat_norm, name_list, criterion='feat_norm', split=split, axs=axs[1, 1], color=color_list)
            summary_epoch_group(output, avg_group_loss_each_uniform, name_list, criterion='loss_each_uniform', split=split, axs=axs[2, 0], color=color_list)
            summary_epoch_group(output, avg_group_grad_norm_uniform, name_list, criterion='grad_norm_uniform', split=split, axs=axs[2, 1], color=color_list)

            handles, labels = axs[0, 0].get_legend_handles_labels()
            if split == "train":
                if dataset == "waterbird":
                    axs[0, 1].legend(handles, labels, loc='upper right')
                elif dataset == "celebA":
                    axs[0, 0].legend(handles, labels, loc='upper right')
            elif split == "val" or split == "test":
                if dataset == "waterbird":
                    axs[2, 1].legend(handles, labels, loc='lower right')
                elif dataset == "celebA":
                    axs[0, 0].legend(handles, labels, loc='upper left')
            fig.suptitle(f'Average criterion per group [{split}]', fontsize=16)
            # set x-axis label for the whole plot
            fig.supxlabel("Epoch", fontsize=14)
            
            fig.tight_layout()
            # fig.savefig(f'plot_{dataset}/{split}/{dataset}_avg_criterion_groups_{split}_reweight_classes.png', format='png', dpi=300)
            fig.savefig(f'plot_{dataset}/{split}/{dataset}_avg_criterion_groups_{split}.png', format='png', dpi=300)
    
    summary_wrt_epoch_per_group("waterbird", "logs_fix", 100)
    summary_wrt_epoch_per_group("celebA", "logs_celebA_a40", 50)

        
          