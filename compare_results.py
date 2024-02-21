import glob
import os
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

LAKE_STRING = "lake={},few_shot={}"
RESULTS_PATH = os.path.join("global_results", "results.pkl")
BASE_FOLDER = "/outputs/agamotto-eye"
PLOT_CLASS_BARS = False
RUN_NAME = 'v1_v4_random_al'


lakes = ['paris6k','places','fsod-base','fsod-novel']
# search_strings = ["17-01*v1", "10-02*logistic", "24-02*no_dropout_restart", "24-02*variable_support_restart", "*simple_matrix*", "*big_query*"]
# names = ["v1", "logistic", "attention_bug-restart", "attention_bug-variable_support_restart", "simple_matrix-variable_support_restart", "big_query"]
# lakes = ['fsod-base', 'fsod-novel']
# search_strings = ["*kmeans++_closest_several_seeds", "*al_several_seeds", "*al+*", "*spectral+svc_tree3_first_several_seeds"]
# names = ["kmeans++-closest", "al", "al+kmeans", "spectral+tree"]
# search_strings = ["*kmeans++_closest_several_seeds", "*al_several_seeds", "*v4-final_several_seeds"]
# names = ["bl++", "al", "v4-final"]
# search_strings = ["*pp-new_code_random50", "*pp-update_v_and_P_new_code"]
# names = ["update P", "update P and v"]
# search_strings = ["*mean_pos-new_code_random50*", "*logistic-new_code_random50", "*pp-new_code_random50"]
# names = ["ProtoNet", "Logistic Regression", "HyperClass (Ours)"]
# search_strings = ["*svm-new_code_random50", "*blip_svm", "*vit_svm"]
# names = ["resnet50", "BLIP", "ViT"]
# search_strings = ["*v4_clip_new_benchmark", "*v4_vit_new_benchmark", "*blip_v4_new_benchmark", "*v4_new_benchmark", "*v4_new_benchmark_bbox"]
# names = ["CLIP", "ViT", "BLIP", "Resnet50", "Resnet50-bbox"]
# search_strings = ["*v4_clip_new_benchmark_bbox", "*v4_vit_new_benchmark_bbox", "*v4_blip_new_benchmark_bbox", "*v4_new_benchmark_bbox"]
# names = ["CLIP", "ViT", "BLIP", "Resnet50"]
# search_strings = ["*v4_new_benchmark", "*run_eva02_764_cs", "old_new_new/*run_764_22k_svm", "old_new_new/*run_764_22k_v4", "*run_dinov2_768_svm","*run_dinov2_768_cs", "*run_dinov2_768_v4"]
# names = ["ResNet50_2048_V4", "EVA02_768_CS", "EVA02_768_SVM", "EVA02_768_V4", "DINOv2_768_SVM","DINOv2_768_CS", "DINOv2_768_V4"]
search_strings = ["*v4_new_benchmark", "*run_eva02_768_cs", "*run_eva02_768_svm","*run_eva02_768_rocchio_0.7"]
names = ["ResNet50_2048_V4", "EVA02_768_CS", "EVA02_768_SVM", "EVA02_768_Rocchio_(0.7,0.3)"]

#search_strings = ["*v4_new_benchmark","*run_eva02_764_cs","*run_eva02_768_rocchio_0.5","*run_eva02_768_rocchio_0.6","*run_eva02_768_rocchio_0.7","*run_eva02_768_rocchio_0.8","*run_eva02_768_rocchio_0.9"]
#names = ["ResNet50_2048_V4", "EVA02_768_CS", "EVA02_768_Rocchio_(0.5,0.5)", "EVA02_768_Rocchio_(0.6,0.4)", "EVA02_768_Rocchio_(0.7,0.3)", "EVA02_768_Rocchio_(0.8,0.2)", "EVA02_768_Rocchio_(0.9,0.1)"]
#search_strings = ["*v1_new_benchmark", "*run_384_svm", "*run_384_22k_svm", "*run_768_svm", "*run_764_22k_svm", "*run_1024_22k_svm"]
#names = ["ResNet50_2048_SVM","EVA02_384_1K_SVM", "EVA02_384_22K_SVM", "EVA02_768_1K_SVM", "EVA02_768_22K_SVM", "EVA02_1024_22K_SVM"]
#search_strings = ["*v4_new_benchmark", "*run_384_v4", "*run_384_22k_v4", "*run_768_v4", "*run_764_22k_v4", "*run_1024_22k_v4"]
#names = ["ResNet50_2048_V4","EVA02_384_1K_V4", "EVA02_384_22K_V4", "EVA02_768_1K_V4", "EVA02_768_22K_V4", "EVA02_1024_22K_V4"]

# search_strings = ["*pp-new_code_random50", "*blip_pp", "*vit_pp"]
# names = ["resnet50", "BLIP", "ViT"]
# search_strings = ["*pp-new_code_no_pretraining_random50", "*pp-new_code_pretrain_v_random50", "*pp-new_code_pretrain_P_random50", "*pp-new_code_random50"]
# names = ["No Meta-training", "Meta-train v", "Meta-train P", "Meta-train both"]
# search_strings = ["*v1-no_bbox", "*v4-no_bbox_several_seeds", "*v4-random_several_seeds", "*v4-final_several_seeds"]
# names = ["v1", "v4-FS", "v4-FS+bbox", "v4-FS+bbox+AL"]
# search_strings = ["*kmeans++_closest_several_seeds", "*al_several_seeds", "spectral+tree"]
# names = ["kmeans++-closest", "al"]
# search_strings = ["*kmeans_new_heuristic", "*al_new_heuristic", "*v4-kmeans_closest_new_heuristic", "*upgrade_kmeans*"]
# names = ["v4-bl_steps_heuristic", "v4-al", "v4-kmeans_closest", "v4-improved_kmeans_longer"]
# search_strings = ["*v4-kmeans", "*kmeans_new_heuristic", "*al_new_heuristic"]
# names = ["v4-bl",  "v4-bl_steps_heuristic", "v4-al"]
iters = [str(i) for i in range(6)]


for lake in lakes:
    folder_path = os.path.join('/outputs/comparisons', RUN_NAME, lake)
    Path(folder_path).mkdir(parents=True, exist_ok=True)
    lake_path = os.path.join(BASE_FOLDER, LAKE_STRING.format(lake, lake))
    ap = []
    p_at_50 = []
    p_at_50_per_class = []
    ap_per_class = []
    categories = []
    lake_names = []
    for i, s in enumerate(search_strings):
        experiment_path = glob.glob(os.path.join(lake_path, s))
        if len(experiment_path) == 0:
            continue
        lake_names.append(names[i])

        cur_ap = []
        cur_p_at_50 = []
        cur_ap_per_class = []
        cur_p_at_50_per_class = []

        if len(os.listdir(experiment_path[0])) < 10:  #TODO handles situation of random seeds but has to be more generic
            results_paths = [os.path.join(experiment_path[0], d, RESULTS_PATH) for d in os.listdir(experiment_path[0])
                             if os.path.isdir(os.path.join(experiment_path[0], d))]
        else:
            results_paths = [os.path.join(experiment_path[0], RESULTS_PATH)]

        for results_path in results_paths:
            if not os.path.exists(results_path):
                continue

            with open(results_path, 'rb') as f:
                results = pickle.load(f)
                cur_ap.append(results['ap'].mean(axis=(0, 2))[:len(iters)])
                cur_p_at_50.append(results['p@k'][..., -1].mean(axis=(0, 2))[:len(iters)])
                cur_p_at_50_per_class.append(results['p@k'][..., -1].mean(axis=2))
                cur_ap_per_class.append(results['ap'].mean(axis=2))
                if 'categories' in results and len(categories) == 0:
                    categories = list(results['categories'])

        if len(cur_ap) == 0:
            continue

        ap.append(np.stack(cur_ap).squeeze())
        p_at_50.append(np.stack(cur_p_at_50).squeeze())
        ap_per_class.append(np.stack(cur_ap_per_class).mean(axis=0))
        p_at_50_per_class.append(np.stack(cur_p_at_50_per_class).mean(axis=0))

    colors = ["C{}".format(i) for i in range(len(p_at_50_per_class))]
    markers = ['-o', '--s', ':^', '-.x', '-p','--r','-k']

    for name, a, c, m in zip(lake_names, ap, colors, markers):
        if len(a.shape) > 1:
            plt.plot(iters, a.mean(axis=0), m, label=name, color=c)
            plt.fill_between(iters, a.mean(axis=0) + a.std(axis=0),
                             a.mean(axis=0) - a.std(axis=0), alpha=0.5, color=c)
            # plt.plot([1, 11, 21, 31], a.mean(axis=0), m, label=name, color=c)
            # plt.fill_between([1, 11, 21, 31], a.mean(axis=0) + a.std(axis=0),
            #                  a.mean(axis=0) - a.std(axis=0), alpha=0.5, color=c)
        else:
            plt.plot(iters, a, m, label=name, color=c)

    # s = {'paris6k': 'Paris-6K', 'places': 'Places', 'fsod-base': 'FSOD', 'fsod-novel': 'FSOD'}[lake]
    # plt.title("Mean Average Precision - {}".format(s))
    plt.title("Mean Average Precision - {}".format(lake))
    plt.xlabel("Iter")
    # plt.xlabel("Number of Samples")
    # plt.xticks([1, 11, 21, 31], [1, 11, 21, 31])
    plt.ylabel("mAP")
    plt.legend()
    plt.grid()
    plt.show()
    # plt.savefig(os.path.join(folder_path, 'mAP.png'))
    # plt.close()
    #continue
    for name, p, c, m in zip(lake_names, p_at_50, colors, markers):
        if len(p.shape) > 1:
            # plt.plot([1, 11, 21, 31], p.mean(axis=0), m, label=name, color=c)
            plt.plot(iters, p.mean(axis=0), m, label=name, color=c)
            # plt.fill_between([1, 11, 21, 31], p.mean(axis=0) + p.std(axis=0),
            #                  p.mean(axis=0) - p.std(axis=0), alpha=0.5, color=c)
            plt.fill_between(iters, p.mean(axis=0) + p.std(axis=0),
                             p.mean(axis=0) - p.std(axis=0), alpha=0.5, color=c)
        else:
            plt.plot(iters, p, m, label=name, color=c)

    plt.title("Precision@50 - {}".format(lake))
    # plt.xlabel("Number of Samples")
    plt.xlabel("Iter")
    # plt.xticks([1, 11, 21, 31], [1, 11, 21, 31])
    plt.ylabel("P@50")
    plt.legend()
    plt.grid()
    # plt.savefig(os.path.join(folder_path, 'p@50'))
    # plt.close()
    plt.show()

    if PLOT_CLASS_BARS:
        for i in iters:
            means = [p[:, i].mean() for p in p_at_50_per_class]
            for j, c in enumerate(categories):
                ps = [p[j, i] for p in p_at_50_per_class]
                for k, (p, name, color, m) in enumerate(sorted(zip(ps, lake_names, colors, means))):
                    label = {'{}: {:.4f}'.format(name, m)} if j == 0 else ''
                    plt.bar(c, p, label=label, color=color, zorder=-k)

            plt.title("P@50 - {} (iter={})".format(lake, i))
            plt.xlabel("category")
            plt.ylabel("p@50")
            plt.xticks(fontsize=6, rotation='vertical')
            plt.legend(loc='lower right', fontsize=6)
            plt.savefig(os.path.join(folder_path, 'p@50-iter={}'.format(i)))
            plt.close()

        for i in iters:
            colors = ["C{}".format(i) for i in range(len(ap_per_class))]
            means = [p[:, i].mean() for p in ap_per_class]
            for j, c in enumerate(categories):
                ps = [p[j, i] for p in ap_per_class]
                for k, (p, name, color, m) in enumerate(sorted(zip(ps, lake_names, colors, means))):
                    label = {'{}: {:.4f}'.format(name, m)} if j == 0 else ''
                    plt.bar(c, p, label=label, color=color, zorder=-k)

            plt.title("AP - {} (iter={})".format(lake, i))
            plt.xlabel("category")
            plt.ylabel("ap")
            plt.xticks(fontsize=6, rotation='vertical')
            plt.legend(loc='lower right', fontsize=6)
            plt.savefig(os.path.join(folder_path, 'ap-iter={}'.format(i)))
            plt.close()
