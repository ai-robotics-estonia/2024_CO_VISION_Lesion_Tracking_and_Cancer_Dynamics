import json
from pathlib import Path

import sys

from numpy import append
sys.path.append("../../..")          # bm-ai-pipelines/common


from common.analytics import get_case_metrics, calculate_performance
from common.analytics_bbox import get_pred_true_metrics
from common.visualise import visualize_pr_curve, read_merge_metrics

def calculate_pr_curve(gt, pred, log_dir, thr_list, subset, classes):
    pr_curve_list = []

    log_dir = Path(log_dir)
    metrics_dir = log_dir / "metrics"
    output_dir = log_dir / "outputs"

    output_dir.mkdir(parents=True, exist_ok=True)
        
    for thr in thr_list:
        calculate_intersection_stats(
            file_true=gt,
            file_pred=pred,
            output_dir=output_dir,
            metrics_dir=metrics_dir,
            thr=thr,
            subset=subset,
            classes=classes
        )

    for cls in classes:
        metrics_dir_cls = metrics_dir / f"class={cls}"
        df_metrics = read_merge_metrics(metrics_dir_cls)
        visualize_pr_curve(df_metrics, ["PR curve"], log_dir / f"pr_curve-class={cls}.png")
        pr_curve_list.append(df_metrics.sort_values('Thr'))

    return pr_curve_list

def calculate_intersection_stats(file_true, file_pred, output_dir, metrics_dir, thr, subset, classes):
    output_file = output_dir / f"thr={thr:.2f}.json"
    get_pred_true_metrics(file_true, file_pred, output_file, thr, subset, classes)

    for cls in classes:
        metrics_dir_cls = metrics_dir / f"class={cls}"
        metrics_dir_cls.mkdir(parents=True, exist_ok=True)
        metrics_file = metrics_dir_cls / f"thr={thr:.2f}.csv"   
        
        with open(output_file, 'r') as f:
            j = json.load(f)
            m = get_case_metrics(j, class_true=cls, class_pred=cls, min_dice_for_match=0.1)
            p = calculate_performance(m, metrics_file)
            print(p)