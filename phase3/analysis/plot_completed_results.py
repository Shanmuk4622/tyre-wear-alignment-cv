"""Static analysis figures and protocol-qualified comparisons from audited CSV."""
import csv
import json
import sys
from pathlib import Path

assert Path(sys.prefix).name == 'cv_conda'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'completion_20261028'
CSV = OUT/'csv'
FIG = OUT/'figures'
FIG.mkdir(exist_ok=True)


def read(name):
    with open(CSV/name, newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def main():
    curves = read('validation_epochs.csv')
    summaries = read('run_summary.csv')
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    colors = ['#1469a8', '#cf5a24', '#298353']
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.4), constrained_layout=True)
    panels = [('balanced_accuracy_observed', 'Validation balanced accuracy (%)', 100),
              ('mean_two_region_dice', 'Validation mean two-region Dice (%)', 100),
              ('human_point_mae_width_fraction', 'Validation human-point error (% width)', 100),
              ('train_loss_classification', 'Mean sampled-batch training classification loss', 1)]
    stats = []
    for seed, color in zip((1, 2, 3), colors):
        rows = [r for r in curves if int(r['seed']) == seed]
        selected = next(r for r in rows if r['selected'] == 'True')
        for axis, (key, label, scale) in zip(axes.flat, panels):
            axis.plot([int(r['epoch']) for r in rows], [float(r[key])*scale for r in rows], color=color, label=f'Seed {seed}', linewidth=1.4)
            axis.axvline(int(selected['epoch']), color=color, linestyle=':', alpha=.5)
            axis.set(xlabel='Epoch', ylabel=label, xlim=(1, 60))
            axis.grid(alpha=.15)
        stats.append(dict(seed=seed, selected_epoch=int(selected['epoch']), validation_balanced_accuracy=float(selected['balanced_accuracy_observed']),
                   final_validation_balanced_accuracy=float(rows[-1]['balanced_accuracy_observed']),
                   first_train_classification_loss=float(rows[0]['train_loss_classification']), last_train_classification_loss=float(rows[-1]['train_loss_classification']),
                   first_gradient_norm=float(rows[0]['train_grad_norm_pre_clip']), last_gradient_norm=float(rows[-1]['train_grad_norm_pre_clip'])))
    axes[0, 0].set_ylim(-2, 102)
    axes[0, 0].legend(ncol=3, loc='upper right', fontsize=8)
    fig.suptitle('TREAD-ONE: stable region learning, unstable validation classification\nDotted lines mark the declared selected epochs; no curves are smoothed.', fontsize=13)
    fig.savefig(FIG/'training_validation_curves.png', dpi=170)
    plt.close(fig)
    predictions = read('test_predictions.csv')
    val = read('validation_predictions.csv')
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for seed in (1, 2, 3):
        epoch = stats[seed-1]['selected_epoch']
        for row_index, records in enumerate(([r for r in val if int(r['seed']) == seed and int(r['epoch']) == epoch],
                    [r for r in predictions if int(r['seed']) == seed and r['endpoint'] == 'selected'])):
            matrix = np.zeros((3, 3), dtype=int)
            for r in records:
                matrix[int(r['label']), int(r['prediction'])] += 1
            ax = axes[row_index, seed-1]
            ax.imshow(matrix, vmin=0, vmax=62, cmap='Blues')
            for y in range(3):
                for x in range(3):
                    ax.text(x, y, str(matrix[y, x]), ha='center', va='center', color='white' if matrix[y, x] > 35 else '#1b2630')
            ax.set(xticks=range(3), xticklabels=['Low', 'Mid', 'High'], yticks=range(3), yticklabels=['Low', 'Mid', 'High'],
                   xlabel='Predicted mileage proxy', ylabel='Target mileage proxy',
                   title=f'Seed {seed}, epoch {epoch}: '+('validation' if row_index == 0 else 'test'))
    fig.suptitle('Selected checkpoints: validation success does not transfer to the held-out low tyre\nValidation: 81 images / 2 tyres. Test: 103 images / 2 tyres. Mid has no target examples.', fontsize=12)
    fig.savefig(FIG/'selected_confusion_matrices.png', dpi=170)
    plt.close(fig)
    registry = json.loads((ROOT.parent/'phase2/workstation/registry.json').read_text())
    (OUT/'PHASE2_INSTALLED_REGISTRY.json').write_text(json.dumps(registry, indent=2)+'\n')
    current = next(r for r in summaries if r['seed'] == '3' and r['endpoint'] == 'selected')
    comparison = []
    for key, metric, direction in [('mobilenetv4', 'balanced_accuracy_observed', 'higher'), ('resnet50', 'balanced_accuracy_observed', 'higher'),
                                   ('segformer', 'mean_two_region_dice', 'higher'), ('yolo26m', 'mean_two_region_dice', 'higher'),
                                   ('hrnet', 'human_point_mae_width_fraction', 'lower')]:
        baseline = registry[key+'-combined']['test_score']
        if key == 'hrnet':
            baseline = -baseline
        candidate = float(current[metric])
        comparison.append(dict(baseline=key, metric=metric, better_direction=direction, installed_phase2_value=baseline,
                          treadone_validation_selected_seed=3, treadone_selected_epoch=20, treadone_value=candidate,
                          candidate_minus_baseline=candidate-baseline,
                          scope='Same frozen Phase2 holdout; different architecture/initialization/joint-selection protocol. Descriptive, not a matched causal ablation.'))
    with open(CSV/'baseline_comparison.csv', 'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison[0]))
        writer.writeheader(); writer.writerows(comparison)
    with open(CSV/'selection_and_loss_diagnostics.csv', 'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(stats[0])); writer.writeheader(); writer.writerows(stats)
    (OUT/'INSIGHT_INPUTS.json').write_text(json.dumps(dict(selection_and_loss=stats, baseline_comparison=comparison), indent=2)+'\n')
    print(json.dumps(dict(selection_and_loss=stats, baseline_comparison=comparison), indent=2))


if __name__ == '__main__':
    main()
