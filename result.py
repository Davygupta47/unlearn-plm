import json, glob, os
from IPython.display import Image, display

# Show eval metrics
eval_files = glob.glob('/content/unlearn-plm/output/tofu/eval/*.json', recursive=True)
for f in eval_files:
    print(f'\n=== {f} ===')
    with open(f) as fp:
        try:
            data = json.load(fp)
            print(json.dumps(data, indent=2))
        except Exception as e:
            print(f'Error reading {f}: {e}')

# Show MIA AUC
auc_file = '/content/unlearn-plm/output/tofu/mia/auc.txt'
if os.path.exists(auc_file):
    print('\n=== MIA AUC ===')
    print(open(auc_file).read())
else:
    print(f'MIA AUC file not found at {auc_file}')

# Show MIA plot
auc_png = '/content/unlearn-plm/output/tofu/mia/auc.png'
if os.path.exists(auc_png):
    print('\n=== MIA ROC Curve ===')
    display(Image(auc_png))
else:
    print(f'MIA plot not found at {auc_png}')
