import matplotlib.pyplot as plt
import torch
import numpy as np
from data_loader import SceneLoader
from models import cnn
from scipy import stats

saved_model = 'saved_models/shapenet3d_prob_full_dataset_v1.pt'
conv_dims = 3
model = cnn.ShapeNet(out_size=4, conv_dims=conv_dims).to('cuda')
model.load_state_dict(torch.load(saved_model))
print('Loaded saved model from: ', saved_model)

root_dirs = ['scenes/single_shape_plain', 'scenes/single_shape_textured_v2']
scenes = SceneLoader(root_dirs=root_dirs, n_scenes=2000, n_frames=20, device='cuda')

test_idxs = scenes.test_idxs
print('Initialized Data Loader')

plot_scenes = np.concatenate((test_idxs[:8], test_idxs[-8:]))
fig, axs = plt.subplots(4, 4, figsize=(18,18))
for i in range(16):
    axis = axs.flat[i]
    axis.set_title(f'Scene {test_idxs[i]} Predictions')
    data = scenes.get_scene(plot_scenes[i])
    gt_shape = data['shape_params'].mean(axis=0) 
    print(f'Scene {test_idxs[i]}: ground truth params: ', gt_shape)
    frames = data['frame']
    if conv_dims == 3:
        frames = frames.reshape(1, 20, 256, 256).unsqueeze(0)

    print('Generating predictions')
    pred_shape = model(frames).detach().cpu().numpy()
    print(pred_shape)
    axis.axvline(gt_shape[0], linewidth=2, label='Ground Truth Shape Params 0', color='red')
    axis.axvline(gt_shape[1], linewidth=2, label='Ground Truth Shape Params 1', color='blue')
    axis.set_xlim(0, 4) 
    colors=['red', 'blue']
    for j in range(2):
        mu = pred_shape[j, 0]
        sigma = pred_shape[j, 1]     
        x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
        axis.plot(x, stats.norm.pdf(x, mu, sigma), color=colors[j], label=f'Predicted Estimate {j}')
    
    if i >= 8:
        axis.set_title('Textured Background')
    else:
        axis.set_title('Plain Background')

print('Saving figure to: estimate_visualization.png')
handles, labels = axis.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')

fig.suptitle('Shape Estimates')
plt.savefig('estimate_visualization.png')
