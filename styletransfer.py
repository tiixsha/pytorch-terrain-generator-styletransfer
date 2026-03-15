import torch
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from skimage.exposure import match_histograms
from skimage.metrics import structural_similarity as ssim
from scipy.stats import wasserstein_distance

def load_image(img_path, shape=512):
    image = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(shape),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def get_features(image, model, layers=None):
    if layers is None:
        layers = {
            '0': 'conv1_1',
            '5': 'conv2_1',
            '10': 'conv3_1',
            '19': 'conv4_1',
            '21': 'conv4_2',  
            '28': 'conv5_1'
        }
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    return torch.mm(tensor, tensor.t())

def total_variation_loss(image):
    loss = torch.sum(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) + \
           torch.sum(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
    return loss

def save_terrain(target_tensor, style_tensor, filename):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(target_tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(target_tensor.device)
    
    def to_numpy(t):
        t = (t.detach().squeeze(0) * std + mean).clamp(0, 1)
        return t.mean(dim=0).cpu().numpy()

    gen_img = to_numpy(target_tensor)
    style_img = to_numpy(style_tensor)
    
    matched = match_histograms(gen_img, style_img)
    plt.imsave(filename, matched, cmap='gray')
    print(f"Saved: {filename}")


def evaluate_terrain(generated_tensor, real_style_tensor):
    print("Terrain Evaluation Metrics")
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(generated_tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(generated_tensor.device)

    gen_img = (generated_tensor.detach().squeeze(0) * std + mean).clamp(0, 1).mean(dim=0).cpu().numpy()
    real_img = (real_style_tensor.detach().squeeze(0) * std + mean).clamp(0, 1).mean(dim=0).cpu().numpy()

    #SSIM (Structural Similarity Index)
    ssim_score = ssim(real_img, gen_img, data_range=1.0)
    print(f"1. SSIM (Structural Match): {ssim_score:.4f}")
    
    #RMS Roughness (Standard Deviation)
    gen_roughness = np.std(gen_img)
    real_roughness = np.std(real_img)
    print(f"2. Roughness (StdDev): Gen {gen_roughness:.4f} vs Real {real_roughness:.4f}")
    
    #Slope Distribution Difference (Wasserstein Distance)
    gx_gen, gy_gen = np.gradient(gen_img)
    gx_real, gy_real = np.gradient(real_img)
        
    slope_gen = np.sqrt(gx_gen**2 + gy_gen**2).flatten()
    slope_real = np.sqrt(gx_real**2 + gy_real**2).flatten()
        
    w_distance = wasserstein_distance(slope_real, slope_gen)
    print(f"3. Slope Distribution Difference (Wasserstein): {w_distance:.4f}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vgg = models.vgg19(weights='VGG19_Weights.DEFAULT').features
    for i, layer in enumerate(vgg):
        if isinstance(layer, torch.nn.MaxPool2d):
            vgg[i] = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
    
    for param in vgg.parameters():
        param.requires_grad_(False) 
    vgg.to(device)

    content = load_image('./Content/content1.png').to(device)
    style = load_image('./Style/himalayas.jpg', shape=content.shape[-2:]).to(device)
    
    content_features = get_features(content, vgg)
    style_features = get_features(style, vgg)
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}
    content_target = content_features['conv4_2'].detach()

    target = content.clone().requires_grad_(True).to(device)

    style_weights = {
        'conv1_1': 0.05,  
        'conv2_1': 0.1,
        'conv3_1': 0.5,
        'conv4_1': 2.0,   
        'conv5_1': 5.0    
    }
    
    content_weight = 1e-3 
    style_weight = 1e9   
    tv_weight = 1e3    
    
    optimizer = optim.Adam([target], lr=0.01) 
    steps = 1000 
    
    print(f"Starting Optimization");

    for i in range(1, steps + 1):
        optimizer.zero_grad()
        target_features = get_features(target, vgg)

        c_loss = torch.mean((target_features['conv4_2'] - content_target)**2)
        
        s_loss = 0
        for layer in style_weights:
            t_feat = target_features[layer]
            s_feat = style_features[layer]
           
            t_gram = gram_matrix(t_feat)
            s_gram = style_grams[layer]
            _, d, h, w = t_feat.shape
            layer_s_loss = style_weights[layer] * torch.mean((t_gram - s_gram)**2)
            s_loss += layer_s_loss / (d * h * w)
            
            mean_loss = torch.mean((t_feat.mean(dim=(2,3)) - s_feat.mean(dim=(2,3)))**2)
            std_loss = torch.mean((t_feat.std(dim=(2,3)) - s_feat.std(dim=(2,3)))**2)
            s_loss += (mean_loss + std_loss) * 20.0 
        
        tv_loss = total_variation_loss(target)

        global_moment_loss = (torch.mean(target) - torch.mean(style))**2 + \
                            (torch.std(target) - torch.std(style))**2

        total_loss = (content_weight * c_loss) + \
                     (style_weight * s_loss) + \
                     (tv_weight * tv_loss) + \
                     (global_moment_loss * 1e6)
        
        total_loss.backward()
        optimizer.step()

        if i % 200 == 0:
            print(f'Step {i}/{steps} | Loss: {total_loss.item():.2f}')


    save_terrain(target, style, "GeneratedTerrain.png")
    evaluate_terrain(target, style)

if __name__ == "__main__":
    main()