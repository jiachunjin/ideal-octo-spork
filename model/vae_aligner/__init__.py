from .vit_vae_aligner import ViTVAEAligner

def get_vae_aligner(config):
    vae_aligner = ViTVAEAligner(config)
    
    return vae_aligner