from einops import rearrange

def pixel_shuffle(x, scale_factor=0.5):
    n, w, h, c = x.size()
    # N, W, H, C --> N, W, H * scale, C // scale
    x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
    # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
    x = x.permute(0, 2, 1, 3).contiguous()
    # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
    x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                int(c / (scale_factor * scale_factor)))

    x = x.permute(0, 2, 1, 3).contiguous()

    return x


def extract_feature_pre_adapter(vision_model, pixel_values):
    vit_embeds = vision_model(
        pixel_values         = pixel_values,
        output_hidden_states = False,
        return_dict          = True
    ).last_hidden_state

    vit_embeds = vit_embeds[:, 1:, :]
    
    h = w = int(vit_embeds.shape[1] ** 0.5)
    vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
    vit_embeds = pixel_shuffle(vit_embeds, scale_factor=0.5)
    vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])

    return vit_embeds

def extract_feature_pre_shuffle_adapter(vision_model, pixel_values):
    vit_embeds = vision_model(
        pixel_values         = pixel_values,
        output_hidden_states = False,
        return_dict          = True
    ).last_hidden_state

    vit_embeds = vit_embeds[:, 1:, :]

    return vit_embeds

def extract_both_clip(vision_model, pixel_values):
    """
    主要是抽取256x4096, 1024x1024是把256个打散，得到的实际是1, 2, 33, 34, ....
    """
    vit_embeds = vision_model(
        pixel_values         = pixel_values,
        output_hidden_states = False,
        return_dict          = True
    ).last_hidden_state

    # clip_1024 = vit_embeds[:, 1:, :].clone()

    vit_embeds = vit_embeds[:, 1:, :]
    h = w = int(vit_embeds.shape[1] ** 0.5)
    vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
    vit_embeds = pixel_shuffle(vit_embeds, scale_factor=0.5)
    clip_256 = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1]).clone()

    clip_1024 = rearrange(clip_256.clone(), "b t (s d) -> b (t s) d", s=4, d=1024) # (B, 1024, 1024)

    return clip_1024, clip_256

def extract_dual_clip(vision_model, pixel_values):
    vit_embeds = vision_model(
        pixel_values         = pixel_values,
        output_hidden_states = False,
        return_dict          = True
    ).last_hidden_state

    vit_embeds = vit_embeds[:, 1:, :]
    clip_1024 = vit_embeds.clone()
    h = w = int(vit_embeds.shape[1] ** 0.5)
    vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
    vit_embeds = pixel_shuffle(vit_embeds, scale_factor=0.5)
    clip_256 = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1]).clone()

    return clip_1024, clip_256