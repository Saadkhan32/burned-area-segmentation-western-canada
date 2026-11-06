"""Model zoo for the benchmark pipeline.
Five architectures of the original study, re-implemented in plain PyTorch so that they run on the Alliance clusters:
  unet, deeplabv3, pspnet : segmentation_models_pytorch, ResNet-50 encoder (ImageNet weights), 9 input bands
  samlora                 : Segment Anything ViT-B image encoder (SA-1B weights, --sam-checkpoint) with LoRA on the
                            attention qkv projections + a light convolutional decoder; input resized to 1024 as in arcgis.learn
  mask2former             : Hugging Face Mask2Former (Swin-L, ADE20K-pretrained) with the patch embedding widened to 9 bands
Newer models (Set 3, chosen from the 2024-2026 burned-area literature, see 03_literature/newer_models_2026-10-13.md):
  deeplabv3plus          : SMP DeepLabV3+ ResNet-50 (the decoder variant most burned-area papers use)
  segformer              : SMP SegFormer MiT-B2 (ImageNet) - CaBuAr, CanadaFire2023 and others benchmark it
  upernet_convnext       : SMP UPerNet with a ConvNeXt-Tiny encoder (modern CNN)
  prithvi / prithvi_lora : Prithvi-EO-2.0-300M (NASA/IBM HLS foundation model, 6 bands Blue..SWIR2) with a UNet decoder via TerraTorch,
                           full fine-tuning or LoRA r=8 on the qkv projections (the recipe of Shibli et al. 2026)
  terramind              : TerraMind-v1-base (ESA/IBM) with the S2L2A modality restricted to the same 6 bands, UNet decoder via TerraTorch
  dinov3_lora            : DINOv3 ViT-L/16 pretrained on SAT-493M (Meta, gated on Hugging Face - accept the licence and set HF_TOKEN),
                           patch embedding widened to 9 bands, LoRA r=8 on q/v projections, multi-layer feature pyramid + light decoder
Every model returns per-pixel class logits (B, 2, H, W) at the input resolution; Mask2Former's class/mask queries are
collapsed to semantic logits (log of the query-weighted class probabilities)."""
import math
import torch, torch.nn as nn, torch.nn.functional as F

N_CLASSES = 2


def widen_first_conv(conv, in_ch):
    """Replace a 3-channel conv with an in_ch-channel one; copy the RGB weights to bands Red, Green, Blue (indices 2, 1, 0),
    initialise the rest with the mean RGB filter (the same idea as segmentation_models_pytorch's in_channels adaptation)."""
    new = nn.Conv2d(in_ch, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, conv.dilation, conv.groups, conv.bias is not None)
    with torch.no_grad():
        w = conv.weight
        mean = w.mean(1, keepdim=True)
        new.weight.copy_(mean.repeat(1, in_ch, 1, 1))
        if w.shape[1] == 3 and in_ch >= 3:
            new.weight[:, 2] = w[:, 0]; new.weight[:, 1] = w[:, 1]; new.weight[:, 0] = w[:, 2]   # RGB -> (Blue, Green, Red) positions
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    return new


def widen_input_conv(module, in_ch):
    """Find the first nn.Conv2d with three input channels inside `module` (the RGB patch-embedding / stem convolution, wherever the
    library version keeps it - transformers 4.x exposes it as SwinBackbone.embeddings..., 5.x as SwinBackbone.swin.embeddings...),
    replace it in place with a widened copy (widen_first_conv) and return the new conv. A `num_channels` attribute on the owning
    submodule is updated too (older transformers versions check it in forward)."""
    for name, mod in module.named_modules():
        for cname, child in mod.named_children():
            if isinstance(child, nn.Conv2d) and child.in_channels == 3:
                new = widen_first_conv(child, in_ch)
                setattr(mod, cname, new)
                if hasattr(mod, 'num_channels'):
                    mod.num_channels = in_ch
                return new
    raise RuntimeError('widen_input_conv: no 3-channel Conv2d found in ' + type(module).__name__)


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank=8, alpha=16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.A = nn.Linear(base.in_features, rank, bias=False)
        self.B = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5)); nn.init.zeros_(self.B.weight)
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + self.B(self.A(x)) * self.scale


class SamLoRASeg(nn.Module):
    """SAM ViT-B image encoder + LoRA (rank r on every attention qkv) + conv decoder. Input (B, 9, h, w) is resized to 1024 x 1024;
    the encoder embedding (256 x 64 x 64) is decoded and upsampled back to (h, w)."""
    def __init__(self, in_ch=9, checkpoint=None, rank=8, img_size=1024):
        super().__init__()
        from segment_anything import sam_model_registry
        sam = sam_model_registry['vit_b'](checkpoint=checkpoint)
        enc = sam.image_encoder
        enc.patch_embed.proj = widen_first_conv(enc.patch_embed.proj, in_ch)
        for p in enc.parameters():
            p.requires_grad = False
        for p in enc.patch_embed.proj.parameters():
            p.requires_grad = True
        for blk in enc.blocks:
            blk.attn.qkv = LoRALinear(blk.attn.qkv, rank=rank)
        if img_size != 1024:   # SAM's absolute position embedding is fixed at 64 x 64 patches; resample it for other input sizes
            s = img_size // 16
            pe = enc.pos_embed.detach().permute(0, 3, 1, 2)
            pe = F.interpolate(pe, size=(s, s), mode='bicubic', align_corners=False).permute(0, 2, 3, 1)
            enc.pos_embed = nn.Parameter(pe, requires_grad=False)
        self.encoder = enc; self.img_size = img_size
        self.decoder = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                                     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                     nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                     nn.Conv2d(64, N_CLASSES, 1))

    def forward(self, x):
        h, w = x.shape[-2:]
        x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        f = self.encoder(x)                       # (B, 256, 64, 64)
        y = self.decoder(f)                        # (B, 2, 256, 256)
        return F.interpolate(y, size=(h, w), mode='bilinear', align_corners=False)


class Mask2FormerSeg(nn.Module):
    """Hugging Face Mask2Former for semantic segmentation with 9-band input. Training uses the native Hungarian-matched
    loss (class + mask + dice) through the `labels` path; inference collapses queries to per-pixel class probabilities."""
    def __init__(self, in_ch=9, pretrained=True, name='facebook/mask2former-swin-large-ade-semantic', tiny=False):
        super().__init__()
        from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerConfig
        if tiny:  # smoke tests only: random small Swin
            cfg = Mask2FormerConfig(num_labels=N_CLASSES, num_queries=20, hidden_dim=64, feature_size=64, mask_feature_size=64,
                                    decoder_layers=2, encoder_layers=2, dim_feedforward=128, num_attention_heads=4, encoder_feedforward_dim=128,
                                    backbone_config={'model_type': 'swin', 'embed_dim': 32, 'depths': [1, 1, 1, 1], 'num_heads': [1, 2, 4, 8],
                                                     'out_features': ['stage1', 'stage2', 'stage3', 'stage4'], 'num_channels': in_ch, 'image_size': 64})
            self.model = Mask2FormerForUniversalSegmentation(cfg)
        else:
            if pretrained:
                self.model = Mask2FormerForUniversalSegmentation.from_pretrained(name, num_labels=N_CLASSES, ignore_mismatched_sizes=True)
            else:   # same Swin-L architecture, random initialisation: eval.py builds the model this way and then loads best.pt
                cfg = Mask2FormerConfig.from_pretrained(name); cfg.num_labels = N_CLASSES
                self.model = Mask2FormerForUniversalSegmentation(cfg)
            widen_input_conv(self.model.model.pixel_level_module.encoder, in_ch)   # Swin patch embedding, wherever this transformers version keeps it
            bc = getattr(self.model.config, 'backbone_config', None)
            if bc is not None and hasattr(bc, 'num_channels'):
                bc.num_channels = in_ch
        self.model.config.num_labels = N_CLASSES

    def forward(self, x, mask_labels=None, class_labels=None):
        out = self.model(pixel_values=x, mask_labels=mask_labels, class_labels=class_labels)
        cls = out.class_queries_logits.softmax(-1)[..., :-1]              # (B, Q, C) drop the no-object class
        masks = out.masks_queries_logits.sigmoid()                        # (B, Q, h', w')
        seg = torch.einsum('bqc,bqhw->bchw', cls, masks)
        seg = F.interpolate(seg, size=x.shape[-2:], mode='bilinear', align_corners=False)
        logits = torch.log(seg.clamp_min(1e-6))
        return (logits, out.loss) if mask_labels is not None else logits


PRITHVI_BANDS = ['BLUE', 'GREEN', 'RED', 'NIR_NARROW', 'SWIR_1', 'SWIR_2']   # our bands 0..5 (B08 stands in for B8A)


class TerraTorchSeg(nn.Module):
    """Prithvi-EO-2.0 / TerraMind through TerraTorch's EncoderDecoderFactory; takes the first six of our nine bands."""
    def __init__(self, backbone, pretrained=True, lora=False, img_size=256):
        super().__init__()
        from terratorch.models import EncoderDecoderFactory
        f = EncoderDecoderFactory()
        necks = [{'name': 'SelectIndices', 'indices': [5, 11, 17, 23] if '300' in backbone or '600' in backbone else [2, 5, 8, 11]},
                 {'name': 'ReshapeTokensToImage'}, {'name': 'LearnedInterpolateToPyramidal'}]
        kw = dict(task='segmentation', backbone=backbone, backbone_pretrained=pretrained, decoder='UNetDecoder',
                  decoder_channels=[512, 256, 128, 64], num_classes=N_CLASSES, necks=necks)
        if backbone.startswith('terramind'):
            kw.update(backbone_modalities=['S2L2A'], backbone_bands={'S2L2A': PRITHVI_BANDS})
        else:
            kw.update(backbone_bands=PRITHVI_BANDS, backbone_img_size=img_size)
        if lora:
            kw['peft_config'] = {'method': 'LORA', 'peft_config_kwargs': {'r': 8, 'lora_alpha': 16, 'target_modules': ['qkv'], 'lora_dropout': 0.05}}
        self.model = f.build_model(**kw)

    def forward(self, x):
        return self.model(x[:, :6]).output


class DINOv3LoRASeg(nn.Module):
    """DINOv3 ViT-L/16 (SAT-493M) backbone with LoRA on q/v projections, widened patch embedding, 4-level feature pyramid
    (layers at 1/4, 1/2, 3/4, end), fused by a light FPN-style decoder to 1/4 resolution and upsampled."""
    def __init__(self, in_ch=9, pretrained=True, name='facebook/dinov3-vitl16-pretrain-sat493m', rank=8, tiny=False):
        super().__init__()
        import transformers
        if tiny:
            cfg = transformers.DINOv3ViTConfig(hidden_size=64, num_hidden_layers=4, num_attention_heads=4, intermediate_size=128, patch_size=16, num_channels=in_ch, image_size=256, num_register_tokens=4)
            self.vit = transformers.DINOv3ViTModel(cfg)
            patch_params = list(self.vit.embeddings.patch_embeddings.parameters())
        else:
            if pretrained:
                self.vit = transformers.DINOv3ViTModel.from_pretrained(name)
            else:   # same architecture without weights (eval.py loads best.pt afterwards)
                self.vit = transformers.DINOv3ViTModel(transformers.DINOv3ViTConfig.from_pretrained(name))
            patch_params = list(widen_input_conv(self.vit, in_ch).parameters()); self.vit.config.num_channels = in_ch
        for p in self.vit.parameters():
            p.requires_grad = False
        for p in patch_params:
            p.requires_grad = True
        layers = self.vit.layer if hasattr(self.vit, 'layer') else self.vit.model.layer
        for layer in layers:
            layer.attention.q_proj = LoRALinear(layer.attention.q_proj, rank=rank)
            layer.attention.v_proj = LoRALinear(layer.attention.v_proj, rank=rank)
        d = self.vit.config.hidden_size; L = self.vit.config.num_hidden_layers
        self.taps = sorted({max(1, L // 4), max(1, L // 2), max(1, 3 * L // 4), L})
        self.patch = self.vit.config.patch_size; self.n_special = 1 + self.vit.config.num_register_tokens
        self.lateral = nn.ModuleList([nn.Conv2d(d, 256, 1) for _ in self.taps])
        self.decoder = nn.Sequential(nn.Conv2d(256 * len(self.taps), 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                                     nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
                                     nn.Conv2d(256, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, N_CLASSES, 1))

    def forward(self, x):
        B, C, H, W = x.shape
        out = self.vit(pixel_values=x, output_hidden_states=True)
        h, w = H // self.patch, W // self.patch
        feats = []
        for conv, t in zip(self.lateral, self.taps):
            tok = out.hidden_states[t][:, self.n_special:, :]             # drop CLS + register tokens
            feats.append(conv(tok.transpose(1, 2).reshape(B, -1, h, w)))
        y = self.decoder(torch.cat(feats, 1))
        return F.interpolate(y, size=(H, W), mode='bilinear', align_corners=False)


def build_model(name, in_ch=9, pretrained=True, sam_checkpoint=None, tiny=False):
    import segmentation_models_pytorch as smp
    w = 'imagenet' if pretrained else None
    if name == 'deeplabv3plus':
        return smp.DeepLabV3Plus('resnet50', encoder_weights=w, in_channels=in_ch, classes=N_CLASSES)
    if name == 'prithvi':
        return TerraTorchSeg('prithvi_eo_v2_300', pretrained=pretrained, lora=False)
    if name == 'prithvi_lora':
        return TerraTorchSeg('prithvi_eo_v2_300', pretrained=pretrained, lora=True)
    if name == 'terramind':
        return TerraTorchSeg('terramind_v1_base', pretrained=pretrained, lora=False)
    if name == 'dinov3_lora':
        return DINOv3LoRASeg(in_ch, pretrained=pretrained, tiny=tiny)
    if name == 'unet':
        return smp.Unet('resnet50', encoder_weights=w, in_channels=in_ch, classes=N_CLASSES)
    if name == 'deeplabv3':
        return smp.DeepLabV3('resnet50', encoder_weights=w, in_channels=in_ch, classes=N_CLASSES)
    if name == 'pspnet':
        return smp.PSPNet('resnet50', encoder_weights=w, in_channels=in_ch, classes=N_CLASSES)
    if name == 'segformer':
        return smp.Segformer('mit_b2', encoder_weights=w, in_channels=in_ch, classes=N_CLASSES)
    if name == 'upernet_convnext':
        return smp.UPerNet('tu-convnext_tiny', encoder_weights=w, in_channels=in_ch, classes=N_CLASSES)
    if name == 'samlora':
        return SamLoRASeg(in_ch, checkpoint=sam_checkpoint if pretrained else None, img_size=256 if tiny else 1024)
    if name == 'mask2former':
        return Mask2FormerSeg(in_ch, pretrained=pretrained, tiny=tiny)
    raise ValueError(name)


def mask2former_targets(labels, ignore=255):
    """Semantic label batch (B, H, W) -> lists of mask_labels (K, H, W) float and class_labels (K,) long per image."""
    mask_labels, class_labels = [], []
    for lab in labels:
        present = [c for c in range(N_CLASSES) if (lab == c).any()]
        if not present:
            present = [0]
        mask_labels.append(torch.stack([(lab == c).float() for c in present]))
        class_labels.append(torch.tensor(present, dtype=torch.long, device=lab.device))
    return mask_labels, class_labels
