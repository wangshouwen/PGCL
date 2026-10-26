import math
import torch
import torch.nn as nn

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from copy import deepcopy
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from .encoder import Transformer

_tokenizer = _Tokenizer()

def load_clip_to_cpu(args):
    backbone_name = args.backbone
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model = clip.build_model_conv_proj(state_dict or model.state_dict(), args)

    return model


class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model, class_spec=False):
        super().__init__()
        n_cls = len(classnames)
        ctx_init_pos = "There is a X in the scene"
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        
        # use given words to initialize context vectors
        ctx_init_pos = ctx_init_pos.replace("_", " ")
        n_ctx_pos = len(ctx_init_pos.split(" "))
        prompt_pos = clip.tokenize(ctx_init_pos)
        prompt_pos = torch.cat([prompt_pos[:,:4], prompt_pos[:,5:]], dim=1)
        with torch.no_grad():
            embedding_pos = clip_model.token_embedding(prompt_pos).type(dtype)
        ctx_vectors_pos = embedding_pos[0, 1:n_ctx_pos, :]

        prompt_prefix_pos = ctx_init_pos
        if class_spec:
            ctx_vectors_pos_ = []
            for _ in range(n_cls):
                ctx_vectors_pos_.append(deepcopy(ctx_vectors_pos))
            ctx_vectors_pos = torch.stack(ctx_vectors_pos_, dim=0)

        print(f'Initial positive context: "{prompt_prefix_pos}"')
        print(f"Number of positive context words (tokens): {n_ctx_pos-1}")

        self.ctx_pos = nn.Parameter(ctx_vectors_pos)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts_pos = [prompt_prefix_pos.replace("X", name) for name in classnames]

        tokenized_prompts_pos = []
        for p_pos in prompts_pos:
            tokenized_prompts_pos.append(clip.tokenize(p_pos))
        tokenized_prompts_pos = torch.cat(tokenized_prompts_pos)
        with torch.no_grad():
            embedding_pos = clip_model.token_embedding(tokenized_prompts_pos).type(dtype)

        self.token_prefix = embedding_pos[:, :1, :]
        token_class = []
        token_suffix = []
        for i, embed in enumerate(embedding_pos):
            each_class = embed[4:4+name_lens[i]]
            each_suffix = embed[7+name_lens[i]:]
            token_class.append(each_class)
            token_suffix.append(each_suffix)

        self.token_class = token_class
        self.token_suffix = token_suffix

        self.n_cls = n_cls
        self.tokenized_prompts = tokenized_prompts_pos
        self.name_lens = name_lens

    def forward(self):
        ctx_pos = self.ctx_pos
        device = ctx_pos.device

        prompts = []
        if ctx_pos.dim() == 2:
            ctx_pos = ctx_pos.unsqueeze(0).expand(self.n_cls, -1, -1)

        for i in range(self.n_cls):

            prompts_pos = torch.cat(
                [
                    self.token_prefix[i].to(device),  # (n_cls, 1, dim)
                    ctx_pos[i][:3],  # (n_cls, n_ctx, dim)
                    self.token_class[i].to(device),
                    ctx_pos[i][3:],
                    self.token_suffix[i].to(device)  # (n_cls, *, dim)
                ],
                dim=0,
            )
            prompts.append(prompts_pos)

        prompts = torch.stack(prompts, dim=0)

        tokenized_prompts = self.tokenized_prompts

        return prompts, tokenized_prompts


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class Element_Wise_Layer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Element_Wise_Layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(in_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        for i in range(self.in_features):
            self.weight[i].data.uniform_(-stdv, stdv)
        if self.bias is not None:
            for i in range(self.in_features):
                self.bias[i].data.uniform_(-stdv, stdv)


    def forward(self, input):
        x = input * self.weight
        x = torch.sum(x,2)
        if self.bias is not None:
            x = x + self.bias
        return x

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None)


class PGCL(nn.Module):
    def __init__(self, args, classnames, clip_model, proj=False, low_dim=128):
        super().__init__()
        self.visual_encoder_type = args.backbone
        self.prompt_learner = PromptLearner(classnames=classnames, clip_model=clip_model, class_spec=True)

        self.image_encoder = clip_model.visual
        fea_dim = self.image_encoder.embed_dim
        self.text_encoder = TextEncoder(clip_model=clip_model)
        self.transformer = Transformer(d_model=fea_dim, nhead=4, dim_ratio=4, dropout_p=0.1,  #dp=0.1
                                       activation="gelu")

        self.dtype = clip_model.dtype
        self.proj = proj
        
        self.classifiers = Element_Wise_Layer(len(classnames), fea_dim)
        if self.proj:
            self.pfc1 = nn.Linear(fea_dim, fea_dim)
            self.pfc2 = nn.Linear(fea_dim, low_dim)

        # self.step = 0

    def prompt_learn_forward(self):
        prompts, tokenized_prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, tokenized_prompts)

        return text_features
    
    def forward(self, image, visualize=False, tsne=False):
        image_features, attn_features, attn_weight = self.image_encoder(image.type(self.dtype))

        self.text_features = self.prompt_learn_forward() #learn prompts
        attn_features = attn_features[:, :, 1:]
        text_features = self.text_features

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        attn_features_norm = attn_features / attn_features.norm(dim=1, keepdim=True)

        b_size = attn_features.size(0)
        text_features = text_features.repeat(b_size, 1, 1)
        attn = torch.bmm(text_features, attn_features_norm) #BxCxHW

        coeff = 10 * attn
        coeff = F.softmax(coeff, dim=-1)

        pos_embed = self.image_encoder.attnpool.positional_embedding[1:]
        class_features = self.transformer(image_features, coeff, text_features, pos_embed)

        tsne_features = class_features

        logits = self.classifiers(class_features)

        if tsne is True:
            return logits, tsne_features

        if self.proj and self.training == True:
            class_features = self.pfc1(class_features)
            class_features = F.relu(class_features)
            class_features = self.pfc2(class_features)
            class_features = class_features / class_features.norm(dim=-1, keepdim=True)
            if visualize is True:
                return logits, class_features, coeff
            return logits, class_features
        
        if visualize is True:
            return logits, None, coeff

        return logits
    
    def backbone_params(self):
        params1 = []
        params2 = []
        for name, param in self.named_parameters():
            if "image_encoder" in name and "attnpool" not in name:
                if "layer4" not in name:
                    param.requires_grad = False
                else:
                    params1.append(param)
            elif "image_encoder" in name and "attnpool" in name:
                params2.append(param)

        return params1, params2
    
    def prompt_params(self):
        params = []
        for name, param in self.named_parameters():
            if "prompt_learner" in name:
                params.append(param)

        return params
    
    def tuned_params(self):
        """Parameters handed to the optimizer alongside the backbone groups.

        NOTE: the "transformer" test matches both this model's decoder
        (``transformer.*``) AND CLIP's text transformer
        (``text_encoder.transformer.*``), so the text encoder is fine-tuned too.
        This is what the published results were produced with; if you want the
        text encoder frozen, match ``name.startswith("transformer.")`` instead.
        """
        params = []
        for name, param in self.named_parameters():
            if "classifiers" in name or "pfc" in name:
                params.append(param)
            elif "transformer" in name:
                params.append(param)

        return params
    

def build_PGCL_model(args, classnames, **kwargs):
    print(f"Loading CLIP (backbone: {args.backbone})")
    clip_model = load_clip_to_cpu(args)
    
    clip_model.float()

    print("Building model")
    model = PGCL(args, classnames, clip_model, proj=args.proj, low_dim=128)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.to(device)

    return model