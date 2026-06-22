import torch
from torch.nn import CrossEntropyLoss
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import types
from transformers import AutoTokenizer, AutoModel, GenerationConfig, GenerationMixin
from .tree import Node, NodeA
from .utils import *

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
BOX_COLOR = "red"

IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'
IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'

def new_prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **model_kwargs
):
    past_length = 0
    if past_key_values is not None:
        if hasattr(past_key_values, "get_seq_length"):
            past_length = past_key_values.get_seq_length()
        elif isinstance(past_key_values, (list, tuple)) and past_key_values[0] is not None:
            past_length = past_key_values[0][0].shape[2]
        if past_length > 0:
            input_ids = input_ids[:, -1:]

    position_ids = model_kwargs.get("position_ids", None)
    if attention_mask is not None and position_ids is None:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_length > 0:
            position_ids = position_ids[:, past_length:]

    if inputs_embeds is not None and past_length == 0:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids}

    model_inputs.update(
        {
            "past_key_values": past_key_values,
            "use_cache": model_kwargs.get("use_cache"),
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
    )
    return model_inputs


def legacy_forward_wrapper(self, *args, **kwargs):
    pkv = kwargs.get('past_key_values', None)

    if pkv is not None and hasattr(pkv, "to_legacy_cache"):
        if pkv.get_seq_length() == 0:
            kwargs['past_key_values'] = None
        else:
            kwargs['past_key_values'] = pkv.to_legacy_cache()
    return self._original_forward(*args, **kwargs)

class ModelInternvl:
    def __init__(self, model_path: str, device: str = "cuda:0", torch_dtype=torch.bfloat16, **kwargs) -> None:
        self.device = device
        self.dtype = torch_dtype

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        self.model = AutoModel.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            device_map=self.device,
            trust_remote_code=True).eval()

        try:
            if hasattr(self.model, "language_model"):
                lm_model = self.model.language_model  # InternLM2ForCausalLM
                lm_class = lm_model.__class__

                print(f"Applying Runtime Patches to {lm_class.__name__}...")

                if not issubclass(lm_class, GenerationMixin):
                    lm_class.__bases__ = lm_class.__bases__ + (GenerationMixin,)
                    print(" - [1/4] GenerationMixin injected.")

                lm_class.prepare_inputs_for_generation = new_prepare_inputs_for_generation
                print(" - [2/4] prepare_inputs_for_generation replaced.")

                if hasattr(lm_model, "model"):
                    inner_model = lm_model.model  # InternLM2Model
                    if not hasattr(inner_model, "_original_forward"):
                        inner_model._original_forward = inner_model.forward
                        inner_model.forward = types.MethodType(legacy_forward_wrapper, inner_model)
                        print(" - [3/4] Inner model forward wrapper applied.")

                if not hasattr(lm_model, "generation_config") or lm_model.generation_config is None:
                    lm_model.generation_config = GenerationConfig.from_model_config(lm_model.config)
                    lm_model.generation_config.pad_token_id = self.tokenizer.pad_token_id
                    lm_model.generation_config.eos_token_id = self.tokenizer.eos_token_id
                    print(" - [4/4] generation_config initialized.")

        except Exception as e:
            print(f"Warning: Failed to apply Runtime patches: {e}")
            import traceback
            traceback.print_exc()

        self.model.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

        self.bias_value = kwargs.get("bias_value", 0.6)
        print("bias_value:", self.bias_value)

        self.input_size = (self.model.config.force_image_size, self.model.config.force_image_size)
        print("input size:", self.input_size)

        self.patch_scale = kwargs.get("patch_scale", None)
        print("patch scale:", self.patch_scale)

        self.generation_config = dict(max_new_tokens=1024, do_sample=False)

        self.anyres_num = kwargs.get("anyres_num", 12)
        print("anyres_num:", self.anyres_num)

        self.background_color = tuple(int(x * 255) for x in IMAGENET_MEAN)
        print("background color:", self.background_color)
        self.init_index_yes_no()
        self.init_prompts()

        self.view_size = 224
        self.scale_size = self.model.config.force_image_size

    def init_index_yes_no(self):
        print("Yes:", self.tokenizer("Yes").input_ids)
        print("No:", self.tokenizer("No").input_ids)
        if len(self.tokenizer("Yes").input_ids) == 1 and len(self.tokenizer("No").input_ids) == 1:
            self.index_yes = self.tokenizer("Yes").input_ids[0]
            self.index_no = self.tokenizer("No").input_ids[0]
        else:
            assert len(self.tokenizer("Yes").input_ids) == 2 and len(self.tokenizer("No").input_ids) == 2
            self.index_yes = self.tokenizer("Yes").input_ids[1]
            self.index_no = self.tokenizer("No").input_ids[1]
        print("index_yes:", self.index_yes)
        print("index_no:", self.index_no)

    def get_confidence_weight(self, node: Node, max_depth: int):
        coeff = (1 - self.bias_value) / (max_depth ** 2)
        return coeff * (node.depth ** 2) + self.bias_value

    def init_prompts(self):
        self.prompts = {
            "search": {
                "pre_information_image2": f"<image>\nThis is the main image, and the section enclosed by the {BOX_COLOR} rectangle is the focus region.\n<image>\nThis is the zoomed-in view of the focus region.\n",
                "pre_information_image1": "<image>\n",
                "latent_prompt": "According to your common sense knowledge and the content of image, is it possible to find a {} by further zooming in the image? Answer Yes or No.",
                "existence_prompt": "Is there a '{}' in the image? Answer Yes or No.",
                "answering_prompt": "Question: '{}'\nCould you answer the question based on the available visual information? Answer Yes or No.",
            }
        }

    def save_crop(self, image_pil, node, image_path):
        resized_bboxes = self.get_patch(node.state.bbox, image_pil.width, image_pil.height,patch_size=self.input_size[0], patch_scale=self.patch_scale)
        cropped_image = image_pil.crop(resized_bboxes)
        cropped_image.save(image_path)

    @torch.no_grad()
    def generate_visual_cues_using_ic(self, ic_examples, question: str, split_tag=r' and |, '):
        ic_question_template = ic_examples["question_template"]
        ic_question_list = ic_examples["question_list"]
        ic_response_list = ic_examples["response_list"]
        fs_prompt = []
        for q, a in zip(ic_question_list, ic_response_list):
            fs_prompt.append((ic_question_template.format(q), a))

        response = self.model.chat(self.tokenizer, None, ic_question_template.format(question), self.generation_config, history=fs_prompt)
        targets_sentence = extract_targets(response)
        if targets_sentence is not None:
            targets = split_targets_sentence(targets_sentence, split_tag)
        else:
            targets = []

        return targets

    def filter_visual_cues(self, image_pil: Image.Image, targets, root_node, decomposed_question_template,answering_confidence_threshold_upper, show_value=False):
        assert root_node.is_root
        ret = []
        if len(targets) > 1:
            for target in targets:
                if target.startswith("all "):
                    ret.append(target)
                    continue
                inputs = {
                    "node": root_node,
                    "image_pil": image_pil,
                    "confidence_type": "answering",
                    "input_ele": decomposed_question_template.format(target)
                }
                inputs['root_anyres'] = False
                conf_root = self.get_confidence_value(**inputs)
                if show_value:
                    print(target, conf_root)
                if conf_root < answering_confidence_threshold_upper:
                    ret.append(target)
        else:
            ret = targets[:]
        return ret

    def get_prompt_from_qs(self, qs, response=None, show_prompt=False):
        template = self.model.conv_template.copy()
        template.system_message = self.model.system_message
        eos_token_id = self.tokenizer.convert_tokens_to_ids(template.sep.strip())
        template.append_message(template.roles[0], qs)
        template.append_message(template.roles[1], response)
        prompt = template.get_prompt()
        return prompt

    def get_prompt_tag(self, image_list):
        if len(image_list) == 1:
            prompt_tag = "global"
        elif len(image_list) == 2:
            prompt_tag = "zoom"
        else:
            raise ValueError
        return prompt_tag

    def is_root_only(self, nodes: List[Node]):
        return (len(nodes) == 1 and nodes[0].is_root)

    def include_root(self, nodes: List[Node]):
        return any(node.is_root for node in nodes)

    def get_bbox_in_square_image(self, bbox, left, top):
        x1, y1, x2, y2 = bbox
        return [x1 + left, y1 + top, x2 + left, y2 + top]

    def draw_bbox_arrow_in_square_image(self, square_image, resized_bbox, color):
        thickness = square_image.width // 120
        new_bbox = visualize_bbox_and_arrow(square_image, resized_bbox, color, thickness, xyxy=True)
        return new_bbox

    def zoom_in(self, croped_view):
        c_width = croped_view.width
        c_height = croped_view.height
        resize_ratio = min(self.scale_size * 1.0 / c_width, self.scale_size * 1.0 / c_height)
        resize_ratio = max(1.0, resize_ratio)
        zoomed_view = croped_view.resize((int(c_width * resize_ratio), int(c_height * resize_ratio)))

        return zoomed_view

    def get_patch(self, bbox, image_width, image_height, patch_size, patch_scale=None):
        object_width = int(np.ceil(bbox[2]))
        object_height = int(np.ceil(bbox[3]))

        object_center_x = int(bbox[0] + bbox[2] / 2)
        object_center_y = int(bbox[1] + bbox[3] / 2)

        patch_width = max(object_width, patch_size)
        patch_height = max(object_height, patch_size)
        if patch_scale is not None:
            patch_width = int(patch_width * patch_scale)
            patch_height = int(patch_height * patch_scale)

        left = max(0, object_center_x - patch_width // 2)
        right = min(left + patch_width, image_width)

        top = max(0, object_center_y - patch_height // 2)
        bottom = min(top + patch_height, image_height)

        return [left, top, right, bottom]

    def process_nodes_to_image_list(self, nodes, image_pil, root_anyres=True):
        square_image, left, top = expand2square(image_pil, self.background_color)
        if self.is_root_only(nodes):
            return [deepcopy(image_pil)] if root_anyres else [square_image.resize(self.input_size)]
        if len(nodes) == 0 or self.include_root(nodes):
            return [deepcopy(image_pil)]

        resized_bboxes = []
        for node in nodes:
            source = getattr(node, 'search_source', 'fine')
            if source == 'fast':
                current_patch_size = self.view_size
                current_patch_scale = None
            else:
                current_patch_size = self.view_size
                current_patch_scale = self.patch_scale

            patch_bbox = self.get_patch(
                bbox=node.state.bbox,
                image_width=image_pil.width,
                image_height=image_pil.height,
                patch_size=current_patch_size,
                patch_scale=current_patch_scale
            )
            resized_bboxes.append(patch_bbox)

        resized_bboxes = merge_bbox_list(resized_bboxes, threshold=0)

        full_color_bboxes = []
        for i in range(len(resized_bboxes)):
            resized_bbox = self.get_bbox_in_square_image(resized_bboxes[i], left, top)
            color_bbox = self.draw_bbox_arrow_in_square_image(square_image, resized_bbox, BOX_COLOR)
            full_color_bboxes.append(color_bbox)

        union_color_bboxes = union_all_bboxes(full_color_bboxes)
        if union_color_bboxes is None:
            return [square_image]
        croped_view = square_image.crop(union_color_bboxes)
        zoomed_view = self.zoom_in(croped_view)

        return [square_image.resize(self.input_size), croped_view, zoomed_view]

    @torch.no_grad()
    def get_confidence_value(self, node: List[Node], image_pil: Image.Image, confidence_type: str, input_ele,root_anyres=True):

        assert confidence_type in ['existence', 'latent', 'answering']
        image_list = self.process_nodes_to_image_list(node, image_pil, root_anyres=root_anyres)

        if len(image_list) > 1:
            raw_image_resize, croped_view, zoom_view = image_list
            image_input = [raw_image_resize, zoom_view]
        else:
            image_input = image_list

        # Process images
        full_pixel_values = []
        num_patches_list = []
        for i, img in enumerate(image_input):
            if i == 0:
                use_anyres = root_anyres if len(image_input) == 1 else False
            else:
                use_anyres = True
            pixel_values = load_image(img, input_size=self.input_size[0], max_num=self.anyres_num,use_anyres=use_anyres)
            full_pixel_values.append(pixel_values.to(self.dtype))
            num_patches_list.append(pixel_values.shape[0])
        full_pixel_values = torch.cat(full_pixel_values, dim=0)

        prompt_tag = "search"
        if len(image_input) > 1:
            pre_information = "pre_information_image2"
        else:
            pre_information = "pre_information_image1"

        qs = self.prompts[prompt_tag][pre_information] + self.prompts[prompt_tag][f"{confidence_type}_prompt"].format(input_ele)
        prompt = self.get_prompt_from_qs(qs)

        image_idx = 0
        for num_patches in num_patches_list:
            while "<image>" in prompt:
                num_patches = num_patches_list[image_idx]
                image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.model.num_image_token * num_patches + IMG_END_TOKEN
                prompt = prompt.replace("<image>", image_tokens, 1)
                image_idx += 1
        assert image_idx == len(num_patches_list)

        model_inputs = self.tokenizer(
            [prompt],
            return_tensors='pt',
            padding=True,
            padding_side='left',
            add_special_tokens=True,
        )
        model_inputs["pixel_values"] = full_pixel_values
        model_inputs['image_flags'] = torch.ones(full_pixel_values.shape[0], dtype=torch.long)

        for k, v in model_inputs.items():
            model_inputs[k] = v.to(self.device)
        outputs = self.model(**model_inputs)

        return self._cal_confidence(outputs)

    @torch.no_grad()
    def _cal_confidence(self, outputs):
        logits_yesno = outputs.logits[0, -1, [self.index_yes, self.index_no]]
        confidence = torch.softmax(logits_yesno, dim=-1)[0]
        confidence = 2 * (confidence.item() - 0.5)  # [-1, 1]
        return confidence

    @torch.inference_mode()
    def free_form_using_nodes(self, image_pil, question, searched_nodes: List[Node], return_zoomed_view=False):
        image_list = self.process_nodes_to_image_list(searched_nodes, image_pil)
        if len(image_list) > 1:
            raw_image_resize, croped_view, zoom_view = image_list
            image_input = [raw_image_resize, zoom_view]
        else:
            image_input = image_list

        prompt_tag = "search"
        if len(image_input) > 1:
            pre_information = "pre_information_image2"
        else:
            pre_information = "pre_information_image1"

        qs = self.prompts[prompt_tag][pre_information] + question

        # Process images
        full_pixel_values = []
        num_patches_list = []
        for i, img in enumerate(image_input):
            if i == 0:
                use_anyres = True if len(image_input) == 1 else False
            else:
                use_anyres = True
            pixel_values = load_image(img, input_size=self.input_size[0], max_num=self.anyres_num,use_anyres=use_anyres)
            full_pixel_values.append(pixel_values.to(self.dtype))
            num_patches_list.append(pixel_values.shape[0])
        full_pixel_values = torch.cat(full_pixel_values, dim=0)
        full_pixel_values = full_pixel_values.to(dtype=self.dtype, device=self.device)
        outputs = self.model.chat(self.tokenizer, full_pixel_values, qs, self.generation_config)
        if return_zoomed_view:
            response = {'text': outputs, 'output_image': image_input[1] if len(image_input) > 1 else image_input[0]}
            return response

        return outputs

    @torch.inference_mode()
    def multiple_choices_inference(self, image_pil, question, options, searched_nodes: List[Node] = None):
        image_list = self.process_nodes_to_image_list(searched_nodes, image_pil)

        if len(image_list) > 1:
            raw_image_resize, croped_view, zoom_view = image_list
            image_input = [raw_image_resize, zoom_view]
        else:
            image_input = image_list

        prompt_tag = "search"
        if len(image_input) > 1:
            pre_information = "pre_information_image2"
        else:
            pre_information = "pre_information_image1"

        qs = self.prompts[prompt_tag][pre_information] + question
        prompt = self.get_prompt_from_qs(qs)
        # Process images
        full_pixel_values = []
        num_patches_list = []
        for i, img in enumerate(image_input):
            if i == 0:
                use_anyres = True if len(image_input) == 1 else False
            else:
                use_anyres = True
            pixel_values = load_image(img, input_size=self.input_size[0], max_num=self.anyres_num,use_anyres=use_anyres)
            full_pixel_values.append(pixel_values.to(self.dtype))
            num_patches_list.append(pixel_values.shape[0])
        full_pixel_values = torch.cat(full_pixel_values, dim=0)

        image_idx = 0
        for num_patches in num_patches_list:
            while "<image>" in prompt:
                num_patches = num_patches_list[image_idx]
                image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.model.num_image_token * num_patches + IMG_END_TOKEN
                prompt = prompt.replace("<image>", image_tokens, 1)
                image_idx += 1
        assert image_idx == len(num_patches_list)

        model_inputs = self.tokenizer(
            [prompt],
            return_tensors='pt',
            padding=True,
            padding_side='left',
            add_special_tokens=True,
        )
        model_inputs["pixel_values"] = full_pixel_values
        model_inputs['image_flags'] = torch.ones(full_pixel_values.shape[0], dtype=torch.long)

        for k, v in model_inputs.items():
            model_inputs[k] = v.to(self.device)

        question_input_ids = model_inputs.input_ids
        len_question_input_ids = question_input_ids.shape[1]

        output_question = self.model(**model_inputs)
        len_question_logits = output_question.logits.shape[1]

        loss_list = []

        for option in options:
            full_prompt = self.get_prompt_from_qs(qs, option)
            image_idx = 0
            for num_patches in num_patches_list:
                while "<image>" in full_prompt:
                    num_patches = num_patches_list[image_idx]
                    image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.model.num_image_token * num_patches + IMG_END_TOKEN
                    full_prompt = full_prompt.replace("<image>", image_tokens, 1)
                    image_idx += 1
            assert image_idx == len(num_patches_list)

            full_model_inputs = self.tokenizer(
                [full_prompt],
                return_tensors='pt',
                padding=True,
                padding_side='left',
                add_special_tokens=True,
            )
            full_model_inputs["pixel_values"] = full_pixel_values
            full_model_inputs['image_flags'] = torch.ones(full_pixel_values.shape[0], dtype=torch.long)
            for k, v in full_model_inputs.items():
                full_model_inputs[k] = v.to(self.device)
            full_input_ids = full_model_inputs.input_ids

            output_option = self.model(**full_model_inputs)
            logits = output_option.logits[:, len_question_logits - 1:-1]

            loss_fct = CrossEntropyLoss()
            logits = logits.view(-1, self.model.config.llm_config.vocab_size)
            labels = full_input_ids[:, len_question_input_ids:].view(-1)
            # print(logits.shape, labels.shape)
            loss = loss_fct(logits, labels)
            loss_list.append(loss)

        option_chosen = torch.stack(loss_list).argmin()

        return option_chosen.cpu().item()


def load_image(image: Image.Image, input_size: int = 448, max_num: int = 12, use_anyres: bool = True):
    transform = build_transform(input_size=input_size)
    if use_anyres:
        images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    else:
        images = [image.resize((input_size, input_size))]
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images
